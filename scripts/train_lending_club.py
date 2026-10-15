"""Pipeline complet du modèle de remboursement anticipé Lending Club.

Enchaîne, de bout en bout et de façon reproductible :

1. chargement et préparation des données (features à l'émission, cible à 12 mois) ;
2. découpe temporelle référence (2013-2015) / surveillance (2016-2017) ;
3. **audit de fuite de données** (liste noire, découpe temporelle, pouvoir
   discriminant unitaire, validation adversariale) ;
4. **benchmark** de trois candidats avec validation hors temps sur 2016 ;
5. sélection du champion et scoring : référence avec scores hors-pli, surveillance
   avec le modèle figé ;
6. écriture des sorties pour le monitoring et la documentation.

Sorties (par défaut sous ``data/processed/lending_club`` et ``reports``) :

* ``reference.csv`` / ``current.csv`` : fenêtres scorées, prêtes pour ``alm-monitor`` ;
* ``model_card.json`` : fiche modèle (features, périodes, métriques, justification) ;
* ``benchmark.json`` et ``leakage_audit.json`` : traces du benchmark et de l'audit ;
* ``models/prepayment_champion.joblib`` : champion sérialisé (import Dataiku / MLflow).

Usage::

    python scripts/train_lending_club.py --raw data/raw/accepted_2007_to_2018Q4.csv.gz
    python scripts/train_lending_club.py --raw data/sample/lending_club_sample.csv.gz --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from alm_monitoring import leakage
from alm_monitoring.data import lending_club as lc
from alm_monitoring.exceptions import AlmMonitoringError
from alm_monitoring.modeling import (
    benchmark_table,
    compare_by_window,
    decide,
    economic_sign_check,
    partial_dependence_table,
    permutation_importance_table,
    run_benchmark,
    select_champion,
)
from alm_monitoring.security import resolve_within

_LOG = logging.getLogger("train_lending_club")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/lending_club"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--holdout-end", default="2017-01-01", help="fin (exclue) de la fenêtre hors temps"
    )
    parser.add_argument(
        "--quick", action="store_true", help="3 plis et échantillon réduit (démo, CI)"
    )
    parser.add_argument("--max-rows", type=int, default=None, help="limite de lignes lues (tests)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - pipeline linéaire lisible
    """Point d'entrée."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    base = Path.cwd()
    try:
        raw_path = resolve_within(base, args.raw)
        out_dir = resolve_within(base, args.out)
        reports_dir = resolve_within(base, args.reports)
        models_dir = resolve_within(base, args.models)
        for directory in (out_dir, reports_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)

        raw = lc.load_lending_club(raw_path, max_rows=args.max_rows)
        prepared = lc.prepare_lending_club(raw)
        prepared = lc.engineer_features(prepared)
        reference, current = lc.split_reference_current(prepared)
        holdout = current[current[lc.TIMESTAMP] < pd.Timestamp(args.holdout_end)]
        base_numeric, base_categorical = list(lc.NUMERIC_FEATURES), list(lc.CATEGORICAL_FEATURES)
        eng_numeric = [*base_numeric, *lc.ENGINEERED_NUMERIC_FEATURES]
        eng_categorical = [*base_categorical, *lc.ENGINEERED_CATEGORICAL_FEATURES]
        _LOG.info(
            "Référence %d prêts, surveillance %d prêts, hors temps %d prêts",
            len(reference),
            len(current),
            len(holdout),
        )

        audit = leakage.audit(
            reference,
            current,
            target=lc.TARGET,
            timestamp=lc.TIMESTAMP,
            numeric_features=eng_numeric,
            categorical_features=eng_categorical,
            forbidden_columns=lc.POST_ORIGINATION_COLUMNS,
        )
        _LOG.info("\n%s", audit.summary())
        if not audit.passed:
            _LOG.error("Audit de fuite en échec : entraînement interrompu")
            return 2

        n_folds = 3 if args.quick else 5
        results = run_benchmark(
            reference,
            holdout,
            target=lc.TARGET,
            numeric_features=base_numeric,
            categorical_features=base_categorical,
            n_folds=n_folds,
            feature_set="base",
        )
        # Même protocole avec les features métier : l'apport du feature engineering se
        # juge hors temps, jamais sur l'échantillon de développement.
        engineered_results = run_benchmark(
            reference,
            holdout,
            target=lc.TARGET,
            numeric_features=eng_numeric,
            categorical_features=eng_categorical,
            n_folds=n_folds,
            feature_set="engineered",
        )
        results.extend(r for r in engineered_results if r.name != "naive_prior")
        champion, reason = select_champion(results)
        _LOG.info("Champion : %s (%s)", champion.label, reason)
        _LOG.info("\n%s", benchmark_table(results).to_string())

        features = list(champion.features)
        reference_scored = reference.assign(**{lc.PREDICTION: champion.oof_scores})
        current_scored = current.assign(
            **{lc.PREDICTION: champion.estimator.predict_proba(current[features])[:, 1]}
        )

        # --- Explicabilité du champion (hors temps) et sens économique
        importance = permutation_importance_table(
            champion.estimator,
            holdout,
            target=lc.TARGET,
            features=features,
            sample_size=10_000 if args.quick else 50_000,
            n_repeats=3 if args.quick else 5,
        )
        pdp_columns = [
            c for c in ("int_rate", "rate_spread", "fico", "loan_to_income") if c in features
        ]
        pdp = partial_dependence_table(
            champion.estimator,
            holdout,
            features=features,
            columns=pdp_columns,
            sample_size=5_000 if args.quick else 20_000,
        )
        signs = economic_sign_check(
            pdp,
            expected={"int_rate": "croissant", "rate_spread": "croissant", "fico": "croissant"},
        )
        _LOG.info("\n%s", importance.head(10).to_string())
        _LOG.info("\n%s", signs.to_string())
        importance.to_csv(reports_dir / "importance_lending_club.csv", index=False)
        pdp.to_csv(reports_dir / "partial_dependence_lending_club.csv", index=False)

        # --- Champion-challenger : le meilleur candidat de l'autre jeu de features, en
        # parallèle du champion sur les fenêtres mensuelles de surveillance.
        challengers = [
            r for r in results if r.feature_set != champion.feature_set and r.name != "naive_prior"
        ]
        challenger_decision = None
        if challengers:
            challenger = min(challengers, key=lambda r: r.out_of_time.brier)
            scored = current_scored.assign(
                challenger_score=challenger.estimator.predict_proba(
                    current[list(challenger.features)]
                )[:, 1]
            )
            comparison = compare_by_window(
                scored,
                target=lc.TARGET,
                timestamp=lc.TIMESTAMP,
                champion_score=lc.PREDICTION,
                challenger_score="challenger_score",
                min_rows=200 if args.quick else 1000,
            )
            challenger_decision = decide(comparison)
            comparison.to_csv(reports_dir / "champion_challenger_lending_club.csv", index=False)
            _LOG.info(
                "Challenger %s : %s (%s)",
                challenger.label,
                challenger_decision.decision,
                challenger_decision.reason,
            )
        reference_scored.to_csv(out_dir / "reference.csv", index=False)
        current_scored.to_csv(out_dir / "current.csv", index=False)
        joblib.dump(champion.estimator, models_dir / "prepayment_champion.joblib")

        by_year = {
            str(year): _yearly(group)
            for year, group in current_scored.groupby(
                current_scored[lc.TIMESTAMP].dt.year, sort=True
            )
        }
        card: dict[str, Any] = {
            "model": "lending_club_prepayment",
            "task": "binary_classification",
            "target": lc.TARGET,
            "target_definition": (
                f"prêt intégralement remboursé dans les {lc.HORIZON_MONTHS} mois suivant "
                "l'émission et au moins un mois avant le terme contractuel"
            ),
            "champion": champion.name,
            "champion_feature_set": champion.feature_set,
            "champion_features": list(champion.features),
            "champion_description": champion.description,
            "selection_rule": reason,
            "economic_sign_check": signs.to_dict(orient="records"),
            "top_features": importance.head(10).to_dict(orient="records"),
            "champion_challenger": challenger_decision.to_dict() if challenger_decision else None,
            "trained_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "features": {
                "base_numeric": base_numeric,
                "base_categorical": base_categorical,
                "engineered_numeric": list(lc.ENGINEERED_NUMERIC_FEATURES),
                "engineered_categorical": list(lc.ENGINEERED_CATEGORICAL_FEATURES),
            },
            "forbidden_columns": list(lc.POST_ORIGINATION_COLUMNS),
            "reference_period": [
                _ym(reference[lc.TIMESTAMP].min()),
                _ym(reference[lc.TIMESTAMP].max()),
            ],
            "holdout_period": [_ym(holdout[lc.TIMESTAMP].min()), _ym(holdout[lc.TIMESTAMP].max())],
            "current_period": [_ym(current[lc.TIMESTAMP].min()), _ym(current[lc.TIMESTAMP].max())],
            "metrics": {
                "reference_out_of_fold": champion.in_time.to_dict(),
                "holdout_out_of_time": champion.out_of_time.to_dict(),
                "current_by_year": by_year,
            },
            "leakage_audit": audit.overall_severity.value,
        }
        (out_dir / "model_card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (reports_dir / "benchmark_lending_club.json").write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (reports_dir / "leakage_audit_lending_club.json").write_text(
            json.dumps(
                [f.to_record() for f in audit.findings], indent=2, ensure_ascii=False, default=str
            ),
            encoding="utf-8",
        )
        _LOG.info("Sorties écrites dans %s, %s et %s", out_dir, reports_dir, models_dir)
    except AlmMonitoringError as exc:
        _LOG.error("Échec : %s", exc)
        return 10
    return 0


def _yearly(group: pd.DataFrame) -> dict[str, float]:
    from alm_monitoring.modeling import compute_metrics  # noqa: PLC0415

    return compute_metrics(group[lc.TARGET].to_numpy(), group[lc.PREDICTION].to_numpy()).to_dict()


def _ym(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


if __name__ == "__main__":
    sys.exit(main())
