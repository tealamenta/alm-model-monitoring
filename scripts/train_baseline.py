"""Entraîne le modèle de propension de base sur la fenêtre de développement et score tout.

Le modèle est volontairement simple et auditable (one-hot + régression logistique), à
l'image du type de modèle comportemental qu'une équipe ALM placerait sous monitoring. Il
est ajusté **uniquement** sur la période de référence (mai 2008 - juin 2009 par défaut),
de sorte que les scores après la date de coupure reflètent ce qu'un modèle figé au moment
de la validation produirait en production. Les scores de référence sont *out-of-fold*
(validation croisée à 5 plis) afin que la performance de référence ne soit pas biaisée
de façon optimiste.

Sorties (CSV, séparateur ``,``) :

* ``data/processed/reference.csv`` - lignes de développement avec un ``score`` out-of-fold ;
* ``data/processed/current.csv``   - lignes postérieures à la coupure scorées par le modèle
  figé ;
* ``data/processed/model_card.json`` - métadonnées d'ajustement et métriques in/out-of-time.

Utilisation::

    uv run python scripts/train_baseline.py [--raw data/raw/bank-additional-full.csv]
                                            [--out data/processed] [--cutoff 2009-07-01]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from alm_monitoring.data.bank_marketing import (
    CATEGORICAL_FEATURES,
    DEFAULT_CUTOFF,
    NUMERIC_FEATURES,
    PREDICTION,
    TARGET,
    TIMESTAMP,
    prepare_bank_marketing,
    split_reference_current,
)
from alm_monitoring.exceptions import AlmMonitoringError
from alm_monitoring.security import resolve_within

RANDOM_STATE = 42
N_FOLDS = 5
ALGORITHM = "OneHotEncoder(min_frequency=20) + StandardScaler + LogisticRegression(C=0.5)"


def build_pipeline() -> Pipeline:
    """One-hot des catégorielles, mise à l'échelle des numériques, régression logistique L2."""
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    model = LogisticRegression(max_iter=2000, C=0.5, class_weight=None, random_state=RANDOM_STATE)
    return Pipeline([("prep", preprocess), ("clf", model)])


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame[TARGET].to_numpy()
    p = frame[PREDICTION].to_numpy()
    return {
        "rows": float(len(frame)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def train_and_score(
    raw: pd.DataFrame, *, cutoff: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prépare, scinde, ajuste sur la référence, score les deux fenêtres."""
    prepared = prepare_bank_marketing(raw)
    reference, current = split_reference_current(prepared, cutoff=cutoff)

    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        build_pipeline(), reference[features], reference[TARGET], cv=folds, method="predict_proba"
    )[:, 1]
    pipeline = build_pipeline()
    pipeline.fit(reference[features], reference[TARGET])

    reference = reference.assign(**{PREDICTION: oof})
    current = current.assign(**{PREDICTION: pipeline.predict_proba(current[features])[:, 1]})

    by_year = {
        str(year): _metrics(group)
        for year, group in current.groupby(current[TIMESTAMP].dt.year, sort=True)
    }
    card: dict[str, Any] = {
        "model": "term_deposit_propensity",
        "algorithm": ALGORITHM,
        "trained_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "cutoff": cutoff,
        "features": {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES)},
        "reference_period": [
            reference[TIMESTAMP].min().strftime("%Y-%m"),
            reference[TIMESTAMP].max().strftime("%Y-%m"),
        ],
        "current_period": [
            current[TIMESTAMP].min().strftime("%Y-%m"),
            current[TIMESTAMP].max().strftime("%Y-%m"),
        ],
        "metrics": {"reference_out_of_fold": _metrics(reference), "current_by_year": by_year},
    }
    return reference, current, card


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/bank-additional-full.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/bank_marketing"))
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    args = parser.parse_args(argv)

    try:
        raw_path = resolve_within(Path.cwd(), args.raw)
        raw = pd.read_csv(raw_path, sep=";")
        reference, current, card = train_and_score(raw, cutoff=args.cutoff)
    except (AlmMonitoringError, OSError, pd.errors.ParserError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.out / "reference.csv", index=False)
    current.to_csv(args.out / "current.csv", index=False)
    (args.out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    ref = card["metrics"]["reference_out_of_fold"]
    print(
        f"référence {card['reference_period']} : lignes={int(ref['rows'])} "
        f"oof_auc={ref['auc']:.3f} brier={ref['brier']:.3f} pos_rate={ref['positive_rate']:.3f}"
    )
    for year, metrics in card["metrics"]["current_by_year"].items():
        print(
            f"courant {year} : lignes={int(metrics['rows'])} "
            f"auc={metrics['auc']:.3f} brier={metrics['brier']:.3f} "
            f"pos_rate={metrics['positive_rate']:.3f}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
