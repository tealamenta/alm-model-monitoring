"""Exporte le champion sérialisé au format MLflow pour import dans Dataiku (ou un registre).

MLflow n'est pas nécessaire au monitoring : il sert ici de **format d'échange** entre
l'entraînement (hors Dataiku) et le Saved Model Dataiku, qui sait importer un modèle
MLflow (``import_mlflow_version_from_path``) et le brancher sur un Model Evaluation
Store. Le même dossier peut être enregistré dans un registre MLflow classique.

Usage::

    python scripts/export_mlflow_model.py --model models/prepayment_champion.joblib \
        --out models/mlflow/prepayment_champion [--track]

Avec ``--track``, un *run* MLflow local (``mlruns/``) est créé avec les métriques de la
fiche modèle, pour illustrer le suivi d'expériences.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib

from alm_monitoring.security import resolve_within

_LOG = logging.getLogger("export_mlflow")


def _signature(card: dict[str, Any]) -> Any:
    """Signature MLflow (entrées = features du champion, sortie = probabilité)."""
    from mlflow.models.signature import ModelSignature  # noqa: PLC0415
    from mlflow.types.schema import ColSpec, Schema  # noqa: PLC0415

    feats = card["features"]
    categorical = set(feats["base_categorical"]) | set(feats["engineered_categorical"])
    inputs = Schema(
        [
            ColSpec("string" if name in categorical else "double", name)
            for name in card["champion_features"]
        ]
    )
    return ModelSignature(inputs=inputs, outputs=Schema([ColSpec("double")]))


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/prepayment_champion.joblib"))
    parser.add_argument(
        "--card", type=Path, default=Path("data/processed/lending_club/model_card.json")
    )
    parser.add_argument("--out", type=Path, default=Path("models/mlflow/prepayment_champion"))
    parser.add_argument(
        "--track", action="store_true", help="enregistre aussi un run MLflow local"
    )
    args = parser.parse_args(argv)
    try:
        import mlflow  # noqa: PLC0415 - dépendance optionnelle
        import mlflow.sklearn  # noqa: PLC0415
    except ImportError:
        _LOG.error("mlflow n'est pas installé : pip install 'alm-model-monitoring[mlflow]'")
        return 10

    base = Path.cwd()
    model_path = resolve_within(base, args.model)
    out_dir = resolve_within(base, args.out)
    card = json.loads(resolve_within(base, args.card).read_text(encoding="utf-8"))
    estimator = joblib.load(model_path)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Format pickle (cloudpickle) : c'est celui que Dataiku charge nativement ; le format
    # skops par défaut de MLflow ≥ 3 refuse les types internes de HistGradientBoosting.
    fmt = mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    # Signature explicite limitée aux features du champion : sans elle, Dataiku déduit la
    # signature du dataset d'import (id, issue_d, score…) et ces colonnes polluent le
    # drift natif du Model Evaluation Store (constaté sur DSS 15.1).
    signature = _signature(card)
    mlflow.sklearn.save_model(
        estimator, str(out_dir), serialization_format=fmt, signature=signature
    )
    n_features = len(signature.inputs.inputs)
    _LOG.info("Modèle MLflow écrit dans %s (%d features signées)", out_dir, n_features)

    if args.track:
        mlflow.set_tracking_uri(f"sqlite:///{base / 'mlflow.db'}")  # registre local SQLite
        mlflow.set_experiment("alm-prepayment")
        with mlflow.start_run(run_name=card.get("champion", "champion")):
            mlflow.log_params({"champion": card.get("champion"), "target": card.get("target")})
            for scope in ("reference_out_of_fold", "holdout_out_of_time"):
                for name, value in card["metrics"][scope].items():
                    mlflow.log_metric(f"{scope}_{name}", float(value))
            mlflow.sklearn.log_model(estimator, name="model", serialization_format=fmt)
        _LOG.info(
            "Run MLflow enregistré (mlflow.db + mlruns/) ; `mlflow ui --backend-store-uri sqlite:///mlflow.db`"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
