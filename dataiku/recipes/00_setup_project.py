"""Amorçage du monitoring natif Dataiku : Saved Model MLflow + Model Evaluation Store.

À exécuter **hors** de DSS (poste de travail, CI) avec le client public ``dataikuapi``.
Le script est idempotent : il réutilise les objets déjà présents (même nom).

Étapes :

1. import du champion sérialisé au format MLflow comme *Saved Model* (prediction type
   ``BINARY_CLASSIFICATION``) et renseignement des métadonnées (cible, classes) ;
2. création du **Model Evaluation Store** ;
3. création de la **recette d'évaluation** (entrée : saved model + dataset courant,
   sortie : le MES) ;
4. construction du MES et lecture des métriques de la dernière évaluation
   (performance + data drift natif).

Variables d'environnement attendues :

* ``DSS_HOST``  : URL de l'instance (ex. ``https://dss.example.com``) ;
* ``DSS_API_KEY`` : clé d'API personnelle (jamais dans le code ni dans le YAML).

Usage::

    python dataiku/recipes/00_setup_project.py --project ALM_MONITORING \
        --mlflow-model models/mlflow/prepayment_champion \
        --current-dataset lc_current --code-env alm-monitoring
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import dataikuapi

from alm_monitoring.data import lending_club as lc

_LOG = logging.getLogger("dataiku_setup")

SAVED_MODEL_NAME = "prepayment_champion"
MES_NAME = "Prepayment - Model Evaluation Store"
RECIPE_NAME = "evaluate_prepayment_champion"


def get_client() -> dataikuapi.DSSClient:
    """Client DSS : clé d'API hors DSS, ou client interne si exécuté dans un notebook DSS.

    La Free Edition n'expose pas l'API publique : dans ce cas le script se lance depuis
    un notebook Jupyter du projet, où ``dataiku.api_client()`` est authentifié sans clé.
    """
    host, key = os.environ.get("DSS_HOST"), os.environ.get("DSS_API_KEY")
    if host and key:
        return dataikuapi.DSSClient(host, key)
    try:
        import dataiku  # noqa: PLC0415 - disponible uniquement dans DSS

        return dataiku.api_client()
    except ImportError as exc:
        raise SystemExit(
            "DSS_HOST et DSS_API_KEY doivent être définies (ou lancer dans un notebook DSS)"
        ) from exc


def get_or_create_saved_model(
    project: dataikuapi.dss.project.DSSProject,
    *,
    mlflow_model_dir: Path,
    current_dataset: str,
    reference_dataset: str,
    code_env: str,
    version_id: str,
) -> Any:
    """Importe le champion MLflow comme Saved Model (ou ajoute une version s'il existe)."""
    existing = [m for m in project.list_saved_models() if m.get("name") == SAVED_MODEL_NAME]
    if existing:
        sm = project.get_saved_model(existing[0]["id"])
        _LOG.info("Saved Model existant réutilisé : %s", sm.id)
    else:
        sm = project.create_mlflow_pyfunc_model(SAVED_MODEL_NAME, "BINARY_CLASSIFICATION")
        _LOG.info("Saved Model créé : %s", sm.id)
    # ``set_active=True`` par défaut : la version importée devient la version active.
    # Un code env *non builtin* contenant mlflow est obligatoire (vérifié sur DSS 15.1).
    version = sm.import_mlflow_version_from_path(version_id, str(mlflow_model_dir), code_env)
    # Cible, classes et features : indispensables pour que le MES calcule performance et drift.
    # Le schéma du dataset doit être typé (double/int) AVANT cet appel, sinon toutes les
    # features sont enregistrées comme catégorielles et le drift natif perd son sens.
    version.set_core_metadata(lc.TARGET, ["0", "1"], get_features_from_dataset=current_dataset)
    # Évaluation initiale sur la référence : obligatoire avant toute recette d'évaluation,
    # c'est elle qui fixe la baseline du drift natif ("model version was not evaluated").
    version.evaluate(reference_dataset)
    _LOG.info("Version %s importée, activée et évaluée sur %s", version_id, reference_dataset)
    return sm


def get_or_create_mes(project: dataikuapi.dss.project.DSSProject) -> Any:
    """Crée le Model Evaluation Store s'il n'existe pas."""
    # DSS 15 renvoie des objets DSSModelEvaluationStore (pas des dicts).
    for mes in project.list_model_evaluation_stores():
        if mes.get_settings().get_raw().get("name") == MES_NAME:
            _LOG.info("MES existant réutilisé : %s", mes.id)
            return mes
    mes = project.create_model_evaluation_store(MES_NAME)
    _LOG.info("MES créé : %s", mes.id)
    return mes


def get_or_create_evaluation_recipe(
    project: dataikuapi.dss.project.DSSProject,
    *,
    saved_model_id: str,
    current_dataset: str,
    mes_id: str,
) -> Any:
    """Recette d'évaluation : saved model + dataset courant → MES."""
    if any(r.name == RECIPE_NAME for r in project.list_recipes(as_type="objects")):
        _LOG.info("Recette d'évaluation existante réutilisée")
        return project.get_recipe(RECIPE_NAME)
    builder = project.new_recipe("evaluation", RECIPE_NAME)
    builder.with_input_model(saved_model_id)
    builder.with_input(current_dataset)
    builder.with_output_evaluation_store(mes_id)
    recipe = builder.create()
    _LOG.info("Recette d'évaluation créée : %s", recipe.name)
    return recipe


def build_and_read(mes: Any) -> dict[str, Any]:
    """Construit le MES (une nouvelle Model Evaluation) et renvoie ses métriques."""
    mes.build()
    me = mes.get_latest_model_evaluation()
    info = me.get_full_info()
    metrics = dict(info.metrics)
    _LOG.info("Évaluation %s : %s", info.creation_date, {k: metrics[k] for k in list(metrics)[:8]})
    drift = me.compute_drift()
    accuracy = drift.drift_model_result.drift_model_accuracy
    _LOG.info(
        "Data drift natif : précision du modèle de dérive %.3f (IC %.3f-%.3f, p=%.3g)",
        accuracy.value,
        accuracy.lower_confidence_interval,
        accuracy.upper_confidence_interval,
        accuracy.pvalue,
    )
    return {"metrics": metrics, "drift_model_accuracy": accuracy.value}


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="clé du projet DSS")
    parser.add_argument(
        "--mlflow-model", type=Path, required=True, help="dossier du modèle MLflow"
    )
    parser.add_argument("--current-dataset", default="lc_current")
    parser.add_argument("--reference-dataset", default="lc_reference")
    parser.add_argument(
        "--code-env", required=True, help="code env contenant mlflow + scikit-learn"
    )
    parser.add_argument("--version-id", default="v1")
    args = parser.parse_args(argv)

    client = get_client()
    project = client.get_project(args.project)
    sm = get_or_create_saved_model(
        project,
        mlflow_model_dir=args.mlflow_model,
        current_dataset=args.current_dataset,
        reference_dataset=args.reference_dataset,
        code_env=args.code_env,
        version_id=args.version_id,
    )
    mes = get_or_create_mes(project)
    get_or_create_evaluation_recipe(
        project, saved_model_id=sm.id, current_dataset=args.current_dataset, mes_id=mes.id
    )
    build_and_read(mes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
