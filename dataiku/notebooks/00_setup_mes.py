"""Amorçage du monitoring natif depuis un notebook Jupyter DSS (exécuté sur DSS 15.1 Free).

La Free Edition n'expose ni l'API publique (pas de clé d'API) ni les déclencheurs de
scénario : ce notebook fait donc, **depuis l'intérieur de DSS** (``dataiku.api_client()``
est authentifié sans clé), ce que ``dataiku/recipes/00_setup_project.py`` fait depuis un
poste de travail sur une édition complète.

Il est idempotent : chaque cellule réutilise les objets déjà présents.

Enseignements vérifiés sur DSS 15.1 (voir docs/guide_implementation_dataiku.md §5) :

* un modèle MLflow exige un code env *non builtin* contenant ``mlflow``, ``scikit-learn``
  et ``statsmodels`` (ce dernier pour le drift natif) ;
* les datasets uploadés arrivent avec un schéma 100 % ``string`` : il faut le typer
  **avant** ``set_core_metadata``, sinon toutes les features sont déclarées catégorielles ;
* ``set_core_metadata(features_list=...)`` limite le drift aux vraies features du modèle
  (sans cela ``id``, ``issue_d``, ``score`` donnent un drift artificiel de 0,9998) ;
* la version importée doit être **évaluée sur la référence** avant toute recette
  d'évaluation ("The MLflow model version was not evaluated after import") ;
* ``list_model_evaluation_stores()`` renvoie des objets, pas des dicts.
"""

import dataiku

# ----------------------------------------------------------------------------- paramètres
MLFLOW_DIR = (
    "/Users/mohamed/Downloads/spuerkess/alm-model-monitoring/models/mlflow/prepayment_champion"
)
SAVED_MODEL_NAME = "prepayment_champion"
MES_NAME = "Prepayment - Model Evaluation Store"
RECIPE_NAME = "evaluate_prepayment_champion"
REFERENCE = "lc_reference"
CURRENT = "lc_current"
CODE_ENV = "alm-monitoring"  # code env avec mlflow + scikit-learn + statsmodels
VERSION_ID = "v2"
NUMERIC = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico",
    "emp_length_years",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "inq_last_6mths",
    "loan_to_income",
    "installment_to_income",
    "rate_spread",
    "credit_history_years",
    "mort_acc_missing",
    "term_x_rate",
]
CATEGORICAL = [
    "grade",
    "home_ownership",
    "purpose",
    "verification_status",
    "application_type",
    "fico_band",
]

client = dataiku.api_client()
project = client.get_default_project()

# ------------------------------------------------- 0. typage du schéma des datasets uploadés
for name in (REFERENCE, CURRENT):
    ds = project.get_dataset(name)
    settings = ds.get_settings()
    for col in settings.schema_columns:
        if col["name"] in NUMERIC or col["name"] == "score":
            col["type"] = "double"
        elif col["name"] == "prepaid_12m":
            col["type"] = "int"
    settings.save()
    print(name, ": schéma typé")

# ------------------------------------------------- 1. Saved Model (import MLflow)
existing = [m for m in project.list_saved_models() if m.get("name") == SAVED_MODEL_NAME]
if existing:
    sm = project.get_saved_model(existing[0]["id"])
    print("Saved Model existant :", sm.id)
else:
    sm = project.create_mlflow_pyfunc_model(SAVED_MODEL_NAME, "BINARY_CLASSIFICATION")
    print("Saved Model créé :", sm.id)

features = [{"name": c, "type": "double"} for c in NUMERIC] + [
    {"name": c, "type": "string"} for c in CATEGORICAL
]
if any(v["id"] == VERSION_ID for v in sm.list_versions()):
    handler = sm.get_external_model_version_handler(VERSION_ID)
    print("Version", VERSION_ID, "déjà importée")
else:
    # set_active=True par défaut ; la version importée devient la version active.
    handler = sm.import_mlflow_version_from_path(VERSION_ID, MLFLOW_DIR, CODE_ENV)
    handler.set_core_metadata("prepaid_12m", ["0", "1"], features_list=features)
    # Évaluation initiale sur la référence : baseline du drift natif (obligatoire).
    handler.evaluate(REFERENCE)
    print(
        "Version",
        VERSION_ID,
        "importée, évaluée sur",
        REFERENCE,
        "avec",
        len(features),
        "features déclarées",
    )

# ------------------------------------------------- 2. Model Evaluation Store
mes = None
for s in project.list_model_evaluation_stores():
    if s.get_settings().get_raw().get("name") == MES_NAME:
        mes = s
        break
if mes is None:
    mes = project.create_model_evaluation_store(MES_NAME)
    print("MES créé :", mes.id)
else:
    print("MES existant :", mes.id)

# ------------------------------------------------- 3. recette d'évaluation
if any(r.name == RECIPE_NAME for r in project.list_recipes(as_type="objects")):
    print("Recette d'évaluation existante")
else:
    b = project.new_recipe("evaluation", RECIPE_NAME)
    b.with_input_model(sm.id)
    b.with_input(CURRENT)
    b.with_output_evaluation_store(mes.id)
    b.create()
    print("Recette d'évaluation créée")

# ------------------------------------------------- 4. build du MES et lecture des métriques
job = mes.build(no_fail=True)
print("Job :", job.id, "->", job.get_status()["baseStatus"]["state"])
m = mes.get_latest_model_evaluation().get_full_info().metrics
print(
    "AUC",
    round(m["auc"], 4),
    "| logLoss",
    round(m["logLoss"], 4),
    "| calibrationLoss",
    round(m["calibrationLoss"], 4),
)
print(
    "dataDrift (précision du modèle de dérive)",
    round(m["dataDrift"], 4),
    "p =",
    m["dataDriftPValue"],
)
print("PSI par feature (top 8) :")
for k, v in sorted(m["psiPerFeature"].items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {k:25s} {v:.3f}")
print("predictionDrift_PSI", round(m["predictionDrift_PSI"], 5))
