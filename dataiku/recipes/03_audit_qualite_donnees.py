"""Recette Python DSS : audit de qualité des données et de la cible avant entraînement.

À coller dans une recette Python du Flow avec :

* entrée  : ``lc_raw`` (fichier Lending Club brut chargé dans DSS) ;
* sorties : ``audit_constats`` (une ligne par contrôle, gravité, message) et
  ``audit_tables`` (détail des règles métier, manquants, aberrants, VIF, volumes
  mensuels, cible par année, en format long).

Le verdict d'exploitabilité est publié en variable de scénario ``alm_data_exploitable``
pour conditionner l'étape d'entraînement. Dans DSS, cet audit complète les outils natifs,
il ne les remplace pas :

* **Explore → Analyze** : distribution, manquants et aberrants colonne par colonne
  (exploration interactive) ;
* **Metrics & Checks / Data Quality rules** sur ``lc_raw`` : nombre de lignes, unicité de
  ``id``, taux de nulls, valeurs attendues (``term`` dans {36, 60}) : les règles simples
  vivent là, versionnées avec le dataset et visibles dans le dashboard ;
* cette recette ajoute ce que les règles natives n'expriment pas : règles métier croisées
  (mensualité vs annuité), mécanisme des manquants, VIF, qualité de la cible, verdict.
"""

import dataiku
import pandas as pd

from alm_monitoring import data_audit as da
from alm_monitoring.data import lending_club as lc
from alm_monitoring.dataiku import DataikuDatasetSource, publish_scenario_variables

raw = DataikuDatasetSource("lc_raw").load()
prepared = lc.prepare_lending_club(raw, clean=False)  # l'audit doit voir les valeurs impossibles
report = da.audit(
    prepared,
    target=lc.TARGET,
    timestamp=lc.TIMESTAMP,
    numeric_features=lc.NUMERIC_FEATURES,
    categorical_features=lc.CATEGORICAL_FEATURES,
    raw=raw,
    rules=da.lending_club_rules(),
    duplicate_keys=[
        "loan_amnt",
        "int_rate",
        "term",
        "issue_d",
        "annual_inc",
        "dti",
        "fico_range_low",
    ],
)

dataiku.Dataset("audit_constats").write_with_schema(report.to_dataframe())
long_tables = []
for name, table in report.tables.items():
    if table.empty:
        continue
    melted = table.melt(var_name="colonne", value_name="valeur")
    melted.insert(0, "controle", name)
    long_tables.append(melted)
dataiku.Dataset("audit_tables").write_with_schema(pd.concat(long_tables, ignore_index=True))

# Variable de scénario : l'étape d'entraînement ne s'exécute que si les données sont exploitables.
from dataiku.scenario import Scenario

Scenario().set_scenario_variables(
    alm_data_exploitable=report.exploitable,
    alm_data_audit_severity=report.overall_severity.value,
)
