"""Création du scénario step-based par l'API, depuis un notebook DSS (exécuté sur DSS 15.1).

Équivalent de ``dataiku/scenarios/alm_monitoring_scenario.json``. Les déclencheurs
temporels et le lancement par API ne sont pas disponibles en Free Edition : le scénario se
lance par le bouton *Run* (interface ou tuile de dashboard) ; sur une édition complète,
ajouter le déclencheur mensuel et le déclencheur « dataset modifié » sur ``lc_current``.

Observation utile : DSS exécute les règles Data Quality **automatiquement après le build**
de ``monitoring_kpis`` (option « Auto-run after build »). L'étape 2 échoue donc d'elle-même
quand la dernière fenêtre est CRITICAL, ce qui fait partir les reporters configurés sur
l'échec ; les étapes 3-4 deviennent redondantes mais restent explicites pour la lisibilité.
"""

import dataiku

client = dataiku.api_client()
project = client.get_default_project()

MES_NAME = "Prepayment - Model Evaluation Store"
mes = next(
    s
    for s in project.list_model_evaluation_stores()
    if s.get_settings().get_raw().get("name") == MES_NAME
)

SYNTHESIS = r"""
import dataiku
from dataiku.scenario import Scenario

RANK = {"ok": 0, "warning": 1, "critical": 2, "error": 3}
scenario = Scenario()
client = dataiku.api_client()
project = client.get_default_project()

# KPIs ALM : dernière fenêtre
kpis = dataiku.Dataset("monitoring_kpis").get_dataframe()
latest = kpis.sort_values("run_at")["run_id"].iloc[-1]
last = kpis[kpis["run_id"] == latest]
worst = max(last["severity"], key=lambda s: RANK.get(s, 0))
breaches = last[last["severity"] != "ok"]

# MES natif : dernière évaluation
mes = project.get_model_evaluation_store("__MES_ID__")
m = mes.get_latest_model_evaluation().get_full_info().metrics

scenario.set_scenario_variables(
    alm_overall_severity=worst,
    alm_n_breaches=int(len(breaches)),
    alm_window=str(last["window_start"].iloc[0]),
    alm_auc_native=round(float(m["auc"]), 4),
    alm_data_drift_native=round(float(m["dataDrift"]), 4),
    alm_top_breaches="; ".join(
        f"{r.kpi}[{r.column if isinstance(r.column, str) else 'modèle'}]={r.value:.3f}"
        for r in breaches.sort_values("value", ascending=False).head(5).itertuples()
    ),
)
print("Gravité globale :", worst, "-", len(breaches), "dépassement(s)")
print("MES natif : AUC", m["auc"], "| drift", m["dataDrift"])
""".replace("__MES_ID__", mes.id)

definition = {
    "type": "step_based",
    "name": "ALM - Monitoring mensuel du modèle de remboursement anticipé",
    "active": True,
    "params": {
        "steps": [
            {
                "type": "build_flowitem",
                "name": "1. Évaluation native (MES)",
                "params": {
                    "builds": [{"type": "MODEL_EVALUATION_STORE", "itemId": mes.id}],
                    "jobType": "RECURSIVE_BUILD",
                },
            },
            {
                "type": "build_flowitem",
                "name": "2. KPIs ALM -> monitoring_kpis",
                "params": {
                    "builds": [{"type": "DATASET", "itemId": "monitoring_kpis"}],
                    "jobType": "RECURSIVE_FORCED_BUILD",
                },
            },
            {
                "type": "compute_metrics",
                "name": "3. Métriques monitoring_kpis",
                "params": {"computes": [{"type": "DATASET", "itemId": "monitoring_kpis"}]},
            },
            {
                "type": "check_dataset",
                "name": "4. Règles de qualité (check Python)",
                "params": {"checks": [{"type": "DATASET", "itemId": "monitoring_kpis"}]},
            },
            {
                "type": "custom_python",
                "name": "5. Synthèse MES + KPIs ALM -> variables",
                "params": {"script": SYNTHESIS},
                "runConditionType": "RUN_IF_STATUS_MATCH",
                "runConditionStatuses": ["SUCCESS", "WARNING", "FAILED"],
            },
        ]
    },
}

existing = [s for s in project.list_scenarios() if s["name"] == definition["name"]]
if existing:
    scenario = project.get_scenario(existing[0]["id"])
    print("Scénario existant :", scenario.id)
else:
    scenario = project.create_scenario(definition["name"], "step_based", definition=definition)
    print("Scénario créé :", scenario.id)

# Édition complète uniquement (LicenseRestrictionException en Free Edition) : ajouter le
# déclencheur mensuel via ``scenario.get_settings().add_monthly_trigger(...)`` puis
# ``settings.save()``, et lancer par ``scenario.run_and_wait()``.
