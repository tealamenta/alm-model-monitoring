"""Scénario DSS *custom Python* : orchestration mensuelle du monitoring.

À coller dans un scénario de type « Custom Python script » (Scenarios → New → Custom).
Déclencheur conseillé : mensuel, le 2 du mois à 06:00 (après l'arrêté), plus un
déclencheur « dataset modifié » sur ``lc_current``.

Déroulé :

1. **build du Model Evaluation Store** (recette d'évaluation native) ;
2. **build de** ``monitoring_kpis`` (recette Python des KPIs ALM) ;
3. **lecture** de la dernière évaluation du MES (performance + drift natif) et des
   KPIs ALM de la dernière fenêtre ;
4. **fusion** dans une synthèse unique et calcul de la gravité globale ;
5. **métriques & checks** natifs sur ``monitoring_kpis`` (le check personnalisé est dans
   ``dataiku/checks/``) ;
6. **variables de scénario** ``alm_*`` pour les étapes suivantes et les reporters ;
7. **message** sur le canal configuré (mail / Teams / Slack) si la gravité atteint le
   seuil, avec le détail des KPIs en dépassement.

Les identifiants (MES, canal) sont lus dans les variables du projet pour ne rien coder
en dur.
"""

import json
import logging

import dataiku
from dataiku.scenario import Scenario

from alm_monitoring.types import Severity

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("scenario_alm")

scenario = Scenario()
variables = dataiku.get_custom_variables()
PROJECT_KEY = dataiku.default_project_key()
MES_ID = variables["alm_mes_id"]  # identifiant du Model Evaluation Store (≠ son nom)
KPI_DATASET = variables.get("alm_output_dataset", "monitoring_kpis")
CHANNEL_ID = variables.get("alm_messaging_channel", "")
MIN_SEVERITY = Severity(variables.get("alm_alert_min_severity", "warning"))

# ------------------------------------------------- 1-2. builds (natif + KPIs ALM)
scenario.build_evaluation_store(MES_ID, step_name="Évaluation native (MES)")
scenario.build_dataset(KPI_DATASET, build_mode="RECURSIVE_FORCED_BUILD", step_name="KPIs ALM")

# ------------------------------------------------- 3. lecture du MES
client = dataiku.api_client()
project = client.get_project(PROJECT_KEY)
mes = project.get_model_evaluation_store(MES_ID)
evaluation = mes.get_latest_model_evaluation().get_full_info()
native_metrics = dict(evaluation.metrics)
native = {
    "auc_native": native_metrics.get("auc"),
    "log_loss_native": native_metrics.get("logLoss"),
    # "dataDrift" = précision du modèle de dérive (clé vérifiée sur DSS 15.1).
    "data_drift_native": native_metrics.get("dataDrift"),
}
LOG.info("MES %s : %s", MES_ID, native)

# ------------------------------------------------- 3. lecture des KPIs ALM
kpis = dataiku.Dataset(KPI_DATASET).get_dataframe()
latest_run = kpis.sort_values("run_at")["run_id"].iloc[-1]
last = kpis[kpis["run_id"] == latest_run]
severities = [Severity(s) for s in last["severity"]]
overall = max(severities, default=Severity.OK)
breaches = last[last["severity"].isin(["warning", "critical", "error"])]

# ------------------------------------------------- 4. synthèse
summary = {
    "run_id": latest_run,
    "window_start": str(last["window_start"].iloc[0]) if "window_start" in last else None,
    "overall_severity": overall.value,
    "n_breaches": len(breaches),
    **native,
}
LOG.info("Synthèse : %s", json.dumps(summary, default=str))

# ------------------------------------------------- 5. métriques & checks natifs
scenario.compute_dataset_metrics(KPI_DATASET, step_name="Métriques monitoring_kpis")
scenario.run_dataset_checks(KPI_DATASET, step_name="Checks monitoring_kpis", fail_fatal=False)

# ------------------------------------------------- 6. variables de scénario
scenario.set_scenario_variables(
    alm_run_id=latest_run,
    alm_overall_severity=overall.value,
    alm_n_breaches=len(breaches),
    alm_auc_native=native["auc_native"],
    alm_data_drift_native=native["data_drift_native"],
)

# ------------------------------------------------- 7. alerte
if overall >= MIN_SEVERITY and CHANNEL_ID:
    top = breaches.sort_values("severity", ascending=False).head(10)
    lines = [
        f"- {row.kpi} [{row.column or 'modèle'}] = {row.value:.3f} ({row.severity})"
        for row in top.itertuples()
    ]
    body = (
        f"Monitoring ALM - {summary['window_start']} : **{overall.value.upper()}**\n"
        f"{len(breaches)} dépassement(s). AUC natif : {native['auc_native']}, "
        f"drift natif : {native['data_drift_native']}\n" + "\n".join(lines)
    )
    sender = scenario.get_message_sender(CHANNEL_ID)
    sender.send(
        subject=f"[ALM monitoring] {overall.value.upper()} - {summary['window_start']}",
        message=body,
    )
    LOG.info("Alerte envoyée sur le canal %s", CHANNEL_ID)
else:
    LOG.info("Aucune alerte : gravité %s sous le seuil %s", overall.value, MIN_SEVERITY.value)
