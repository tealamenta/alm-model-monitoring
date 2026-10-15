"""Recette Python DSS : calcul des KPIs ALM et alimentation du dataset ``monitoring_kpis``.

À coller dans une recette Python du Flow avec :

* entrées  : ``lc_reference`` (fenêtre de référence scorée), ``lc_current`` (fenêtre à
  surveiller, scorée par le modèle figé) ;
* sortie   : ``monitoring_kpis`` (table longue, une ligne par KPI x colonne x fenêtre).

La configuration YAML est lue dans les ressources du projet (``lib/python`` ou managed
folder ``configs``) ; par défaut, elle est intégrée ci-dessous via les variables du
projet pour rester autonome.

Le dataset ``monitoring_kpis`` est le socle de tout le reste : métriques & checks,
dashboard, et variables de scénario. Il est alimenté en **append** pour conserver
l'historique.
"""

import logging

import dataiku
from dataiku import pandasutils as pdu

from alm_monitoring import ModelMonitor
from alm_monitoring.config import MonitoringConfig
from alm_monitoring.dataiku import DataikuDatasetSource, publish_scenario_variables, write_report

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("recette_alm_kpis")

# ------------------------------------------------------------------ paramètres
variables = dataiku.get_custom_variables()
CONFIG_PATH = variables.get("alm_config_path", "lib/python/configs/lending_club_prepayment.yaml")
REFERENCE_DATASET = variables.get("alm_reference_dataset", "lc_reference")
CURRENT_DATASET = variables.get("alm_current_dataset", "lc_current")
OUTPUT_DATASET = variables.get("alm_output_dataset", "monitoring_kpis")
# Fréquence de découpe de la fenêtre courante (alias pandas) ; vide = fenêtre entière.
PERIOD = variables.get("alm_period", "MS")

# ------------------------------------------------------------------ chargement
config = MonitoringConfig.from_yaml(dataiku.get_project_root_path() + "/" + CONFIG_PATH)  # type: ignore[attr-defined]
monitor = ModelMonitor(config)

reference = DataikuDatasetSource(REFERENCE_DATASET).load()
current = DataikuDatasetSource(CURRENT_DATASET).load()
LOG.info("Référence %d lignes, courant %d lignes", len(reference), len(current))

# ------------------------------------------------------------------ calcul
if PERIOD:
    reports = list(monitor.run_over_time(reference, current, freq=PERIOD))
else:
    reports = [monitor.run(reference, current)]

for report in reports:
    write_report(report, OUTPUT_DATASET, append=True)
    LOG.info(
        "Fenêtre %s : %s (%d résultats)",
        report.metadata.get("window_start", "complète"),
        report.overall_severity.value,
        len(report.results),
    )

# La dernière fenêtre pilote les variables exposées au scénario (``${alm_overall_severity}``).
latest = reports[-1]
publish_scenario_variables(latest, extra={"alm_window_start": latest.metadata.get("window_start")})
monitor.alerts.process(latest)
