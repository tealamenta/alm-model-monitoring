"""Recette visuelle « ALM Monitoring KPIs » : formulaire → configuration → KPIs.

Le formulaire du plugin remplace le YAML : la configuration `MonitoringConfig` est
construite à partir des paramètres saisis, puis le monitor standard est exécuté. Toute
nouvelle version du package (nouveau KPI) est automatiquement disponible dans la recette.
"""

import dataiku
from alm_plugin_lib.config_builder import build_config_from_params
from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

from alm_monitoring import ModelMonitor

params = get_recipe_config()
reference = dataiku.Dataset(get_input_names_for_role("reference")[0]).get_dataframe()
current = dataiku.Dataset(get_input_names_for_role("current")[0]).get_dataframe()
output = dataiku.Dataset(get_output_names_for_role("kpis")[0])

config = build_config_from_params(params)
monitor = ModelMonitor(config)
period = params.get("period") or None

if period:
    frames = [r.to_dataframe() for r in monitor.run_over_time(reference, current, freq=period)]
else:
    frames = [monitor.run(reference, current).to_dataframe()]

import pandas as pd

table = pd.concat(frames, ignore_index=True)
if params.get("append", True):
    output.spec_item["appendMode"] = True
output.write_with_schema(table)
