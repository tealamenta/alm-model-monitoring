"""Adaptateurs reliant le framework à l'API Python de Dataiku DSS.

Le SDK ``dataiku`` n'est disponible qu'à l'intérieur de DSS ; il est donc importé
paresseusement via :func:`_sdk`, que les tests remplacent par un faux. Rien ici n'est
nécessaire pour utiliser la bibliothèque hors de Dataiku.

Correspondance avec les fonctionnalités natives de Dataiku :

* :class:`DataikuDatasetSource` lit les fenêtres de référence / courante depuis des
  datasets DSS.
* :func:`write_report` ajoute la table de KPI au format long à un dataset DSS, qui
  sert de table sous-jacente aux graphiques/tableaux de bord natifs de Dataiku.
* :func:`dataiku_check` transforme un rapport en résultat de *custom Python check* sur
  le dataset de monitoring, de sorte qu'une étape « Run checks » de scénario échoue et
  déclenche les reporters (mail / Teams) configurés sur le scénario.
* :func:`publish_scenario_variables` expose la sévérité globale aux étapes de scénario.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pandas as pd

from alm_monitoring.exceptions import DataikuIntegrationError
from alm_monitoring.report import MonitoringReport
from alm_monitoring.types import Severity

_LOG = logging.getLogger(__name__)

_SEVERITY_TO_CHECK = {
    Severity.OK: "OK",
    Severity.WARNING: "WARNING",
    Severity.CRITICAL: "ERROR",
    Severity.ERROR: "ERROR",
}


def _sdk() -> ModuleType:
    try:
        import dataiku  # noqa: PLC0415 - disponible uniquement dans DSS
    except ImportError as exc:  # pragma: no cover - uniquement hors de DSS
        raise DataikuIntegrationError(
            "Le package 'dataiku' n'est disponible qu'à l'intérieur de DSS"
        ) from exc
    return dataiku


class DataikuDatasetSource:
    """Lit un dataset DSS (limite de lignes / colonnes optionnelles) en DataFrame."""

    def __init__(
        self,
        name: str,
        *,
        project_key: str | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
    ) -> None:
        if not name or any(ch.isspace() for ch in name):
            raise DataikuIntegrationError("Nom de dataset invalide", details={"name": name})
        self._name = name
        self._project_key = project_key
        self._limit = limit
        self._columns = columns

    def load(self) -> pd.DataFrame:  # noqa: D102
        dataset = _sdk().Dataset(self._name, project_key=self._project_key)
        kwargs: dict[str, Any] = {"infer_with_pandas": True}
        if self._limit is not None:
            kwargs["limit"] = self._limit
        if self._columns is not None:
            kwargs["columns"] = self._columns
        try:
            frame = dataset.get_dataframe(**kwargs)
        except Exception as exc:
            raise DataikuIntegrationError(
                "Échec de lecture du dataset Dataiku", details={"dataset": self._name}
            ) from exc
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)


def write_report(
    report: MonitoringReport, dataset_name: str, *, append: bool = True
) -> pd.DataFrame:
    """Ajoute (ou écrase) les lignes du rapport dans un dataset DSS et les renvoie."""
    frame = report.to_dataframe()
    dataset = _sdk().Dataset(dataset_name)
    try:
        if append:
            dataset.spec_item["appendMode"] = True
        dataset.write_with_schema(frame)
    except Exception as exc:
        raise DataikuIntegrationError(
            "Échec d'écriture du dataset de monitoring", details={"dataset": dataset_name}
        ) from exc
    _LOG.info("%d lignes de KPI écrites dans %s", len(frame), dataset_name)
    return frame


def dataiku_check(report: MonitoringReport) -> tuple[str, str]:
    """Renvoie ``(outcome, message)`` au format attendu par les custom Python checks de DSS."""
    severity = report.overall_severity
    breaches = report.breaches(Severity.WARNING)
    worst = ", ".join(
        f"{r.kpi}[{r.column or 'model'}]={r.value:.3f}"
        if r.value is not None
        else f"{r.kpi}: {r.message}"
        for r in sorted(breaches, key=lambda r: r.severity, reverse=True)[:5]
    )
    message = f"{severity.value.upper()} - {len(breaches)} dépassement(s)" + (
        f": {worst}" if worst else ""
    )
    return _SEVERITY_TO_CHECK[severity], message


def publish_scenario_variables(
    report: MonitoringReport, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Expose les résultats clés comme variables de scénario (``${alm_overall_severity}``...)."""
    variables: dict[str, Any] = {
        "alm_run_id": report.run_id,
        "alm_overall_severity": report.overall_severity.value,
        "alm_n_breaches": len(report.breaches(Severity.WARNING)),
        "alm_model_name": report.model_name,
        "alm_model_version": report.model_version,
        **dict(extra or {}),
    }
    try:
        scenario = _sdk().scenario.Scenario()
        scenario.set_scenario_variables(**variables)
    except Exception as exc:
        raise DataikuIntegrationError("Échec de définition des variables de scénario") from exc
    return variables
