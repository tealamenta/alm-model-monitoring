"""Catalogue des KPI.

L'import de ce package enregistre tous les indicateurs intégrés. Les projets externes
peuvent ajouter les leurs en dérivant :class:`alm_monitoring.kpis.KPI` et en décorant
la classe avec :func:`alm_monitoring.kpis.register`.
"""

from alm_monitoring.kpis import data_quality, drift, performance  # noqa: F401 - enregistrement
from alm_monitoring.kpis.base import (
    KPI,
    KPIResult,
    MonitoringContext,
    available_kpis,
    build_kpi,
    build_kpis,
    iter_kpi_metadata,
    register,
)

__all__ = [
    "KPI",
    "KPIResult",
    "MonitoringContext",
    "available_kpis",
    "build_kpi",
    "build_kpis",
    "iter_kpi_metadata",
    "register",
]
