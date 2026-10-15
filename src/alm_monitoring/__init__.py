"""Monitoring des modèles ALM : un cadre MLOps holistique, prêt pour Dataiku.

API publique::

    from alm_monitoring import ModelMonitor, MonitoringConfig

    monitor = ModelMonitor.from_yaml("configs/monitoring.yaml")
    report = monitor.run(reference_df, current_df)
    monitor.alerts.process(report)
"""

from alm_monitoring._version import __version__
from alm_monitoring.alerts import AlertManager
from alm_monitoring.config import MonitoringConfig
from alm_monitoring.exceptions import AlmMonitoringError
from alm_monitoring.kpis import KPI, KPIResult, MonitoringContext, available_kpis, register
from alm_monitoring.monitor import ModelMonitor
from alm_monitoring.report import MonitoringReport
from alm_monitoring.types import KPICategory, Severity, TaskType

__all__ = [
    "KPI",
    "AlertManager",
    "AlmMonitoringError",
    "KPICategory",
    "KPIResult",
    "ModelMonitor",
    "MonitoringConfig",
    "MonitoringContext",
    "MonitoringReport",
    "Severity",
    "TaskType",
    "__version__",
    "available_kpis",
    "register",
]
