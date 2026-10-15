"""Rapport de monitoring : instantané immuable et sérialisable d'une exécution de monitoring."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from alm_monitoring.kpis.base import KPIResult
from alm_monitoring.types import KPICategory, Severity


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True, slots=True)
class MonitoringReport:
    """Résultat d'une exécution de tous les KPI configurés.

    Attributes:
        model_name: Identifiant du modèle surveillé.
        model_version: Version du modèle surveillé.
        results: Résultats individuels des KPI.
        reference_rows: Taille de la fenêtre de référence.
        current_rows: Taille de la fenêtre courante.
        run_id: Identifiant unique de l'exécution (UUID4).
        run_at: Horodatage UTC de l'exécution.
        metadata: Contexte libre (dates de fenêtre, clé de projet Dataiku...).
    """

    model_name: str
    model_version: str
    results: tuple[KPIResult, ...]
    reference_rows: int
    current_rows: int
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- synthèse

    @property
    def overall_severity(self) -> Severity:
        """Pire sévérité parmi tous les résultats de KPI (OK s'il n'y en a aucun)."""
        return max((r.severity for r in self.results), default=Severity.OK)

    def breaches(self, min_severity: Severity = Severity.WARNING) -> list[KPIResult]:
        """Résultats dont la sévérité est au moins ``min_severity``."""
        return [r for r in self.results if r.severity >= min_severity]

    def by_category(self, category: KPICategory) -> list[KPIResult]:
        """Résultats appartenant à une famille de KPI."""
        return [r for r in self.results if r.category is category]

    def summary(self) -> dict[str, Any]:
        """Agrégat compact, adapté aux tableaux de bord."""
        severity_counts = Counter(r.severity.value for r in self.results)
        category_worst = {
            cat.value: max((r.severity for r in self.by_category(cat)), default=Severity.OK).value
            for cat in KPICategory
        }
        return {
            "run_id": self.run_id,
            "run_at": self.run_at.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "overall_severity": self.overall_severity.value,
            "n_results": len(self.results),
            "severity_counts": {s.value: severity_counts.get(s.value, 0) for s in Severity},
            "category_worst_severity": category_worst,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "metadata": dict(self.metadata),
        }

    # --------------------------------------------------------- sérialisation

    def to_dataframe(self) -> pd.DataFrame:
        """Une ligne par résultat de KPI, avec les colonnes de niveau exécution répétées.

        C'est le format écrit dans le dataset Dataiku *monitoring_kpis* et lu par les
        tableaux de bord.
        """
        records = [
            {
                "run_id": self.run_id,
                "run_at": self.run_at,
                "window_start": self.metadata.get("window_start"),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "config_fingerprint": self.metadata.get("config_fingerprint"),
                "package_version": self.metadata.get("package_version"),
                **r.to_record(),
            }
            for r in self.results
        ]
        columns = [
            "run_id",
            "run_at",
            "window_start",
            "model_name",
            "model_version",
            "config_fingerprint",
            "package_version",
            "kpi",
            "category",
            "column",
            "value",
            "severity",
            "warning_threshold",
            "critical_threshold",
            "message",
        ]
        frame = pd.DataFrame.from_records(records)
        detail_cols = sorted(c for c in frame.columns if c.startswith("detail_"))
        return frame.reindex(columns=[*columns, *detail_cols])

    def to_dict(self) -> dict[str, Any]:
        """Représentation compatible JSON (les détails sont convertis en chaîne si besoin)."""
        return {
            **self.summary(),
            "results": [
                {
                    **r.to_record(),
                    "thresholds": r.thresholds.model_dump() if r.thresholds else None,
                }
                for r in self.results
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Sérialise en JSON (les scalaires NumPy et les datetimes sont pris en charge)."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default, sort_keys=False)

    @staticmethod
    def concat(reports: Iterable[MonitoringReport]) -> pd.DataFrame:
        """Empile plusieurs rapports dans un DataFrame au format long (historique / séries)."""
        frames = [r.to_dataframe() for r in reports]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _json_default(obj: object) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "item"):  # scalaire numpy
        return obj.item()
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    raise TypeError(f"Un objet de type {type(obj).__name__} n'est pas sérialisable en JSON")
