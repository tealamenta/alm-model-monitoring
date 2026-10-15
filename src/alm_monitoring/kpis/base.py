"""Abstractions KPI : conteneur de résultat, contexte d'évaluation, classe de base et registre.

Un KPI compare un dataset de *référence* (les données d'entraînement/validation du
modèle, ou un instantané de base figé) à un dataset *courant* (données de production
récentes). Chaque KPI renvoie un :class:`KPIResult` par colonne surveillée, ou un
résultat unique pour les métriques globales telles que l'AUC.

Ajouter un nouveau KPI revient à dériver :class:`KPI`, à décorer la classe avec
:func:`register` et à implémenter :meth:`KPI.compute`. Le moniteur, le rapport, les
tableaux de bord et les recettes Dataiku le prennent en compte automatiquement : c'est
ce qui rend le framework « holistique » et applicable aux futurs modèles sans
modification de code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from alm_monitoring.config import KPISpec, ModelSpec, Thresholds
from alm_monitoring.exceptions import ConfigurationError, InsufficientDataError, UnknownKPIError
from alm_monitoring.types import KPICategory, Severity, TaskType


@dataclass(frozen=True, slots=True)
class KPIResult:
    """Résultat d'une évaluation de KPI.

    Attributes:
        kpi: Nom du KPI tel qu'enregistré.
        category: Famille fonctionnelle du KPI.
        column: Colonne concernée par le résultat, ou ``None`` pour les métriques
            au niveau du modèle.
        value: Statistique calculée, ``None`` lorsque le KPI a échoué.
        severity: Statut après comparaison aux seuils.
        thresholds: Seuils appliqués (``None`` si purement informatif).
        details: Contexte structuré additionnel (p-values, valeurs de référence...).
        message: Explication lisible, utilisée surtout lorsque ``severity`` vaut ERROR.
    """

    kpi: str
    category: KPICategory
    column: str | None
    value: float | None
    severity: Severity
    thresholds: Thresholds | None = None
    details: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Aplatit le résultat en une ligne exploitable dans un DataFrame ou un dataset Dataiku."""
        return {
            "kpi": self.kpi,
            "category": self.category.value,
            "column": self.column,
            "value": self.value,
            "severity": self.severity.value,
            "warning_threshold": self.thresholds.warning if self.thresholds else None,
            "critical_threshold": self.thresholds.critical if self.thresholds else None,
            "message": self.message,
            **{f"detail_{k}": v for k, v in sorted(self.details.items())},
        }


@dataclass(frozen=True, slots=True)
class MonitoringContext:
    """Ensemble immuable de tout ce dont un KPI a besoin pour s'exécuter."""

    reference: pd.DataFrame
    current: pd.DataFrame
    model: ModelSpec
    min_sample_size: int = 100

    def numeric_columns(self, restrict: Sequence[str] | None = None) -> list[str]:
        """Colonnes de variables numériques, éventuellement intersectées avec ``restrict``."""
        return self._select(self.model.features.numeric, restrict)

    def categorical_columns(self, restrict: Sequence[str] | None = None) -> list[str]:
        """Colonnes de variables catégorielles, éventuellement intersectées avec ``restrict``."""
        return self._select(self.model.features.categorical, restrict)

    def all_feature_columns(self, restrict: Sequence[str] | None = None) -> list[str]:
        """Toutes les colonnes de variables, éventuellement intersectées avec ``restrict``."""
        return self._select(self.model.features.all, restrict)

    def has_labels(self) -> bool:
        """Indique si le dataset courant contient des labels de vérité terrain exploitables."""
        target = self.model.target
        return target in self.current.columns and bool(self.current[target].notna().any())

    def require_min_samples(self, *frames: pd.DataFrame, what: str = "dataset") -> None:
        """Lève :class:`InsufficientDataError` si un DataFrame est plus petit que le minimum."""
        for frame in frames:
            if len(frame) < self.min_sample_size:
                raise InsufficientDataError(
                    f"Pas assez d'observations dans {what}",
                    details={"rows": len(frame), "min_sample_size": self.min_sample_size},
                )

    @staticmethod
    def _select(columns: Sequence[str], restrict: Sequence[str] | None) -> list[str]:
        if restrict is None:
            return list(columns)
        wanted = set(restrict)
        return [c for c in columns if c in wanted]


class KPI(ABC):
    """Classe de base de tout indicateur de monitoring.

    Les sous-classes déclarent des métadonnées au niveau de la classe et implémentent
    :meth:`compute`. L'évaluation des seuils et la construction des résultats sont
    gérées ici afin que tous les KPI se comportent de manière cohérente.
    """

    name: ClassVar[str]
    category: ClassVar[KPICategory]
    description: ClassVar[str] = ""
    higher_is_worse: ClassVar[bool] = True
    default_thresholds: ClassVar[Thresholds | None] = None
    supported_tasks: ClassVar[frozenset[TaskType] | None] = None
    requires_labels: ClassVar[bool] = False

    def __init__(
        self,
        *,
        thresholds: Thresholds | None = None,
        columns: Sequence[str] | None = None,
        **params: Any,
    ) -> None:
        self.thresholds = thresholds if thresholds is not None else self.default_thresholds
        self.columns = list(columns) if columns is not None else None
        self.params: dict[str, Any] = dict(params)
        self._validate_thresholds()
        self._validate_params()

    # ------------------------------------------------------------- hooks

    @abstractmethod
    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
        """Évalue le KPI sur le contexte fourni."""

    def _validate_params(self) -> None:
        """Valide ``self.params`` ; à surcharger pour rejeter les paramètres inconnus/invalides."""
        if self.params:
            raise ConfigurationError(
                f"Le KPI '{self.name}' n'accepte pas de paramètres",
                details={"params": sorted(self.params)},
            )

    # ---------------------------------------------------------- helpers

    def _validate_thresholds(self) -> None:
        if self.thresholds is None:
            return
        warning, critical = self.thresholds.warning, self.thresholds.critical
        if warning is None or critical is None:
            return
        ordered = warning <= critical if self.higher_is_worse else warning >= critical
        if not ordered:
            direction = "élevées" if self.higher_is_worse else "basses"
            raise ConfigurationError(
                f"Seuils incohérents pour '{self.name}' : les valeurs {direction} sont pires, "
                f"donc 'critical' doit être {'>=' if self.higher_is_worse else '<='} 'warning'",
                details={"warning": warning, "critical": critical},
            )

    def evaluate(self, value: float) -> Severity:
        """Associe une valeur de KPI à une sévérité selon les seuils et le sens de lecture."""
        if self.thresholds is None or value != value:  # noqa: PLR0124 - détection de NaN
            return Severity.OK
        critical, warning = self.thresholds.critical, self.thresholds.warning
        if self.higher_is_worse:
            if critical is not None and value >= critical:
                return Severity.CRITICAL
            if warning is not None and value >= warning:
                return Severity.WARNING
        else:
            if critical is not None and value <= critical:
                return Severity.CRITICAL
            if warning is not None and value <= warning:
                return Severity.WARNING
        return Severity.OK

    def result(
        self,
        value: float,
        *,
        column: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> KPIResult:
        """Construit un :class:`KPIResult` dont la sévérité est dérivée de ``value``."""
        return KPIResult(
            kpi=self.name,
            category=self.category,
            column=column,
            value=float(value),
            severity=self.evaluate(float(value)),
            thresholds=self.thresholds,
            details=dict(details or {}),
        )

    def error_result(self, message: str, *, column: str | None = None) -> KPIResult:
        """Construit un résultat signalant que le KPI n'a pas pu être calculé."""
        return KPIResult(
            kpi=self.name,
            category=self.category,
            column=column,
            value=None,
            severity=Severity.ERROR,
            thresholds=self.thresholds,
            message=message,
        )

    def supports(self, task: TaskType) -> bool:
        """Indique si le KPI est applicable au type de tâche donné."""
        return self.supported_tasks is None or task in self.supported_tasks

    def __repr__(self) -> str:
        return f"{type(self).__name__}(thresholds={self.thresholds!r}, params={self.params!r})"


# ---------------------------------------------------------------------- registre

_REGISTRY: dict[str, type[KPI]] = {}


def register(cls: type[KPI]) -> type[KPI]:
    """Décorateur de classe ajoutant une implémentation de KPI au registre global."""
    name = getattr(cls, "name", None)
    if not name:
        raise ConfigurationError(f"La classe KPI {cls.__name__} doit définir un 'name'")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ConfigurationError(f"Le nom de KPI '{name}' est déjà enregistré")
    _REGISTRY[name] = cls
    return cls


def available_kpis() -> dict[str, type[KPI]]:
    """Renvoie une copie du registre (nom -> classe)."""
    return dict(_REGISTRY)


def iter_kpi_metadata() -> Iterator[dict[str, Any]]:
    """Produit des métadonnées adaptées à la documentation pour chaque KPI enregistré."""
    for name, cls in sorted(_REGISTRY.items()):
        yield {
            "name": name,
            "category": cls.category.value,
            "description": cls.description,
            "higher_is_worse": cls.higher_is_worse,
            "default_warning": cls.default_thresholds.warning if cls.default_thresholds else None,
            "default_critical": cls.default_thresholds.critical
            if cls.default_thresholds
            else None,
            "requires_labels": cls.requires_labels,
            "tasks": sorted(t.value for t in cls.supported_tasks)
            if cls.supported_tasks
            else "all",
        }


def build_kpi(spec: KPISpec) -> KPI:
    """Instancie un KPI à partir de son entrée de configuration.

    Raises:
        UnknownKPIError: Si ``spec.name`` n'est pas enregistré.
        ConfigurationError: Si les paramètres ou les seuils sont invalides.
    """
    try:
        cls = _REGISTRY[spec.name]
    except KeyError as exc:
        raise UnknownKPIError(
            f"KPI inconnu '{spec.name}'", details={"available": sorted(_REGISTRY)}
        ) from exc
    return cls(thresholds=spec.thresholds, columns=spec.columns, **spec.params)


def build_kpis(specs: Iterable[KPISpec]) -> list[KPI]:
    """Instancie chaque KPI activé dans ``specs``."""
    return [build_kpi(s) for s in specs if s.enabled]
