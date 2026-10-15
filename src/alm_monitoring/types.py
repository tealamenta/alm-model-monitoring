"""Énumérations et alias de types partagés."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray: TypeAlias = npt.NDArray[np.float64]
else:  # pragma: no cover - alias d'exécution uniquement
    FloatArray: TypeAlias = "np.ndarray"


class Severity(StrEnum):
    """Résultat de la comparaison d'une valeur de KPI à ses seuils.

    Les membres sont ordonnés de sorte que ``max()`` sur une collection renvoie le pire
    statut. Les quatre opérateurs de comparaison sont définis explicitement : ``str``
    fournit déjà des versions lexicographiques, que ``functools.total_ordering`` ne
    remplacerait pas.
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"  # le KPI n'a pas pu être calculé

    @property
    def rank(self) -> int:
        """Rang numérique utilisé pour l'ordonnancement (plus élevé = plus grave)."""
        return _SEVERITY_RANK[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank >= other.rank

    __hash__ = StrEnum.__hash__


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.OK: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
    Severity.ERROR: 3,
}


class TaskType(StrEnum):
    """Types de tâches d'apprentissage supervisé pris en charge."""

    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"


class KPICategory(StrEnum):
    """Famille fonctionnelle d'un KPI, pour regrouper tableaux de bord et documentation."""

    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
