"""Indicateurs de performance du modèle (requièrent des labels sur la fenêtre courante).

En ALM, les labels arrivent souvent avec retard (un remboursement anticipé ou un
retrait de dépôt est observé des mois après la prédiction). Ces KPI renvoient donc un
statut ERROR explicite lorsque les labels sont indisponibles, au lieu de répondre OK
en silence, et les KPI de prediction drift servent d'indicateurs avancés entre-temps.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from alm_monitoring.config import Thresholds
from alm_monitoring.exceptions import (
    ConfigurationError,
    DataError,
    InsufficientDataError,
    KPIComputationError,
)
from alm_monitoring.kpis import _stats
from alm_monitoring.kpis.base import KPI, KPIResult, MonitoringContext, register
from alm_monitoring.types import FloatArray, KPICategory, TaskType

_BINARY = frozenset({TaskType.BINARY_CLASSIFICATION})
_REGRESSION = frozenset({TaskType.REGRESSION})


def _labelled_arrays(
    ctx: MonitoringContext, frame: pd.DataFrame, *, what: str
) -> tuple[FloatArray, FloatArray]:
    """Renvoie des tableaux ``(y_true, y_pred)`` alignés, lignes incomplètes supprimées."""
    target, prediction = ctx.model.target, ctx.model.prediction
    missing = [c for c in (target, prediction) if c not in frame.columns]
    if missing:
        raise InsufficientDataError(
            f"Des colonnes requises par les KPI de performance sont absentes du dataset {what}",
            details={"missing": missing},
        )
    pair = frame[[target, prediction]].apply(pd.to_numeric, errors="coerce").dropna()
    y_true = pair[target].to_numpy(dtype="float64")
    y_pred = pair[prediction].to_numpy(dtype="float64")
    if y_true.size < ctx.min_sample_size:
        raise InsufficientDataError(
            f"Pas assez d'observations labellisées dans le dataset {what}",
            details={"rows": int(y_true.size), "min_sample_size": ctx.min_sample_size},
        )
    if ctx.model.task is TaskType.BINARY_CLASSIFICATION:
        if not np.isin(y_true, (0.0, 1.0)).all():
            raise KPIComputationError("La cible binaire doit être encodée en 0/1")
        if ((y_pred < 0.0) | (y_pred > 1.0)).any():
            raise KPIComputationError("Les probabilités prédites doivent être dans [0, 1]")
        if np.unique(y_true).size < 2:  # noqa: PLR2004
            raise InsufficientDataError(
                "Les deux classes doivent être présentes pour calculer la métrique"
            )
    return y_true, y_pred


class _PerformanceKPI(KPI):
    """Base des métriques à valeur unique calculées sur la fenêtre courante."""

    requires_labels = True
    allowed_params: ClassVar[frozenset[str]] = frozenset()

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:
        """Métrique implémentée par chaque sous-classe."""
        raise NotImplementedError

    def _validate_params(self) -> None:
        unknown = set(self.params) - self.allowed_params
        if unknown:
            raise ConfigurationError(
                f"Paramètres inconnus pour le KPI '{self.name}'",
                details={"params": sorted(unknown)},
            )

    def _score(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        return float(type(self).metric(y_true, y_pred))

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
        if not self.supports(ctx.model.task):
            return [self.error_result(f"KPI non applicable à la tâche '{ctx.model.task.value}'")]
        try:
            y_true, y_pred = _labelled_arrays(ctx, ctx.current, what="current")
            value = self._score(y_true, y_pred)
            n_total = len(ctx.current)
            # Assiette réelle du calcul : une performance mesurée sur 60 % des lignes n'a
            # pas la même valeur qu'une performance sur 100 % (voir KPI label_coverage).
            details: dict[str, Any] = {
                "n_labelled": int(y_true.size),
                "n_total": n_total,
                "excluded_rate": 1.0 - int(y_true.size) / n_total if n_total else None,
            }
            try:
                ref_true, ref_pred = _labelled_arrays(ctx, ctx.reference, what="reference")
                details["reference_value"] = self._score(ref_true, ref_pred)
            except DataError:
                details["reference_value"] = None
            return [self.result(value, details=details)]
        except (DataError, KPIComputationError) as exc:
            return [self.error_result(str(exc))]


@register
class AUC(_PerformanceKPI):
    """Aire sous la courbe ROC."""

    name = "auc"
    category = KPICategory.PERFORMANCE
    description = "AUC ROC sur la fenêtre courante (pouvoir discriminant)."
    higher_is_worse = False
    default_thresholds = Thresholds(warning=0.75, critical=0.70)
    supported_tasks = _BINARY

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(skm.roc_auc_score(y_true, y_pred))


@register
class BrierScore(_PerformanceKPI):
    """Erreur quadratique moyenne des prédictions probabilistes."""

    name = "brier_score"
    category = KPICategory.PERFORMANCE
    description = "Score de Brier (plus bas est meilleur) ; combine calibration et raffinement."
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.15)
    supported_tasks = _BINARY

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(skm.brier_score_loss(y_true, y_pred))


@register
class LogLoss(_PerformanceKPI):
    """Entropie croisée binaire."""

    name = "log_loss"
    category = KPICategory.PERFORMANCE
    description = "Perte logarithmique (log loss) sur la fenêtre courante."
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.35, critical=0.50)
    supported_tasks = _BINARY

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(skm.log_loss(y_true, np.clip(y_pred, 1e-9, 1 - 1e-9)))


@register
class ExpectedCalibrationError(_PerformanceKPI):
    """Écart entre les probabilités prédites et les fréquences observées."""

    name = "calibration_error"
    category = KPICategory.PERFORMANCE
    description = (
        "Expected Calibration Error avec 10 classes de largeur égale. Critique en ALM : une "
        "probabilité de remboursement anticipé ou de rétention des dépôts mal calibrée "
        "biaise directement les projections de flux, même si le classement (AUC) est intact."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.03, critical=0.06)
    supported_tasks = _BINARY

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return _stats.expected_calibration_error(y_true, y_pred)


@register
class LevelBias(_PerformanceKPI):
    """Biais de niveau : |moyenne prédite - taux observé| sur la fenêtre courante.

    C'est l'indicateur de performance le plus parlant en ALM : même si le classement
    (AUC) est intact, un modèle qui prédit 13 % de remboursements anticipés quand on
    en observe 16 % sous-estime l'écoulement des actifs et biaise les projections de
    flux. La valeur signée est conservée dans ``details`` (positif = surestimation).
    """

    name = "level_bias"
    category = KPICategory.PERFORMANCE
    description = (
        "Valeur absolue de l'écart entre la probabilité moyenne prédite et le taux "
        "observé sur la fenêtre courante (biais de niveau, en points de probabilité)."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.02, critical=0.04)
    supported_tasks = _BINARY

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(abs(y_pred.mean() - y_true.mean()))

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        try:
            y_true, y_pred = _labelled_arrays(ctx, ctx.current, what="current")
        except (DataError, KPIComputationError) as exc:
            return [self.error_result(str(exc))]
        signed = float(y_pred.mean() - y_true.mean())
        n_total = len(ctx.current)
        details: dict[str, Any] = {
            "n_labelled": int(y_true.size),
            "n_total": n_total,
            "excluded_rate": 1.0 - int(y_true.size) / n_total if n_total else None,
            "signed_bias": signed,
            "mean_prediction": float(y_pred.mean()),
            "observed_rate": float(y_true.mean()),
        }
        try:
            ref_true, ref_pred = _labelled_arrays(ctx, ctx.reference, what="reference")
            details["reference_value"] = float(abs(ref_pred.mean() - ref_true.mean()))
        except DataError:
            details["reference_value"] = None
        return [self.result(abs(signed), details=details)]


@register
class RMSE(_PerformanceKPI):
    """Racine de l'erreur quadratique moyenne (régression)."""

    name = "rmse"
    category = KPICategory.PERFORMANCE
    description = "Racine de l'erreur quadratique moyenne (RMSE) sur la fenêtre courante."
    higher_is_worse = True
    supported_tasks = _REGRESSION

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


@register
class MAE(_PerformanceKPI):
    """Erreur absolue moyenne (régression)."""

    name = "mae"
    category = KPICategory.PERFORMANCE
    description = "Erreur absolue moyenne (MAE) sur la fenêtre courante."
    higher_is_worse = True
    supported_tasks = _REGRESSION

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(skm.mean_absolute_error(y_true, y_pred))


@register
class R2(_PerformanceKPI):
    """Coefficient de détermination (régression)."""

    name = "r2"
    category = KPICategory.PERFORMANCE
    description = "R² sur la fenêtre courante."
    higher_is_worse = False
    default_thresholds = Thresholds(warning=0.50, critical=0.30)
    supported_tasks = _REGRESSION

    @staticmethod
    def metric(y_true: FloatArray, y_pred: FloatArray) -> float:  # noqa: D102
        return float(skm.r2_score(y_true, y_pred))


@register
class PerformanceDegradation(_PerformanceKPI):
    """Perte relative de la métrique principale par rapport à la fenêtre de référence.

    La métrique principale est l'AUC en classification et le R² en régression. Une
    valeur de 0,05 signifie que la métrique courante est 5 % en dessous de son niveau
    de référence.
    """

    name = "performance_degradation"
    category = KPICategory.PERFORMANCE
    description = (
        "Dégradation relative de la métrique principale (AUC ou R²) entre les fenêtres de "
        "référence et courante : (référence - courante) / référence."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.05, critical=0.10)

    @staticmethod
    def _primary(task: TaskType, y_true: FloatArray, y_pred: FloatArray) -> float:
        if task is TaskType.BINARY_CLASSIFICATION:
            return float(skm.roc_auc_score(y_true, y_pred))
        return float(skm.r2_score(y_true, y_pred))

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        try:
            cur_true, cur_pred = _labelled_arrays(ctx, ctx.current, what="current")
            ref_true, ref_pred = _labelled_arrays(ctx, ctx.reference, what="reference")
        except (DataError, KPIComputationError) as exc:
            return [self.error_result(str(exc))]
        reference = self._primary(ctx.model.task, ref_true, ref_pred)
        current = self._primary(ctx.model.task, cur_true, cur_pred)
        if reference <= 0:
            return [
                self.error_result("Métrique de référence non positive ; dégradation indéfinie")
            ]
        degradation = (reference - current) / reference
        return [
            self.result(
                degradation,
                details={"reference_value": reference, "current_value": current},
            )
        ]
