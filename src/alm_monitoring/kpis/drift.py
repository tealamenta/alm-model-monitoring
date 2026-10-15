"""Indicateurs de data drift et de prediction drift.

Chaque indicateur compare la distribution d'une colonne sur la fenêtre courante à celle
de la fenêtre de référence. Voir ``docs/kpi_methodology.md`` pour la justification
ALM des seuils.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from alm_monitoring.config import Thresholds
from alm_monitoring.exceptions import ConfigurationError, DataError, KPIComputationError
from alm_monitoring.kpis import _stats
from alm_monitoring.kpis.base import KPI, KPIResult, MonitoringContext, register
from alm_monitoring.types import KPICategory

_DEFAULT_BINS = 10
_MIN_BINS = 2
_MAX_BINS = 100


class _BinnedKPI(KPI):
    """Plomberie commune aux KPI acceptant un paramètre ``bins``."""

    allowed_params: ClassVar[frozenset[str]] = frozenset({"bins"})

    def _validate_params(self) -> None:
        unknown = set(self.params) - self.allowed_params
        if unknown:
            raise ConfigurationError(
                f"Paramètres inconnus pour le KPI '{self.name}'",
                details={"params": sorted(unknown)},
            )
        bins = self.params.get("bins", _DEFAULT_BINS)
        if (
            not isinstance(bins, int)
            or isinstance(bins, bool)
            or not _MIN_BINS <= bins <= _MAX_BINS
        ):
            raise ConfigurationError(
                f"'bins' du KPI '{self.name}' doit être un entier dans [{_MIN_BINS}, {_MAX_BINS}]",
                details={"bins": bins},
            )

    @property
    def bins(self) -> int:
        """Nombre de classes par quantiles utilisées pour les colonnes numériques."""
        return int(self.params.get("bins", _DEFAULT_BINS))

    def _per_column(
        self,
        ctx: MonitoringContext,
        columns: Sequence[str],
        fn: Callable[[str], KPIResult],
    ) -> list[KPIResult]:
        """Applique ``fn`` à chaque colonne, en convertissant les erreurs de données en ERROR."""
        results: list[KPIResult] = []
        for column in columns:
            if column not in ctx.reference.columns or column not in ctx.current.columns:
                results.append(
                    self.error_result("Colonne absente d'un des datasets", column=column)
                )
                continue
            try:
                results.append(fn(column))
            except (DataError, KPIComputationError) as exc:
                results.append(self.error_result(str(exc), column=column))
        return results

    def _numeric_distributions(
        self, ctx: MonitoringContext, column: str
    ) -> tuple[Any, Any, int, int]:
        ref = _stats.clean_numeric(ctx.reference[column])
        cur = _stats.clean_numeric(ctx.current[column])
        edges = _stats.reference_bin_edges(ref, self.bins)
        return (
            _stats.binned_frequencies(ref, edges),
            _stats.binned_frequencies(cur, edges),
            ref.size,
            cur.size,
        )


@register
class PSI(_BinnedKPI):
    """Population Stability Index par variable (numérique : quantiles ; catégorielle : niveaux)."""

    name = "psi"
    category = KPICategory.DATA_DRIFT
    description = (
        "Population Stability Index entre les distributions de référence et courante. "
        "Les colonnes numériques utilisent des classes par quantiles de la référence, "
        "les colonnes catégorielles leurs niveaux."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.25)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        ctx.require_min_samples(ctx.reference, ctx.current)
        numeric = ctx.numeric_columns(self.columns)
        categorical = ctx.categorical_columns(self.columns)

        def numeric_psi(column: str) -> KPIResult:
            ref_f, cur_f, n_ref, n_cur = self._numeric_distributions(ctx, column)
            value = _stats.psi(ref_f, cur_f)
            return self.result(
                value,
                column=column,
                details={"bins": int(ref_f.size), "n_reference": n_ref, "n_current": n_cur},
            )

        def categorical_psi(column: str) -> KPIResult:
            ref_f, cur_f, levels = _stats.categorical_frequencies(
                ctx.reference[column], ctx.current[column]
            )
            value = _stats.psi(ref_f, cur_f)
            return self.result(value, column=column, details={"levels": len(levels)})

        return self._per_column(ctx, numeric, numeric_psi) + self._per_column(
            ctx, categorical, categorical_psi
        )


@register
class JensenShannon(_BinnedKPI):
    """Distance de Jensen-Shannon, alternative bornée ([0, 1]) au PSI."""

    name = "js_distance"
    category = KPICategory.DATA_DRIFT
    description = (
        "Distance de Jensen-Shannon (base 2) entre les distributions de référence et courante."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.20)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        ctx.require_min_samples(ctx.reference, ctx.current)

        def numeric_js(column: str) -> KPIResult:
            ref_f, cur_f, *_ = self._numeric_distributions(ctx, column)
            return self.result(_stats.jensen_shannon(ref_f, cur_f), column=column)

        def categorical_js(column: str) -> KPIResult:
            ref_f, cur_f, _ = _stats.categorical_frequencies(
                ctx.reference[column], ctx.current[column]
            )
            return self.result(_stats.jensen_shannon(ref_f, cur_f), column=column)

        return self._per_column(ctx, ctx.numeric_columns(self.columns), numeric_js) + (
            self._per_column(ctx, ctx.categorical_columns(self.columns), categorical_js)
        )


@register
class KolmogorovSmirnov(_BinnedKPI):
    """Statistique KS à deux échantillons sur les variables numériques (p-value dans details)."""

    name = "ks_test"
    category = KPICategory.DATA_DRIFT
    description = (
        "Statistique de Kolmogorov-Smirnov à deux échantillons sur les variables numériques. "
        "La statistique est seuillée (sans échelle, stable pour n grand) ; la p-value est "
        "conservée à titre indicatif."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.20)
    allowed_params = frozenset()

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        ctx.require_min_samples(ctx.reference, ctx.current)

        def ks(column: str) -> KPIResult:
            ref = _stats.clean_numeric(ctx.reference[column])
            cur = _stats.clean_numeric(ctx.current[column])
            statistic, p_value = _stats.ks_statistic(ref, cur)
            return self.result(statistic, column=column, details={"p_value": p_value})

        return self._per_column(ctx, ctx.numeric_columns(self.columns), ks)


@register
class CramersV(_BinnedKPI):
    """Association du khi-deux entre l'appartenance à une fenêtre et une variable catégorielle."""

    name = "cramers_v"
    category = KPICategory.DATA_DRIFT
    description = (
        "V de Cramér corrigé du biais, issu d'un test d'indépendance du khi-deux entre "
        "l'indicateur de fenêtre (référence/courante) et une variable catégorielle."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.30)
    allowed_params = frozenset()

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        ctx.require_min_samples(ctx.reference, ctx.current)

        def cv(column: str) -> KPIResult:
            value, p_value = _stats.cramers_v(ctx.reference[column], ctx.current[column])
            return self.result(value, column=column, details={"p_value": p_value})

        return self._per_column(ctx, ctx.categorical_columns(self.columns), cv)


@register
class PredictionDrift(_BinnedKPI):
    """PSI sur la sortie du modèle : le signal le plus précoce lorsque les labels arrivent tard."""

    name = "prediction_drift"
    category = KPICategory.PREDICTION_DRIFT
    description = (
        "Population Stability Index de la colonne de prédiction/score. Détecte les "
        "changements de comportement avant que la vérité terrain ne soit observable "
        "(p. ex. remboursements anticipés réalisés des mois plus tard)."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.10, critical=0.25)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        ctx.require_min_samples(ctx.reference, ctx.current)
        column = ctx.model.prediction

        def score_psi(col: str) -> KPIResult:
            ref_f, cur_f, n_ref, n_cur = self._numeric_distributions(ctx, col)
            ref_mean = float(_stats.clean_numeric(ctx.reference[col]).mean())
            cur_mean = float(_stats.clean_numeric(ctx.current[col]).mean())
            return self.result(
                _stats.psi(ref_f, cur_f),
                column=col,
                details={
                    "mean_reference": ref_mean,
                    "mean_current": cur_mean,
                    "n_reference": n_ref,
                    "n_current": n_cur,
                },
            )

        return self._per_column(ctx, [column], score_psi)
