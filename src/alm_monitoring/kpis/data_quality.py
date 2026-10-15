"""Indicateurs de qualité des données d'entrée.

Les problèmes de qualité des données (nouvelles valeurs manquantes, catégories jamais
vues à l'entraînement, valeurs hors de la plage d'entraînement) sont la cause racine la
plus fréquente des défaillances silencieuses d'un modèle en production, et ils sont
détectables immédiatement, sans labels.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from alm_monitoring.config import Thresholds
from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.kpis import _stats
from alm_monitoring.kpis.base import KPI, KPIResult, MonitoringContext, register
from alm_monitoring.types import KPICategory


class _QualityKPI(KPI):
    allowed_params: ClassVar[frozenset[str]] = frozenset()

    def _validate_params(self) -> None:
        unknown = set(self.params) - self.allowed_params
        if unknown:
            raise ConfigurationError(
                f"Paramètres inconnus pour le KPI '{self.name}'",
                details={"params": sorted(unknown)},
            )


@register
class MissingRate(_QualityKPI):
    """Part de valeurs manquantes par variable sur la fenêtre courante."""

    name = "missing_rate"
    category = KPICategory.DATA_QUALITY
    description = "Fraction de valeurs nulles par variable sur la fenêtre courante."
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.05, critical=0.20)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        results: list[KPIResult] = []
        for column in ctx.all_feature_columns(self.columns):
            if column not in ctx.current.columns:
                results.append(
                    self.error_result("Colonne absente du dataset courant", column=column)
                )
                continue
            current_rate = float(ctx.current[column].isna().mean())
            reference_rate = (
                float(ctx.reference[column].isna().mean())
                if column in ctx.reference.columns
                else None
            )
            results.append(
                self.result(
                    current_rate,
                    column=column,
                    details={"reference_rate": reference_rate},
                )
            )
        return results


@register
class OutOfRangeRate(_QualityKPI):
    """Part de valeurs numériques hors de la plage de référence.

    La plage est définie par des quantiles de la référence (``lower_q``/``upper_q``,
    0 % / 100 % par défaut, c'est-à-dire le min/max stricts) afin que des queues
    extrêmes mais légitimes puissent être tolérées par configuration.
    """

    name = "out_of_range_rate"
    category = KPICategory.DATA_QUALITY
    description = (
        "Fraction des valeurs numériques courantes hors de la plage [min, max] de référence."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.01, critical=0.05)
    allowed_params = frozenset({"lower_q", "upper_q"})

    def _validate_params(self) -> None:
        super()._validate_params()
        lower = self.params.get("lower_q", 0.0)
        upper = self.params.get("upper_q", 1.0)
        if not (0.0 <= lower < upper <= 1.0):
            raise ConfigurationError(
                "'lower_q' et 'upper_q' doivent vérifier 0 <= lower_q < upper_q <= 1",
                details={"lower_q": lower, "upper_q": upper},
            )

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        lower_q = float(self.params.get("lower_q", 0.0))
        upper_q = float(self.params.get("upper_q", 1.0))
        results: list[KPIResult] = []
        for column in ctx.numeric_columns(self.columns):
            if column not in ctx.reference.columns or column not in ctx.current.columns:
                results.append(
                    self.error_result("Colonne absente d'un des datasets", column=column)
                )
                continue
            ref = _stats.clean_numeric(ctx.reference[column])
            cur = _stats.clean_numeric(ctx.current[column])
            if ref.size == 0 or cur.size == 0:
                results.append(self.error_result("Aucune valeur finie", column=column))
                continue
            low, high = np.quantile(ref, [lower_q, upper_q])
            rate = float(((cur < low) | (cur > high)).mean())
            results.append(
                self.result(
                    rate,
                    column=column,
                    details={"lower_bound": float(low), "upper_bound": float(high)},
                )
            )
        return results


@register
class UnseenCategoriesRate(_QualityKPI):
    """Part des lignes courantes dont la catégorie n'a jamais été observée dans la référence."""

    name = "unseen_categories_rate"
    category = KPICategory.DATA_QUALITY
    description = (
        "Fraction des lignes courantes portant un niveau catégoriel absent de la référence "
        "(p. ex. un nouveau code produit que le modèle n'a jamais vu)."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(warning=0.01, critical=0.05)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        results: list[KPIResult] = []
        for column in ctx.categorical_columns(self.columns):
            if column not in ctx.reference.columns or column not in ctx.current.columns:
                results.append(
                    self.error_result("Colonne absente d'un des datasets", column=column)
                )
                continue
            known = set(ctx.reference[column].dropna().astype("string").unique())
            cur = ctx.current[column].dropna().astype("string")
            if cur.empty:
                results.append(self.error_result("Aucune valeur non nulle", column=column))
                continue
            unseen_mask = ~cur.isin(known)
            unseen_levels = sorted(cur[unseen_mask].unique().tolist())[:10]
            results.append(
                self.result(
                    float(unseen_mask.mean()),
                    column=column,
                    details={"unseen_levels_sample": unseen_levels},
                )
            )
        return results


@register
class SchemaConformity(_QualityKPI):
    """Nombre de violations de schéma (colonnes manquantes, dtypes incompatibles) côté courant."""

    name = "schema_conformity"
    category = KPICategory.DATA_QUALITY
    description = (
        "Nombre de violations de schéma dans le dataset courant : colonnes attendues "
        "manquantes, colonnes numériques non numériques ou colonne de prédiction absente."
    )
    higher_is_worse = True
    default_thresholds = Thresholds(critical=1.0)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        expected = [*ctx.model.features.all, ctx.model.prediction]
        violations: list[dict[str, Any]] = [
            {"column": column, "issue": "missing"}
            for column in expected
            if column not in ctx.current.columns
        ]
        violations.extend(
            {"column": column, "issue": f"not numeric ({ctx.current[column].dtype})"}
            for column in [*ctx.numeric_columns(), ctx.model.prediction]
            if column in ctx.current.columns
            and not pd.api.types.is_numeric_dtype(ctx.current[column])
        )
        return [self.result(float(len(violations)), details={"violations": violations})]


@register
class LabelCoverage(_QualityKPI):
    """Part des lignes de la fenêtre courante dont la cible **et** la prédiction sont connues.

    Les KPI de performance sont calculés sur les lignes labellisées seulement. Si 40 % des
    labels manquent, un AUC calculé sur les 60 % restants peut être biaisé (les prêts
    labellisés ne sont pas tirés au hasard : ils sont plus anciens, plus matures) sans que
    rien ne le signale. Ce KPI rend la couverture visible et seuillée ; les KPI de
    performance reprennent les mêmes effectifs dans leurs détails.

    Pour une cible différée (remboursement anticipé à 12 mois), une fenêtre récente a une
    couverture nulle : le résultat est alors CRITICAL, ce qui est l'information juste
    (« la performance de ce mois n'est pas encore mesurable »).
    """

    name = "label_coverage"
    category = KPICategory.DATA_QUALITY
    description = (
        "Fraction des lignes courantes avec cible et prédiction renseignées et valides ; "
        "assiette réelle des KPI de performance."
    )
    higher_is_worse = False
    default_thresholds = Thresholds(warning=0.95, critical=0.80)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:  # noqa: D102
        target, prediction = ctx.model.target, ctx.model.prediction
        total = len(ctx.current)
        if total == 0:
            return [self.error_result("Dataset courant vide")]
        if target not in ctx.current.columns or prediction not in ctx.current.columns:
            missing = [c for c in (target, prediction) if c not in ctx.current.columns]
            return [
                self.result(
                    0.0,
                    details={
                        "n_total": total,
                        "n_labelled": 0,
                        "excluded_rate": 1.0,
                        "missing_columns": missing,
                    },
                )
            ]
        pair = ctx.current[[target, prediction]].apply(pd.to_numeric, errors="coerce")
        labelled = int(pair.notna().all(axis=1).sum())
        return [
            self.result(
                labelled / total,
                details={
                    "n_total": total,
                    "n_labelled": labelled,
                    "excluded_rate": 1.0 - labelled / total,
                    "target_missing_rate": float(pair[target].isna().mean()),
                },
            )
        ]
