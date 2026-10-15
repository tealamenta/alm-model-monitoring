"""Cas limites des KPI de dérive : distributions dégénérées, colonnes absentes ou vides.

Un KPI de dérive doit, dans ces situations, soit renvoyer une valeur stable et
interprétable, soit un résultat ERROR explicite ; jamais une exception, un NaN silencieux
ou une fausse alerte.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.config import ModelSpec
from alm_monitoring.kpis import MonitoringContext
from alm_monitoring.kpis.drift import PSI, CramersV, JensenShannon, KolmogorovSmirnov
from alm_monitoring.types import Severity

pytestmark = pytest.mark.unit

_DRIFT_KPIS = [PSI, JensenShannon, KolmogorovSmirnov]


def _ctx(reference: pd.DataFrame, current: pd.DataFrame, model: ModelSpec) -> MonitoringContext:
    return MonitoringContext(reference=reference, current=current, model=model)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


class TestIdenticalDistributions:
    @pytest.mark.parametrize("kpi_cls", [*_DRIFT_KPIS, CramersV])
    def test_reference_equals_current_is_ok_and_near_zero(
        self, reference_df: pd.DataFrame, model_spec: ModelSpec, kpi_cls: type
    ) -> None:
        results = kpi_cls().compute(_ctx(reference_df, reference_df.copy(), model_spec))
        assert results, "au moins une colonne évaluée"
        for r in results:
            assert r.severity is Severity.OK, r.to_record()
            assert _finite(r.value) and r.value is not None and r.value <= 1e-9

    def test_resampled_copy_stays_ok(
        self, reference_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        """Un ré-échantillonnage avec remise de la référence ne doit pas alerter."""
        boot = reference_df.sample(frac=1.0, replace=True, random_state=0)
        for kpi_cls in _DRIFT_KPIS:
            for r in kpi_cls().compute(_ctx(reference_df, boot, model_spec)):
                assert r.severity is Severity.OK, (kpi_cls.__name__, r.to_record())


class TestDegenerateColumns:
    def test_constant_numeric_in_both_is_ok(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        ref = reference_df.assign(age=42.0)
        cur = current_stable_df.assign(age=42.0)
        for kpi_cls in _DRIFT_KPIS:
            by_col = {r.column: r for r in kpi_cls().compute(_ctx(ref, cur, model_spec))}
            r = by_col["age"]
            assert r.severity is Severity.OK and r.value == pytest.approx(0.0), (
                kpi_cls.__name__,
                r.to_record(),
            )

    def test_constant_reference_shifted_current_is_detected(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        """Référence constante (42) et courant constant ailleurs (43) : dérive maximale."""
        ref = reference_df.assign(age=42.0)
        cur = current_stable_df.assign(age=43.0)
        for kpi_cls in (PSI, JensenShannon):
            by_col = {r.column: r for r in kpi_cls().compute(_ctx(ref, cur, model_spec))}
            assert by_col["age"].severity is Severity.CRITICAL, (
                kpi_cls.__name__,
                by_col["age"].to_record(),
            )

    def test_all_nan_current_column_is_error_not_ok(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        """Une colonne entièrement vide côté courant ne peut pas être « sans dérive »."""
        cur = current_stable_df.assign(age=np.nan)
        for kpi_cls in _DRIFT_KPIS:
            by_col = {r.column: r for r in kpi_cls().compute(_ctx(reference_df, cur, model_spec))}
            assert by_col["age"].severity is Severity.ERROR, (
                kpi_cls.__name__,
                by_col["age"].to_record(),
            )
            assert by_col["age"].value is None

    def test_single_category_everywhere_is_ok(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        ref = reference_df.assign(product="loan")
        cur = current_stable_df.assign(product="loan")
        for kpi_cls in (PSI, JensenShannon, CramersV):
            by_col = {r.column: r for r in kpi_cls().compute(_ctx(ref, cur, model_spec))}
            r = by_col["product"]
            assert r.severity is Severity.OK, (kpi_cls.__name__, r.to_record())
            assert _finite(r.value) and r.value is not None and r.value <= 1e-9

    def test_new_category_replacing_all_rows_is_critical(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        cur = current_stable_df.assign(product="brand_new")
        by_col = {r.column: r for r in PSI().compute(_ctx(reference_df, cur, model_spec))}
        assert by_col["product"].severity is Severity.CRITICAL

    def test_missing_column_in_current_is_error(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        cur = current_stable_df.drop(columns=["rate_spread"])
        for kpi_cls in _DRIFT_KPIS:
            by_col = {r.column: r for r in kpi_cls().compute(_ctx(reference_df, cur, model_spec))}
            assert by_col["rate_spread"].severity is Severity.ERROR, kpi_cls.__name__


class TestValuesStayBoundedAndFinite:
    @pytest.mark.parametrize("kpi_cls", [JensenShannon, KolmogorovSmirnov, CramersV])
    def test_bounded_metrics_in_unit_interval(
        self,
        reference_df: pd.DataFrame,
        current_drifted_df: pd.DataFrame,
        model_spec: ModelSpec,
        kpi_cls: type,
    ) -> None:
        for r in kpi_cls().compute(_ctx(reference_df, current_drifted_df, model_spec)):
            if r.severity is not Severity.ERROR:
                assert r.value is not None and 0.0 <= r.value <= 1.0 + 1e-12, r.to_record()

    def test_psi_is_non_negative_and_finite(
        self, reference_df: pd.DataFrame, current_drifted_df: pd.DataFrame, model_spec: ModelSpec
    ) -> None:
        for r in PSI().compute(_ctx(reference_df, current_drifted_df, model_spec)):
            assert r.value is not None and math.isfinite(r.value) and r.value >= 0.0
