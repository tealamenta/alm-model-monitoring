"""Tests de la classe de base KPI, du conteneur de résultat, du contexte et du registre."""

from __future__ import annotations

import pandas as pd
import pytest

from alm_monitoring.config import KPISpec, ModelSpec, Thresholds
from alm_monitoring.exceptions import ConfigurationError, InsufficientDataError, UnknownKPIError
from alm_monitoring.kpis import (
    KPI,
    KPIResult,
    MonitoringContext,
    available_kpis,
    build_kpi,
    build_kpis,
    iter_kpi_metadata,
    register,
)
from alm_monitoring.kpis import base as kpi_base
from alm_monitoring.types import KPICategory, Severity, TaskType

pytestmark = pytest.mark.unit


class _Dummy(KPI):
    name = "dummy_kpi"
    category = KPICategory.DATA_DRIFT
    default_thresholds = Thresholds(warning=0.1, critical=0.2)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
        return [self.result(0.15)]


class _LowerWorse(KPI):
    name = "lower_worse"
    category = KPICategory.PERFORMANCE
    higher_is_worse = False
    default_thresholds = Thresholds(warning=0.8, critical=0.7)

    def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
        return []


class TestEvaluate:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.05, Severity.OK),
            (0.1, Severity.WARNING),
            (0.19, Severity.WARNING),
            (0.2, Severity.CRITICAL),
        ],
    )
    def test_higher_is_worse(self, value: float, expected: Severity) -> None:
        assert _Dummy().evaluate(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.9, Severity.OK), (0.8, Severity.WARNING), (0.7, Severity.CRITICAL)],
    )
    def test_lower_is_worse(self, value: float, expected: Severity) -> None:
        assert _LowerWorse().evaluate(value) is expected

    def test_nan_is_ok(self) -> None:
        assert _Dummy().evaluate(float("nan")) is Severity.OK

    def test_no_thresholds_is_ok(self) -> None:
        class NoThreshold(_Dummy):
            default_thresholds = None

        assert NoThreshold().evaluate(10.0) is Severity.OK

    def test_only_critical_threshold(self) -> None:
        kpi = _Dummy(thresholds=Thresholds(critical=0.5))
        assert kpi.evaluate(0.4) is Severity.OK
        assert kpi.evaluate(0.5) is Severity.CRITICAL


class TestThresholdValidation:
    def test_inconsistent_higher_is_worse(self) -> None:
        with pytest.raises(ConfigurationError, match="incohérents"):
            _Dummy(thresholds=Thresholds(warning=0.5, critical=0.1))

    def test_inconsistent_lower_is_worse(self) -> None:
        with pytest.raises(ConfigurationError, match="incohérents"):
            _LowerWorse(thresholds=Thresholds(warning=0.5, critical=0.9))

    def test_params_rejected_by_default(self) -> None:
        with pytest.raises(ConfigurationError, match="n'accepte pas de paramètres"):
            _Dummy(foo=1)

    def test_repr(self) -> None:
        assert "dummy" not in repr(_Dummy()) or "_Dummy(" in repr(_Dummy())


class TestResults:
    def test_result_and_record(self) -> None:
        res = _Dummy().result(0.15, column="x", details={"b": 1, "a": 2})
        assert res.severity is Severity.WARNING
        record = res.to_record()
        assert record["kpi"] == "dummy_kpi"
        assert record["warning_threshold"] == 0.1
        assert record["critical_threshold"] == 0.2
        assert list(record)[-2:] == ["detail_a", "detail_b"]

    def test_error_result(self) -> None:
        res = _Dummy().error_result("boom", column="x")
        assert res.severity is Severity.ERROR
        assert res.value is None
        assert res.message == "boom"
        assert res.to_record()["message"] == "boom"

    def test_result_without_thresholds_has_null_threshold_columns(self) -> None:
        res = KPIResult("k", KPICategory.PERFORMANCE, None, 1.0, Severity.OK)
        assert res.to_record()["warning_threshold"] is None


class TestContext:
    def test_column_selection(self, model_spec: ModelSpec) -> None:
        ctx = MonitoringContext(pd.DataFrame(), pd.DataFrame(), model_spec)
        assert ctx.numeric_columns() == ["age", "balance", "rate_spread", "tenure_months"]
        assert ctx.numeric_columns(["balance", "unknown"]) == ["balance"]
        assert ctx.categorical_columns(["channel"]) == ["channel"]
        assert ctx.all_feature_columns()[-1] == "channel"

    def test_has_labels(self, model_spec: ModelSpec) -> None:
        no_col = MonitoringContext(pd.DataFrame(), pd.DataFrame({"x": [1]}), model_spec)
        assert not no_col.has_labels()
        all_na = MonitoringContext(pd.DataFrame(), pd.DataFrame({"y": [None]}), model_spec)
        assert not all_na.has_labels()
        ok = MonitoringContext(pd.DataFrame(), pd.DataFrame({"y": [1, None]}), model_spec)
        assert ok.has_labels()

    def test_require_min_samples(self, model_spec: ModelSpec) -> None:
        small = pd.DataFrame({"x": range(5)})
        ctx = MonitoringContext(small, small, model_spec, min_sample_size=10)
        with pytest.raises(InsufficientDataError) as exc:
            ctx.require_min_samples(small, what="reference")
        assert exc.value.details == {"rows": 5, "min_sample_size": 10}


class TestRegistry:
    def test_all_documented_kpis_are_registered(self) -> None:
        expected = {
            "psi", "js_distance", "ks_test", "cramers_v", "prediction_drift",
            "auc", "brier_score", "log_loss", "calibration_error", "performance_degradation",
            "rmse", "mae", "r2",
            "missing_rate", "out_of_range_rate", "unseen_categories_rate", "schema_conformity",
        }  # fmt: skip
        assert expected <= set(available_kpis())

    def test_registry_returns_copy(self) -> None:
        available_kpis().clear()
        assert available_kpis()

    def test_metadata_is_sorted_and_complete(self) -> None:
        meta = list(iter_kpi_metadata())
        names = [m["name"] for m in meta]
        assert names == sorted(names)
        auc = next(m for m in meta if m["name"] == "auc")
        assert auc["higher_is_worse"] is False
        assert auc["requires_labels"] is True
        assert auc["tasks"] == ["binary_classification"]
        assert next(m for m in meta if m["name"] == "psi")["tasks"] == "all"

    def test_build_kpi_from_spec(self) -> None:
        kpi = build_kpi(KPISpec(name="psi", params={"bins": 5}, columns=["age"]))
        assert kpi.name == "psi"
        assert kpi.columns == ["age"]
        assert kpi.params == {"bins": 5}

    def test_unknown_kpi(self) -> None:
        with pytest.raises(UnknownKPIError) as exc:
            build_kpi(KPISpec(name="nope"))
        assert "psi" in exc.value.details["available"]

    def test_build_kpis_skips_disabled(self) -> None:
        kpis = build_kpis([KPISpec(name="psi"), KPISpec(name="auc", enabled=False)])
        assert [k.name for k in kpis] == ["psi"]

    def test_register_requires_name_and_uniqueness(self) -> None:
        class NoName(KPI):
            category = KPICategory.DATA_DRIFT

            def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
                return []

        with pytest.raises(ConfigurationError, match="doit définir un 'name'"):
            register(NoName)

        class Clash(KPI):
            name = "psi"
            category = KPICategory.DATA_DRIFT

            def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
                return []

        with pytest.raises(ConfigurationError, match="déjà enregistré"):
            register(Clash)

    def test_register_is_idempotent_for_same_class(self) -> None:
        cls = available_kpis()["psi"]
        assert register(cls) is cls

    def test_register_new_kpi_then_cleanup(self) -> None:
        register(_Dummy)
        try:
            assert available_kpis()["dummy_kpi"] is _Dummy
            assert build_kpi(KPISpec(name="dummy_kpi")).supports(TaskType.REGRESSION)
        finally:
            del kpi_base._REGISTRY["dummy_kpi"]
