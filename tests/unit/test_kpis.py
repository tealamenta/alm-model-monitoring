"""Tests comportementaux des KPI de drift, de qualité des données et de performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.config import ModelSpec, Thresholds
from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.kpis import MonitoringContext, available_kpis
from alm_monitoring.kpis.data_quality import (
    MissingRate,
    OutOfRangeRate,
    SchemaConformity,
    UnseenCategoriesRate,
)
from alm_monitoring.kpis.drift import (
    PSI,
    CramersV,
    JensenShannon,
    KolmogorovSmirnov,
    PredictionDrift,
)
from alm_monitoring.kpis.performance import (
    AUC,
    MAE,
    R2,
    RMSE,
    BrierScore,
    ExpectedCalibrationError,
    LogLoss,
    PerformanceDegradation,
)
from alm_monitoring.types import Severity, TaskType

pytestmark = pytest.mark.unit


def _by_column(results: list) -> dict[str | None, object]:  # type: ignore[type-arg]
    return {r.column: r for r in results}


# ---------------------------------------------------------------------------- drift


class TestPSI:
    def test_stable_data_is_ok(self, ctx_stable: MonitoringContext) -> None:
        results = PSI().compute(ctx_stable)
        assert {r.column for r in results} == set(ctx_stable.all_feature_columns())
        assert all(r.severity is Severity.OK for r in results), [r.to_record() for r in results]

    def test_drift_is_detected(self, ctx_drifted: MonitoringContext) -> None:
        by_col = _by_column(PSI().compute(ctx_drifted))
        assert by_col["rate_spread"].severity is Severity.CRITICAL
        assert by_col["age"].severity is not Severity.OK
        assert by_col["product"].severity is not Severity.OK
        assert by_col["rate_spread"].details["n_current"] == len(ctx_drifted.current)

    def test_restricted_columns(self, ctx_stable: MonitoringContext) -> None:
        results = PSI(columns=["age", "segment"]).compute(ctx_stable)
        assert sorted(r.column for r in results) == ["age", "segment"]  # type: ignore[type-var]

    def test_missing_column_yields_error_result(self, ctx_stable: MonitoringContext) -> None:
        ctx = MonitoringContext(
            ctx_stable.reference, ctx_stable.current.drop(columns=["age"]), ctx_stable.model
        )
        by_col = _by_column(PSI().compute(ctx))
        assert by_col["age"].severity is Severity.ERROR
        assert by_col["balance"].severity is Severity.OK

    def test_insufficient_sample_raises(self, ctx_stable: MonitoringContext) -> None:
        ctx = MonitoringContext(
            ctx_stable.reference, ctx_stable.current.head(10), ctx_stable.model
        )
        from alm_monitoring.exceptions import InsufficientDataError

        with pytest.raises(InsufficientDataError):
            PSI().compute(ctx)

    @pytest.mark.parametrize("bins", [1, 101, 2.5, True, "10"])
    def test_invalid_bins(self, bins: object) -> None:
        with pytest.raises(ConfigurationError):
            PSI(bins=bins)

    def test_unknown_param(self) -> None:
        with pytest.raises(ConfigurationError, match="Paramètres inconnus"):
            PSI(foo=1)

    def test_bins_default(self) -> None:
        assert PSI().bins == 10
        assert PSI(bins=5).bins == 5


class TestOtherDriftKPIs:
    def test_js_distance_bounds_and_detection(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        stable = _by_column(JensenShannon().compute(ctx_stable))
        drifted = _by_column(JensenShannon().compute(ctx_drifted))
        for col in stable:
            assert 0.0 <= stable[col].value <= 1.0
        assert drifted["rate_spread"].value > stable["rate_spread"].value
        assert drifted["rate_spread"].severity is Severity.CRITICAL

    def test_ks_numeric_only(self, ctx_drifted: MonitoringContext) -> None:
        results = KolmogorovSmirnov().compute(ctx_drifted)
        assert {r.column for r in results} == set(ctx_drifted.numeric_columns())
        rate = _by_column(results)["rate_spread"]
        assert rate.severity is Severity.CRITICAL
        assert rate.details["p_value"] < 1e-6

    def test_ks_rejects_bins_param(self) -> None:
        with pytest.raises(ConfigurationError):
            KolmogorovSmirnov(bins=5)

    def test_cramers_v_categorical_only(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        stable = _by_column(CramersV().compute(ctx_stable))
        assert set(stable) == set(ctx_stable.categorical_columns())
        assert all(r.severity is Severity.OK for r in stable.values())
        drifted = _by_column(CramersV().compute(ctx_drifted))
        assert drifted["product"].value > stable["product"].value

    def test_prediction_drift(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        (stable,) = PredictionDrift().compute(ctx_stable)
        (drifted,) = PredictionDrift().compute(ctx_drifted)
        assert stable.column == "score"
        assert stable.severity is Severity.OK
        assert drifted.severity is Severity.CRITICAL
        assert drifted.details["mean_current"] != pytest.approx(drifted.details["mean_reference"])


# --------------------------------------------------------------------- data quality


class TestDataQuality:
    def test_missing_rate(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        stable = _by_column(MissingRate().compute(ctx_stable))
        assert all(r.value == 0.0 and r.severity is Severity.OK for r in stable.values())
        drifted = _by_column(MissingRate().compute(ctx_drifted))
        assert drifted["balance"].value == pytest.approx(0.30, abs=0.03)
        assert drifted["balance"].severity is Severity.CRITICAL
        assert drifted["balance"].details["reference_rate"] == 0.0

    def test_missing_rate_missing_column(self, ctx_stable: MonitoringContext) -> None:
        ctx = MonitoringContext(
            ctx_stable.reference, ctx_stable.current.drop(columns=["segment"]), ctx_stable.model
        )
        assert _by_column(MissingRate().compute(ctx))["segment"].severity is Severity.ERROR

    def test_out_of_range(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        stable = _by_column(OutOfRangeRate().compute(ctx_stable))
        assert all(r.value < 0.01 for r in stable.values())
        drifted = _by_column(OutOfRangeRate().compute(ctx_drifted))
        assert drifted["rate_spread"].severity is Severity.CRITICAL
        assert drifted["rate_spread"].details["upper_bound"] < drifted["rate_spread"].value + 10

    def test_out_of_range_quantile_params(self, ctx_stable: MonitoringContext) -> None:
        kpi = OutOfRangeRate(lower_q=0.05, upper_q=0.95)
        result = _by_column(kpi.compute(ctx_stable))["age"]
        assert result.value == pytest.approx(0.10, abs=0.03)

    @pytest.mark.parametrize(
        "params", [{"lower_q": -0.1}, {"upper_q": 1.5}, {"lower_q": 0.6, "upper_q": 0.4}]
    )
    def test_out_of_range_invalid_params(self, params: dict[str, float]) -> None:
        with pytest.raises(ConfigurationError):
            OutOfRangeRate(**params)

    def test_out_of_range_error_cases(self, ctx_stable: MonitoringContext) -> None:
        current = ctx_stable.current.copy()
        current["age"] = np.nan
        ctx = MonitoringContext(
            ctx_stable.reference, current.drop(columns=["balance"]), ctx_stable.model
        )
        by_col = _by_column(OutOfRangeRate().compute(ctx))
        assert by_col["age"].severity is Severity.ERROR
        assert by_col["balance"].severity is Severity.ERROR

    def test_unseen_categories(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        stable = _by_column(UnseenCategoriesRate().compute(ctx_stable))
        assert all(r.value == 0.0 for r in stable.values())
        drifted = _by_column(UnseenCategoriesRate().compute(ctx_drifted))
        product = drifted["product"]
        assert product.value == pytest.approx(0.10, abs=0.03)
        assert product.severity is Severity.CRITICAL
        assert product.details["unseen_levels_sample"] == ["structured_note"]

    def test_unseen_categories_error_cases(self, ctx_stable: MonitoringContext) -> None:
        current = ctx_stable.current.copy()
        current["segment"] = pd.array([None] * len(current), dtype="string")
        ctx = MonitoringContext(
            ctx_stable.reference, current.drop(columns=["channel"]), ctx_stable.model
        )
        by_col = _by_column(UnseenCategoriesRate().compute(ctx))
        assert by_col["segment"].severity is Severity.ERROR
        assert by_col["channel"].severity is Severity.ERROR

    def test_schema_conformity_ok(self, ctx_stable: MonitoringContext) -> None:
        (result,) = SchemaConformity().compute(ctx_stable)
        assert result.value == 0.0
        assert result.severity is Severity.OK
        assert result.column is None

    def test_schema_conformity_violations(self, ctx_stable: MonitoringContext) -> None:
        current = ctx_stable.current.drop(columns=["age"]).copy()
        current["balance"] = current["balance"].astype("string")
        ctx = MonitoringContext(ctx_stable.reference, current, ctx_stable.model)
        (result,) = SchemaConformity().compute(ctx)
        assert result.value == 2.0
        assert result.severity is Severity.CRITICAL
        issues = {v["column"]: v["issue"] for v in result.details["violations"]}
        assert issues["age"] == "missing"
        assert issues["balance"].startswith("not numeric")

    def test_quality_kpis_reject_unknown_params(self) -> None:
        with pytest.raises(ConfigurationError):
            MissingRate(foo=1)


# ---------------------------------------------------------------------- performance


class TestBinaryPerformance:
    def test_stable_model_is_healthy(self, ctx_stable: MonitoringContext) -> None:
        for cls in (AUC, BrierScore, LogLoss, ExpectedCalibrationError):
            (result,) = cls().compute(ctx_stable)
            assert result.severity is Severity.OK, (cls.name, result.to_record())
            assert result.details["n_labelled"] == len(ctx_stable.current)
            assert result.details["reference_value"] is not None

    def test_degraded_model_is_flagged(self, ctx_drifted: MonitoringContext) -> None:
        (auc,) = AUC().compute(ctx_drifted)
        (ece,) = ExpectedCalibrationError().compute(ctx_drifted)
        assert auc.severity is Severity.CRITICAL
        assert ece.severity is Severity.CRITICAL

    def test_performance_degradation(
        self, ctx_stable: MonitoringContext, ctx_drifted: MonitoringContext
    ) -> None:
        (stable,) = PerformanceDegradation().compute(ctx_stable)
        (drifted,) = PerformanceDegradation().compute(ctx_drifted)
        assert abs(stable.value) < 0.05
        assert drifted.value > 0.10
        assert drifted.severity is Severity.CRITICAL
        assert drifted.details["reference_value"] > drifted.details["current_value"]

    def test_missing_labels_gives_error_result(self, ctx_stable: MonitoringContext) -> None:
        ctx = MonitoringContext(
            ctx_stable.reference, ctx_stable.current.drop(columns=["y"]), ctx_stable.model
        )
        (auc,) = AUC().compute(ctx)
        assert auc.severity is Severity.ERROR
        assert "absentes" in (auc.message or "")
        (deg,) = PerformanceDegradation().compute(ctx)
        assert deg.severity is Severity.ERROR

    def test_reference_without_labels_still_scores_current(
        self, ctx_stable: MonitoringContext
    ) -> None:
        ctx = MonitoringContext(
            ctx_stable.reference.drop(columns=["y"]), ctx_stable.current, ctx_stable.model
        )
        (auc,) = AUC().compute(ctx)
        assert auc.severity is Severity.OK
        assert auc.details["reference_value"] is None

    def test_too_few_labels(self, ctx_stable: MonitoringContext) -> None:
        current = ctx_stable.current.copy()
        current.loc[current.index[50:], "y"] = np.nan
        ctx = MonitoringContext(ctx_stable.reference, current, ctx_stable.model)
        (auc,) = AUC().compute(ctx)
        assert auc.severity is Severity.ERROR
        assert "Pas assez d'observations labellisées" in (auc.message or "")

    def test_invalid_encodings(self, ctx_stable: MonitoringContext) -> None:
        bad_target = ctx_stable.current.assign(y=2)
        ctx = MonitoringContext(ctx_stable.reference, bad_target, ctx_stable.model)
        assert "0/1" in (AUC().compute(ctx)[0].message or "")

        bad_score = ctx_stable.current.assign(score=1.5)
        ctx = MonitoringContext(ctx_stable.reference, bad_score, ctx_stable.model)
        assert "[0, 1]" in (AUC().compute(ctx)[0].message or "")

        one_class = ctx_stable.current.assign(y=0)
        ctx = MonitoringContext(ctx_stable.reference, one_class, ctx_stable.model)
        assert "deux classes" in (AUC().compute(ctx)[0].message or "")

    def test_task_mismatch(self, ctx_stable: MonitoringContext) -> None:
        reg_spec = ctx_stable.model.model_copy(update={"task": TaskType.REGRESSION})
        ctx = MonitoringContext(ctx_stable.reference, ctx_stable.current, reg_spec)
        assert not AUC().supports(TaskType.REGRESSION)
        (result,) = AUC().compute(ctx)
        assert result.severity is Severity.ERROR

    def test_unknown_param_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            AUC(foo=1)


class TestRegressionPerformance:
    @pytest.fixture
    def reg_ctx(self) -> MonitoringContext:
        rng = np.random.default_rng(0)
        n = 500
        x = rng.normal(size=n)
        y = 2 * x + rng.normal(scale=0.5, size=n)
        frame = pd.DataFrame({"x": x, "target": y, "pred": 2 * x})
        spec = ModelSpec.model_validate(
            {
                "name": "reg",
                "version": "1",
                "task": "regression",
                "target": "target",
                "prediction": "pred",
                "features": {"numeric": ["x"]},
            }
        )
        return MonitoringContext(frame, frame, spec)

    def test_metrics(self, reg_ctx: MonitoringContext) -> None:
        (rmse,) = RMSE().compute(reg_ctx)
        (mae,) = MAE().compute(reg_ctx)
        (r2,) = R2().compute(reg_ctx)
        assert rmse.value == pytest.approx(0.5, abs=0.1)
        assert mae.value < rmse.value
        assert r2.value > 0.9
        assert r2.severity is Severity.OK
        assert rmse.thresholds is None and rmse.severity is Severity.OK

    def test_configured_thresholds(self, reg_ctx: MonitoringContext) -> None:
        (rmse,) = RMSE(thresholds=Thresholds(warning=0.1, critical=0.3)).compute(reg_ctx)
        assert rmse.severity is Severity.CRITICAL

    def test_degradation_uses_r2(self, reg_ctx: MonitoringContext) -> None:
        (deg,) = PerformanceDegradation().compute(reg_ctx)
        assert deg.value == pytest.approx(0.0, abs=1e-9)

    def test_degradation_non_positive_reference(self, reg_ctx: MonitoringContext) -> None:
        bad_ref = reg_ctx.reference.assign(pred=reg_ctx.reference["target"].mean())
        ctx = MonitoringContext(bad_ref, reg_ctx.current, reg_ctx.model)
        (deg,) = PerformanceDegradation().compute(ctx)
        assert deg.severity is Severity.ERROR

    def test_binary_kpis_not_applicable(self, reg_ctx: MonitoringContext) -> None:
        assert not AUC().supports(reg_ctx.model.task)
        assert RMSE().supports(reg_ctx.model.task)


def test_every_registered_kpi_runs_on_synthetic_data(ctx_drifted: MonitoringContext) -> None:
    """Test de fumée : aucun KPI enregistré ne lève sur un dataset drifté mais valide."""
    for name, cls in available_kpis().items():
        kpi = cls()
        if not kpi.supports(ctx_drifted.model.task):
            continue
        results = kpi.compute(ctx_drifted)
        assert results, name
        assert all(r.kpi == name for r in results)


class TestLabelCoverage:
    def test_full_coverage_is_ok(self, ctx_stable: MonitoringContext) -> None:
        from alm_monitoring.kpis.data_quality import LabelCoverage

        (r,) = LabelCoverage().compute(ctx_stable)
        assert r.severity is Severity.OK
        assert r.value == pytest.approx(1.0)
        assert r.details["n_total"] == r.details["n_labelled"] == len(ctx_stable.current)
        assert r.details["excluded_rate"] == pytest.approx(0.0)

    def test_partial_labels_degrade_severity_and_are_visible_in_performance_details(
        self, ctx_stable: MonitoringContext
    ) -> None:
        """40 % de labels manquants : label_coverage CRITICAL, et chaque KPI de performance
        expose l'assiette réelle (n_labelled / n_total / excluded_rate)."""
        from alm_monitoring.kpis.data_quality import LabelCoverage

        current = ctx_stable.current.copy()
        n = len(current)
        current.loc[current.index[: int(0.4 * n)], ctx_stable.model.target] = np.nan
        ctx = MonitoringContext(
            reference=ctx_stable.reference, current=current, model=ctx_stable.model
        )
        (cov,) = LabelCoverage().compute(ctx)
        assert cov.severity is Severity.CRITICAL
        assert cov.value == pytest.approx(0.6, abs=1e-3)
        assert cov.details["target_missing_rate"] == pytest.approx(0.4, abs=1e-3)

        (auc,) = AUC().compute(ctx)
        assert auc.severity is not Severity.ERROR
        assert auc.details["n_total"] == n
        assert auc.details["n_labelled"] == pytest.approx(0.6 * n, abs=1)
        assert auc.details["excluded_rate"] == pytest.approx(0.4, abs=1e-3)

    def test_no_target_column_is_zero_coverage(self, ctx_stable: MonitoringContext) -> None:
        from alm_monitoring.kpis.data_quality import LabelCoverage

        current = ctx_stable.current.drop(columns=[ctx_stable.model.target])
        ctx = MonitoringContext(
            reference=ctx_stable.reference, current=current, model=ctx_stable.model
        )
        (r,) = LabelCoverage().compute(ctx)
        assert r.severity is Severity.CRITICAL
        assert r.value == 0.0
        assert r.details["missing_columns"] == [ctx_stable.model.target]
