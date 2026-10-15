"""Tests du conteneur de rapport, des types et des exceptions."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.config import Thresholds
from alm_monitoring.exceptions import AlmMonitoringError, ConfigurationError, UnknownKPIError
from alm_monitoring.kpis import KPIResult
from alm_monitoring.report import MonitoringReport, _json_default
from alm_monitoring.types import KPICategory, Severity, TaskType

pytestmark = pytest.mark.unit


def _result(
    kpi: str, severity: Severity, column: str | None = None, **details: object
) -> KPIResult:
    return KPIResult(
        kpi=kpi,
        category=KPICategory.PERFORMANCE if kpi == "auc" else KPICategory.DATA_DRIFT,
        column=column,
        value=None if severity is Severity.ERROR else 0.5,
        severity=severity,
        thresholds=Thresholds(warning=0.1, critical=0.2),
        details=dict(details),
        message="failed" if severity is Severity.ERROR else None,
    )


@pytest.fixture
def report() -> MonitoringReport:
    return MonitoringReport(
        model_name="m",
        model_version="1",
        results=(
            _result("psi", Severity.OK, "a", n=np.int64(3)),
            _result("psi", Severity.WARNING, "b"),
            _result("auc", Severity.CRITICAL),
            _result("psi", Severity.ERROR, "c"),
        ),
        reference_rows=100,
        current_rows=50,
        metadata={"window_start": "2024-01-01", "api_token": "s3cr3t"},
    )


class TestSeverity:
    def test_ordering(self) -> None:
        assert Severity.OK < Severity.WARNING < Severity.CRITICAL < Severity.ERROR
        assert max(Severity.WARNING, Severity.OK) is Severity.WARNING
        assert Severity.CRITICAL >= Severity.CRITICAL

    def test_values(self) -> None:
        assert Severity("warning") is Severity.WARNING
        assert TaskType("regression") is TaskType.REGRESSION
        assert KPICategory("data_quality") is KPICategory.DATA_QUALITY


class TestExceptions:
    def test_message_and_details(self) -> None:
        exc = UnknownKPIError("Unknown", details={"available": ["psi"]})
        assert isinstance(exc, ConfigurationError)
        assert isinstance(exc, AlmMonitoringError)
        assert exc.code
        assert "available" in str(exc)
        payload = exc.to_dict()
        assert payload["details"] == {"available": ["psi"]}
        assert payload["code"] == exc.code

    def test_without_details(self) -> None:
        exc = AlmMonitoringError("plain")
        assert str(exc) == "plain"
        assert exc.details == {}


class TestReport:
    def test_overall_and_breaches(self, report: MonitoringReport) -> None:
        assert report.overall_severity is Severity.ERROR
        assert [r.column for r in report.breaches()] == ["b", None, "c"]
        assert [r.column for r in report.breaches(Severity.CRITICAL)] == [None, "c"]
        assert len(report.by_category(KPICategory.PERFORMANCE)) == 1

    def test_empty_report_is_ok(self) -> None:
        empty = MonitoringReport("m", "1", (), 1, 1)
        assert empty.overall_severity is Severity.OK
        assert empty.breaches() == []
        assert MonitoringReport.concat([]).empty

    def test_summary(self, report: MonitoringReport) -> None:
        summary = report.summary()
        assert summary["overall_severity"] == "error"
        assert summary["n_results"] == 4
        assert summary["severity_counts"] == {"ok": 1, "warning": 1, "critical": 1, "error": 1}
        assert summary["category_worst_severity"]["performance"] == "critical"
        assert summary["category_worst_severity"]["data_quality"] == "ok"
        assert summary["metadata"]["window_start"] == "2024-01-01"
        datetime.fromisoformat(summary["run_at"])

    def test_run_metadata_defaults(self) -> None:
        rep = MonitoringReport("m", "1", (), 1, 1)
        assert len(rep.run_id) == 36
        assert rep.run_at.tzinfo is UTC

    def test_to_dataframe_shape(self, report: MonitoringReport) -> None:
        frame = report.to_dataframe()
        assert len(frame) == 4
        assert list(frame.columns[:5]) == [
            "run_id", "run_at", "window_start", "model_name", "model_version"
        ]  # fmt: skip
        assert (frame["window_start"] == "2024-01-01").all()
        assert "detail_n" in frame.columns
        assert frame["detail_n"].iloc[0] == 3
        assert frame["severity"].tolist() == ["ok", "warning", "critical", "error"]
        assert frame["run_id"].nunique() == 1

    def test_to_dict_and_json(self, report: MonitoringReport) -> None:
        payload = json.loads(report.to_json())
        assert payload["model_name"] == "m"
        assert len(payload["results"]) == 4
        assert payload["results"][0]["thresholds"] == {"warning": 0.1, "critical": 0.2}
        assert payload["results"][0]["detail_n"] == 3

    def test_concat(self, report: MonitoringReport) -> None:
        other = MonitoringReport("m", "1", report.results, 1, 1)
        frame = MonitoringReport.concat([report, other])
        assert len(frame) == 8
        assert frame["run_id"].nunique() == 2

    def test_immutable(self, report: MonitoringReport) -> None:
        with pytest.raises(AttributeError):
            report.model_name = "x"  # type: ignore[misc]


class TestJsonDefault:
    def test_supported_types(self) -> None:
        now = datetime.now(tz=UTC)
        assert _json_default(now) == now.isoformat()
        assert _json_default(np.float64(1.5)) == 1.5
        assert _json_default({"a": 1}) == {"a": 1}
        assert _json_default(frozenset({1})) == [1]
        assert _json_default((1, 2)) == [1, 2]
        assert _json_default(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"

    def test_unsupported_type(self) -> None:
        with pytest.raises(TypeError):
            _json_default(object())
