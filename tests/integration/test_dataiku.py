"""Adaptateurs Dataiku exercés contre un faux SDK ``dataiku``."""

from __future__ import annotations

import sys
import types
from typing import Any

import pandas as pd
import pytest

from alm_monitoring.dataiku import (
    DataikuDatasetSource,
    dataiku_check,
    publish_scenario_variables,
    write_report,
)
from alm_monitoring.exceptions import DataikuIntegrationError
from alm_monitoring.kpis import KPIResult
from alm_monitoring.report import MonitoringReport
from alm_monitoring.types import KPICategory, Severity

pytestmark = pytest.mark.integration


class _FakeDataset:
    instances: list[_FakeDataset] = []

    def __init__(self, name: str, project_key: str | None = None) -> None:
        self.name = name
        self.project_key = project_key
        self.spec_item: dict[str, Any] = {}
        self.written: pd.DataFrame | None = None
        self.kwargs: dict[str, Any] = {}
        _FakeDataset.instances.append(self)

    def get_dataframe(self, **kwargs: Any) -> pd.DataFrame:
        self.kwargs = kwargs
        if self.name == "broken":
            raise RuntimeError("DSS down")
        return pd.DataFrame({"a": [1, 2]})

    def write_with_schema(self, frame: pd.DataFrame) -> None:
        if self.name == "readonly":
            raise RuntimeError("no write access")
        self.written = frame


class _FakeScenario:
    variables: dict[str, Any] = {}

    def set_scenario_variables(self, **variables: Any) -> None:
        if variables.get("alm_model_name") == "fail":
            raise RuntimeError("not in a scenario")
        _FakeScenario.variables = variables


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("dataiku")
    module.Dataset = _FakeDataset  # type: ignore[attr-defined]
    module.scenario = types.SimpleNamespace(Scenario=_FakeScenario)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dataiku", module)
    _FakeDataset.instances.clear()
    _FakeScenario.variables = {}
    return module


def _report(*severities: Severity, name: str = "m") -> MonitoringReport:
    results = tuple(
        KPIResult("psi", KPICategory.DATA_DRIFT, f"c{i}", 0.3 + i / 10, s)
        for i, s in enumerate(severities)
    )
    return MonitoringReport(name, "1", results, 10, 10)


class TestDatasetSource:
    def test_load_passes_options(self) -> None:
        frame = DataikuDatasetSource("ref", project_key="ALM", limit=5, columns=["a"]).load()
        assert len(frame) == 2
        ds = _FakeDataset.instances[0]
        assert ds.project_key == "ALM"
        assert ds.kwargs == {"infer_with_pandas": True, "limit": 5, "columns": ["a"]}

    def test_invalid_name(self) -> None:
        with pytest.raises(DataikuIntegrationError):
            DataikuDatasetSource("has space")
        with pytest.raises(DataikuIntegrationError):
            DataikuDatasetSource("")

    def test_sdk_error_wrapped(self) -> None:
        with pytest.raises(DataikuIntegrationError, match="Échec de lecture"):
            DataikuDatasetSource("broken").load()

    def test_missing_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "dataiku", None)  # type: ignore[arg-type]
        with pytest.raises(DataikuIntegrationError, match="qu'à l'intérieur de DSS"):
            DataikuDatasetSource("ref").load()


class TestWriteReport:
    def test_append(self) -> None:
        frame = write_report(_report(Severity.OK, Severity.WARNING), "monitoring_kpis")
        ds = _FakeDataset.instances[0]
        assert ds.spec_item == {"appendMode": True}
        assert ds.written is not None and len(ds.written) == 2
        assert list(frame.columns)[:2] == ["run_id", "run_at"]

    def test_overwrite(self) -> None:
        write_report(_report(Severity.OK), "monitoring_kpis", append=False)
        assert _FakeDataset.instances[0].spec_item == {}

    def test_write_error_wrapped(self) -> None:
        with pytest.raises(DataikuIntegrationError, match="Échec d'écriture"):
            write_report(_report(Severity.OK), "readonly")


class TestCheckAndVariables:
    def test_check_ok(self) -> None:
        assert dataiku_check(_report(Severity.OK)) == ("OK", "OK - 0 dépassement(s)")

    def test_check_warning_and_error(self) -> None:
        outcome, message = dataiku_check(_report(Severity.WARNING, Severity.CRITICAL))
        assert outcome == "ERROR"
        assert message.startswith("CRITICAL - 2 dépassement(s): psi[c1]=0.400, psi[c0]=0.300")
        assert dataiku_check(_report(Severity.WARNING))[0] == "WARNING"

    def test_check_with_error_result(self) -> None:
        failed = KPIResult(
            "auc", KPICategory.PERFORMANCE, None, None, Severity.ERROR, message="no labels"
        )
        report = MonitoringReport("m", "1", (failed,), 1, 1)
        outcome, message = dataiku_check(report)
        assert outcome == "ERROR"
        assert "auc: no labels" in message

    def test_scenario_variables(self) -> None:
        report = _report(Severity.WARNING)
        variables = publish_scenario_variables(report, {"custom": 1})
        assert variables["alm_overall_severity"] == "warning"
        assert variables["alm_n_breaches"] == 1
        assert variables["alm_run_id"] == report.run_id
        assert _FakeScenario.variables == variables

    def test_scenario_error_wrapped(self) -> None:
        with pytest.raises(DataikuIntegrationError, match="variables de scénario"):
            publish_scenario_variables(_report(Severity.OK, name="fail"))
