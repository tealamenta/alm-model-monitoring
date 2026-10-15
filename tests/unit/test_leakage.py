"""Tests de l'audit de fuite de données."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring import leakage
from alm_monitoring.exceptions import SchemaValidationError
from alm_monitoring.types import Severity

pytestmark = pytest.mark.unit


def _frame(n: int, *, year: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(shift, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * x1 + 0.1 * x2)))).astype(int)
    return pd.DataFrame(
        {
            "date": pd.Timestamp(f"{year}-01-01")
            + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
            "x1": x1,
            "x2": x2,
            "cat": rng.choice(["a", "b", "c"], n),
            "y": y,
            "leak": y + rng.normal(0, 0.05, n),  # révèle la cible
        }
    )


def test_forbidden_columns() -> None:
    ok = leakage.check_forbidden_columns(["x1", "x2"], ["total_pymnt"])
    assert ok.severity is Severity.OK
    bad = leakage.check_forbidden_columns(["x1", "total_pymnt"], ["total_pymnt"])
    assert bad.severity is Severity.CRITICAL
    assert bad.details["columns"] == ["total_pymnt"]


def test_temporal_split() -> None:
    reference, current = _frame(200, year=2014), _frame(200, year=2016)
    assert leakage.check_temporal_split(reference, current, "date").severity is Severity.OK
    overlap = leakage.check_temporal_split(current, current, "date")
    assert overlap.severity is Severity.CRITICAL
    with pytest.raises(SchemaValidationError, match="date"):
        leakage.check_temporal_split(reference.drop(columns="date"), current, "date")


def test_single_feature_power_flags_leaky_column() -> None:
    frame = _frame(2000, year=2014)
    clean = leakage.check_single_feature_power(frame, target="y", numeric_features=["x1", "x2"])
    assert clean.severity is Severity.OK
    assert set(clean.details["top_auc"]) == {"x1", "x2"}
    leaky = leakage.check_single_feature_power(frame, target="y", numeric_features=["x1", "leak"])
    assert leaky.severity is Severity.CRITICAL
    assert "leak" in leaky.details["suspicious"]


def test_adversarial_validation_distinguishes_shifted_windows() -> None:
    reference, same = _frame(1500, year=2014, seed=1), _frame(1500, year=2016, seed=2)
    shifted = _frame(1500, year=2016, shift=3.0, seed=3)
    quiet = leakage.adversarial_validation(
        reference, same, numeric_features=["x1", "x2"], categorical_features=["cat"]
    )
    assert quiet.severity is Severity.OK
    assert quiet.details["auc"] < 0.6
    loud = leakage.adversarial_validation(
        reference, shifted, numeric_features=["x1", "x2"], categorical_features=["cat"]
    )
    assert loud.severity is Severity.WARNING  # jamais CRITICAL seul : dérive ou fuite
    assert loud.details["auc"] > 0.9
    assert next(iter(loud.details["top_features"])) == "x1"


def test_audit_report_aggregates_findings() -> None:
    reference, current = _frame(1200, year=2014, seed=4), _frame(1200, year=2016, seed=5)
    report = leakage.audit(
        reference,
        current,
        target="y",
        timestamp="date",
        numeric_features=["x1", "x2"],
        categorical_features=["cat"],
        forbidden_columns=["leak"],
    )
    assert len(report.findings) == 4
    assert report.passed
    assert report.overall_severity is Severity.OK
    table = report.to_dataframe()
    assert list(table["check"]) == [
        "forbidden_columns",
        "temporal_split",
        "single_feature_power",
        "adversarial_validation",
    ]
    assert "Audit de fuite : OK" in report.summary()

    failing = leakage.audit(
        reference,
        current,
        target="y",
        timestamp="date",
        numeric_features=["x1", "leak"],
        categorical_features=[],
        forbidden_columns=["leak"],
    )
    assert not failing.passed
    assert failing.overall_severity is Severity.CRITICAL
