"""Le backend PySpark doit reproduire les valeurs du backend pandas (test lent, optionnel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.kpis import _stats

pyspark = pytest.importorskip("pyspark")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("alm-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _frames(seed: int, shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame(
        {
            "amount": rng.lognormal(9, 0.5, 3000),
            "count": rng.integers(0, 4, 3000).astype(float),
            "grade": rng.choice(list("ABCD"), 3000, p=[0.4, 0.3, 0.2, 0.1]),
            "issue_d": pd.Timestamp("2015-01-01"),
        }
    )
    cur = pd.DataFrame(
        {
            "amount": rng.lognormal(9 + shift, 0.5, 1500),
            "count": rng.integers(0, 4, 1500).astype(float),
            "grade": rng.choice(list("ABCD"), 1500, p=[0.2, 0.3, 0.3, 0.2]),
            "issue_d": pd.to_datetime(rng.choice(["2016-01-01", "2016-02-01"], 1500)),
        }
    )
    cur.loc[:20, "amount"] = np.nan
    return ref, cur


def test_spark_matches_pandas(spark) -> None:
    from alm_monitoring.spark.drift import (
        compute_drift,
        drift_by_period,
        missing_rates,
    )

    ref, cur = _frames(seed=7, shift=0.4)
    table = compute_drift(
        spark.createDataFrame(ref),
        spark.createDataFrame(cur),
        numeric=["amount", "count"],
        categorical=["grade"],
    ).set_index("column")

    ref_a, cur_a = _stats.clean_numeric(ref["amount"]), _stats.clean_numeric(cur["amount"])
    edges = _stats.reference_bin_edges(ref_a, 10)
    expected_psi = _stats.psi(
        _stats.binned_frequencies(ref_a, edges), _stats.binned_frequencies(cur_a, edges)
    )
    assert table.loc["amount", "psi"] == pytest.approx(expected_psi, abs=0.02)
    assert table.loc["amount", "psi"] > 0.25  # dérive injectée

    # Variable à faible cardinalité : une classe par valeur distincte, comme en pandas.
    assert table.loc["count", "levels"] == 4
    assert table.loc["count", "psi"] < 0.05

    ref_g, cur_g, _ = _stats.categorical_frequencies(ref["grade"], cur["grade"])
    assert table.loc["grade", "psi"] == pytest.approx(_stats.psi(ref_g, cur_g), abs=1e-9)
    assert table.loc["grade", "levels"] == 4

    missing = missing_rates(spark.createDataFrame(cur), ["amount", "count"]).set_index("column")
    assert missing.loc["amount", "missing_rate"] == pytest.approx(21 / 1500)
    assert missing.loc["count", "missing_rate"] == 0.0

    by_period = drift_by_period(
        spark.createDataFrame(ref),
        spark.createDataFrame(cur),
        timestamp="issue_d",
        numeric=["amount"],
    )
    assert sorted(by_period["window_start"].dt.strftime("%Y-%m")) == ["2016-01", "2016-02"]


def test_constant_reference_is_detected_in_both_backends(spark) -> None:
    """Référence constante, courant décalé : les deux backends doivent alerter (régression)."""
    from alm_monitoring.spark.drift import compute_drift

    ref = pd.DataFrame({"flag": np.full(500, 1.0)})
    cur = pd.DataFrame({"flag": np.full(300, 2.0)})
    table = compute_drift(
        spark.createDataFrame(ref), spark.createDataFrame(cur), numeric=["flag"], categorical=[]
    ).set_index("column")

    edges = _stats.reference_bin_edges(ref["flag"].to_numpy(), 10)
    expected = _stats.psi(
        _stats.binned_frequencies(ref["flag"].to_numpy(), edges),
        _stats.binned_frequencies(cur["flag"].to_numpy(), edges),
    )
    assert expected > 0.25
    assert table.loc["flag", "psi"] == pytest.approx(expected, rel=1e-6)
