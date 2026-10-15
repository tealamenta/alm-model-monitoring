"""Tests de fumée des figures Plotly (ignorés lorsque l'extra ``viz`` est absent)."""

from __future__ import annotations

import pandas as pd
import pytest

from alm_monitoring import ModelMonitor, MonitoringConfig
from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.report import MonitoringReport

pytest.importorskip("plotly")

from alm_monitoring import viz

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def history(
    synthetic_config: MonitoringConfig,
    reference_df: pd.DataFrame,
    current_stable_df: pd.DataFrame,
    current_drifted_df: pd.DataFrame,
) -> pd.DataFrame:
    monitor = ModelMonitor(synthetic_config)
    reports = [
        monitor.run(reference_df, current_stable_df, metadata={"window_start": "2024-02-01"}),
        monitor.run(reference_df, current_drifted_df, metadata={"window_start": "2024-03-01"}),
    ]
    return MonitoringReport.concat(reports)


def test_severity_heatmap(history: pd.DataFrame) -> None:
    fig = viz.severity_heatmap(history)
    (trace,) = fig.data
    assert trace.type == "heatmap"
    assert list(trace.x) == ["2024-02", "2024-03"]
    labels = history.drop_duplicates(["kpi", "column"])
    assert len(trace.y) == len(labels)

    with pytest.raises(ConfigurationError, match="vide"):
        viz.severity_heatmap(history.iloc[0:0])


def test_heatmap_falls_back_to_run_at(history: pd.DataFrame) -> None:
    fig = viz.severity_heatmap(history.drop(columns=["window_start"]))
    assert fig.data[0].type == "heatmap"
    # Une fenêtre absente sur certaines exécutions (fenêtre complète mêlée à des tranches)
    # fonctionne aussi.
    partial = history.copy()
    partial.loc[partial["window_start"] == "2024-03-01", "window_start"] = None
    assert viz.severity_heatmap(partial).data[0].type == "heatmap"


def test_kpi_timeseries(history: pd.DataFrame) -> None:
    fig = viz.kpi_timeseries(history, "psi", column="rate_spread")
    assert fig.data[0].type == "scatter"
    assert len(fig.data[0].x) == 2
    assert len(fig.layout.shapes) == 2  # lignes warning + critical
    assert "rate_spread" in fig.layout.title.text

    model_level = viz.kpi_timeseries(history, "auc", title="AUC")
    assert model_level.layout.title.text == "AUC"

    with pytest.raises(ConfigurationError, match="Aucune ligne"):
        viz.kpi_timeseries(history, "psi", column="does_not_exist")


def test_distribution_comparison(
    reference_df: pd.DataFrame, current_drifted_df: pd.DataFrame
) -> None:
    numeric = viz.distribution_comparison(
        reference_df["rate_spread"], current_drifted_df["rate_spread"], name="rate_spread"
    )
    assert [t.type for t in numeric.data] == ["histogram", "histogram"]
    assert numeric.layout.barmode == "overlay"

    categorical = viz.distribution_comparison(
        reference_df["product"], current_drifted_df["product"], name="product"
    )
    assert [t.type for t in categorical.data] == ["bar", "bar"]
    assert "structured_note" in categorical.data[0].x


def test_severity_summary(history: pd.DataFrame) -> None:
    single_run = history[history["run_id"] == history["run_id"].iloc[-1]]
    fig = viz.severity_summary(single_run)
    names = {t.name for t in fig.data}
    assert "critical" in names
    assert fig.layout.barmode == "stack"
