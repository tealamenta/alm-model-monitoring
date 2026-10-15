"""Tests du benchmark de modèles et de la règle de sélection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.modeling import (
    benchmark_table,
    build_candidates,
    compute_metrics,
    run_benchmark,
    select_champion,
)

pytestmark = pytest.mark.unit


def _data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b"], n)
    logit = 1.2 * x1 - 0.5 * x2 + np.where(cat == "a", 0.4, -0.4) - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x1[rng.random(n) < 0.05] = np.nan  # manquants : les candidats doivent les tolérer
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def test_compute_metrics_fields() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3])
    m = compute_metrics(y, p)
    assert m.rows == 8
    assert m.auc == 1.0
    assert m.positive_rate == 0.5
    assert m.level_bias == pytest.approx(m.mean_prediction - 0.5)
    assert set(m.to_dict()) >= {"auc", "brier", "log_loss", "calibration_error", "level_bias"}


def test_build_candidates_names() -> None:
    candidates = build_candidates(["x1", "x2"], ["cat"])
    assert list(candidates) == ["naive_prior", "logistic_regression", "hist_gradient_boosting"]


def test_run_benchmark_and_select_champion() -> None:
    reference, holdout = _data(3000, seed=1), _data(1500, seed=2)
    results = run_benchmark(
        reference,
        holdout,
        target="y",
        numeric_features=["x1", "x2"],
        categorical_features=["cat"],
        n_folds=3,
    )
    names = [r.name for r in results]
    assert names == ["naive_prior", "logistic_regression", "hist_gradient_boosting"]
    naive, logistic = results[0], results[1]
    assert naive.out_of_time.auc == pytest.approx(0.5)
    assert logistic.out_of_time.auc > 0.75
    assert len(logistic.oof_scores) == len(reference)
    champion, reason = select_champion(results)
    assert champion.name != "naive_prior"
    assert "Brier" in reason or "calibration" in reason
    table = benchmark_table(results)
    assert list(table.index) == [f"{n} [base]" for n in names]
    assert "pr_auc_hors_temps" in table.columns
    assert results[1].out_of_time.average_precision > 0.3
    assert {"auc_hors_temps", "ece_hors_temps", "biais_niveau_hors_temps"} <= set(table.columns)
    serialised = logistic.to_dict()
    assert serialised["in_time"]["rows"] == len(reference)
    assert "auc_decay" in serialised


def test_select_champion_falls_back_on_calibration() -> None:
    reference, holdout = _data(2000, seed=3), _data(800, seed=4)
    results = run_benchmark(
        reference,
        holdout,
        target="y",
        numeric_features=["x1", "x2"],
        categorical_features=["cat"],
        n_folds=2,
    )
    champion, reason = select_champion(results, max_calibration_error=0.0)
    assert "aucun candidat sous le seuil" in reason
    assert champion.name != "naive_prior"
    with pytest.raises(ConfigurationError, match="non trivial"):
        select_champion([results[0]])


def test_run_benchmark_rejects_single_fold() -> None:
    frame = _data(200, seed=5)
    with pytest.raises(ConfigurationError, match="n_folds"):
        run_benchmark(
            frame, frame, target="y", numeric_features=["x1"], categorical_features=[], n_folds=1
        )
