"""Tests de l'explicabilité et du champion-challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from alm_monitoring.exceptions import SchemaValidationError
from alm_monitoring.modeling import (
    compare_by_window,
    decide,
    economic_sign_check,
    partial_dependence_table,
    permutation_importance_table,
)

pytestmark = pytest.mark.unit


def _data(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, noise = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - 0.3)))).astype(int)
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
    return pd.DataFrame({"x1": x1, "noise": noise, "y": y, "issue_d": dates})


def test_permutation_importance_ranks_signal_first() -> None:
    frame = _data()
    model = LogisticRegression().fit(frame[["x1", "noise"]], frame["y"])
    table = permutation_importance_table(
        model, frame, target="y", features=["x1", "noise"], n_repeats=3
    )
    assert table.iloc[0]["feature"] == "x1"
    assert table.iloc[0]["importance_moyenne"] > table.iloc[1]["importance_moyenne"]


def test_partial_dependence_and_sign_check() -> None:
    frame = _data()
    model = LogisticRegression().fit(frame[["x1", "noise"]], frame["y"])
    pdp = partial_dependence_table(
        model, frame, features=["x1", "noise"], columns=["x1", "absent"], grid_resolution=10
    )
    assert set(pdp["feature"]) == {"x1"}
    assert len(pdp) == 10
    check = economic_sign_check(pdp, expected={"x1": "croissant", "noise": "croissant"})
    x1 = check.set_index("feature").loc["x1"]
    assert x1["observe"] == "croissant"
    assert bool(x1["conforme"]) is True
    assert check.set_index("feature").loc["noise", "observe"] == "indisponible"


def test_compare_by_window_and_decide() -> None:
    frame = _data(6000, seed=1)
    # champion mal calibré (coefficient trop faible), challenger = vraie probabilité
    frame["champion"] = 1 / (1 + np.exp(-(0.5 * frame["x1"] - 0.3)))
    frame["challenger"] = 1 / (1 + np.exp(-(1.5 * frame["x1"] - 0.3)))
    table = compare_by_window(
        frame,
        target="y",
        timestamp="issue_d",
        champion_score="champion",
        challenger_score="challenger",
        min_rows=200,
    )
    assert len(table) == 6
    assert (table["delta_brier"] < 0).all()
    promoted = decide(table)
    assert promoted.decision == "promouvoir"
    assert promoted.windows == 6
    assert promoted.to_dict()["share_better_brier"] == 1.0

    worse = table.assign(delta_brier=table["delta_brier"].abs())
    assert decide(worse).decision == "rejeter"
    mixed = table.assign(delta_calibration=0.01)
    assert decide(mixed).decision == "indetermine"

    with pytest.raises(SchemaValidationError, match="absente"):
        compare_by_window(
            frame,
            target="y",
            timestamp="issue_d",
            champion_score="champion",
            challenger_score="nope",
        )
    with pytest.raises(SchemaValidationError, match="fenêtre"):
        compare_by_window(
            frame,
            target="y",
            timestamp="issue_d",
            champion_score="champion",
            challenger_score="challenger",
            min_rows=10_000,
        )
