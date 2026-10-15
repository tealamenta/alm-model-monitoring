"""Tests de la préparation Lending Club : cible à horizon fixe, censure, découpe temporelle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.data import lending_club as lc
from alm_monitoring.exceptions import DataSourceError, SchemaValidationError

pytestmark = pytest.mark.unit


def _raw(
    n: int = 12, *, issue: str = "Jan-2015", status: str = "Fully Paid", last: str = "Jun-2015"
) -> pd.DataFrame:
    base = {
        "id": [str(i) for i in range(n)],
        "loan_amnt": np.linspace(1000, 30000, n),
        "term": [" 36 months"] * n,
        "int_rate": np.linspace(6, 25, n),
        "installment": np.linspace(50, 900, n),
        "grade": ["A", "B", "C", "D"] * (n // 4),
        "emp_length": ["10+ years", "< 1 year", "3 years", None] * (n // 4),
        "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"] * (n // 4),
        "annual_inc": np.linspace(20000, 200000, n),
        "verification_status": ["Verified"] * n,
        "issue_d": [issue] * n,
        "loan_status": [status] * n,
        "purpose": ["car"] * n,
        "dti": np.linspace(1, 30, n),
        "fico_range_low": np.full(n, 700.0),
        "fico_range_high": np.full(n, 704.0),
        "inq_last_6mths": np.zeros(n),
        "open_acc": np.full(n, 8.0),
        "pub_rec": np.zeros(n),
        "revol_bal": np.linspace(0, 50000, n),
        "revol_util": np.linspace(0, 100, n),
        "total_acc": np.full(n, 20.0),
        "mort_acc": np.ones(n),
        "pub_rec_bankruptcies": np.zeros(n),
        "application_type": ["Individual"] * n,
        "initial_list_status": ["w"] * n,
        "last_pymnt_d": [last] * n,
        "earliest_cr_line": ["Jan-2005"] * n,
        "total_pymnt": np.linspace(1000, 30000, n),  # colonne post-octroi : doit disparaître
    }
    return pd.DataFrame(base)


def test_prepare_builds_target_and_drops_post_origination_columns() -> None:
    prepared = lc.prepare_lending_club(_raw())
    assert lc.TARGET in prepared.columns
    assert "total_pymnt" not in prepared.columns
    assert "loan_status" not in prepared.columns
    assert set(lc.NUMERIC_FEATURES) <= set(prepared.columns)
    assert prepared[lc.TARGET].eq(1).all()  # soldé 5 mois après l'émission, terme 36 mois
    assert prepared["fico"].iloc[0] == pytest.approx(702.0)
    assert prepared["term_months"].iloc[0] == 36
    assert prepared["emp_length_years"].tolist()[:3] == [10.0, 0.5, 3.0]
    assert np.isnan(prepared["emp_length_years"].iloc[3])


@pytest.mark.parametrize(
    ("status", "last", "expected"),
    [
        ("Fully Paid", "Jun-2015", 1),  # anticipé dans l'horizon
        ("Fully Paid", "Jun-2016", 0),  # soldé mais après l'horizon de 12 mois
        ("Charged Off", "Jun-2015", 0),  # défaut, pas un remboursement
        ("Current", None, 0),  # encore en cours
    ],
)
def test_target_definition(status: str, last: str | None, expected: int) -> None:
    prepared = lc.prepare_lending_club(_raw(status=status, last=last))
    assert prepared[lc.TARGET].eq(expected).all()


def test_fully_paid_at_term_is_not_prepayment() -> None:
    # Terme 36 mois, soldé au 36e mois : remboursement normal, hors horizon de toute façon,
    # mais vérifions avec un horizon large que la condition « avant terme » joue.
    prepared = lc.prepare_lending_club(
        _raw(last="Jan-2018"), horizon_months=48, snapshot_date="2019-03-01"
    )
    assert prepared[lc.TARGET].eq(0).all()


def test_unobservable_cohorts_are_dropped() -> None:
    recent = _raw(issue="Jun-2018", last="Dec-2018")
    prepared = lc.prepare_lending_club(recent, snapshot_date="2019-03-01")
    assert prepared.empty  # 12 mois non écoulés entre juin 2018 et mars 2019


def test_prepare_requires_raw_columns() -> None:
    with pytest.raises(SchemaValidationError, match="manquantes"):
        lc.prepare_lending_club(_raw().drop(columns=["last_pymnt_d"]))


def test_split_reference_current_is_strictly_ordered() -> None:
    frames = [
        lc.prepare_lending_club(_raw(issue=f"Jan-{year}", last=f"Jun-{year}"))
        for year in (2013, 2014, 2016)
    ]
    prepared = pd.concat(frames, ignore_index=True)
    reference, current = lc.split_reference_current(prepared)
    assert reference[lc.TIMESTAMP].max() < current[lc.TIMESTAMP].min()
    assert len(reference) == 24
    assert len(current) == 12
    with pytest.raises(SchemaValidationError, match="avant la fenêtre"):
        lc.split_reference_current(
            prepared,
            reference_period=("2013-01-01", "2016-12-31"),
            current_period=("2016-01-01", "2017-12-31"),
        )


def test_cohort_summary_by_year() -> None:
    prepared = lc.prepare_lending_club(_raw(issue="Mar-2014", last="Jun-2014"))
    summary = lc.cohort_summary(prepared)
    assert list(summary.index) == [2014]
    assert summary.loc[2014, "prets"] == 12
    assert summary.loc[2014, "taux_prepayment"] == 1.0


def test_clean_impossible_values_sets_nan_and_keeps_rows() -> None:
    raw = _raw()
    raw.loc[0, "dti"] = 999.0
    raw.loc[1, "annual_inc"] = 0.0
    raw.loc[2, "revol_util"] = 250.0
    prepared = lc.prepare_lending_club(raw)
    assert len(prepared) == len(raw)
    assert np.isnan(prepared.loc[0, "dti"])
    assert np.isnan(prepared.loc[1, "annual_inc"])
    assert np.isnan(prepared.loc[2, "revol_util"])
    untouched = lc.prepare_lending_club(raw, clean=False)
    assert untouched.loc[0, "dti"] == 999.0


def test_engineer_features_adds_business_columns() -> None:
    prepared = lc.prepare_lending_club(_raw())
    prepared.loc[0, "mort_acc"] = np.nan
    engineered = lc.engineer_features(prepared)
    assert set(lc.ENGINEERED_NUMERIC_FEATURES) <= set(engineered.columns)
    assert set(lc.ENGINEERED_CATEGORICAL_FEATURES) <= set(engineered.columns)
    assert engineered["loan_to_income"].iloc[0] == pytest.approx(1000 / 20000, rel=1e-5)
    # un seul mois d'émission : l'écart au marché somme à zéro
    assert engineered["rate_spread"].sum() == pytest.approx(0.0, abs=1e-3)
    assert engineered["credit_history_years"].iloc[0] == pytest.approx(10.0, abs=0.1)
    assert engineered["mort_acc_missing"].tolist()[:2] == [1.0, 0.0]
    assert engineered["fico_band"].iloc[0] == "700-739"


def test_load_from_sample_file(tmp_path) -> None:
    raw = _raw()
    raw.loc[0, "id"] = "ABC"  # identifiants mixtes
    path = tmp_path / "sample.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")
    loaded = lc.load_lending_club(path, max_rows=8)
    assert len(loaded) == 8
    assert loaded["loan_amnt"].dtype == "float32"
    assert str(loaded["grade"].dtype) == "category"
    with pytest.raises(DataSourceError, match="introuvable"):
        lc.load_lending_club(tmp_path / "absent.csv")
