"""Tests de l'audit de qualité des données."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alm_monitoring import data_audit as da
from alm_monitoring.types import Severity

pytestmark = pytest.mark.unit


def _prepared(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "issue_d": pd.Timestamp("2015-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
            "loan_amnt": rng.lognormal(9, 0.4, n),
            "int_rate": rng.normal(13, 4, n).clip(5, 30),
            "dti": rng.normal(18, 8, n).clip(0, 60),
            "grade": rng.choice(list("ABCDE"), n),
            "constant": 1.0,
            "y": (rng.random(n) < 0.15).astype(int),
        }
    )
    frame["installment"] = frame["loan_amnt"] * 0.03  # colinéaire avec le montant
    return frame


def test_duplicates_detects_exact_and_semantic() -> None:
    frame = _prepared(200)
    clean = da.check_duplicates(frame, id_column="id", key_columns=["loan_amnt", "int_rate"])
    assert clean.severity is Severity.OK
    doubled = pd.concat([frame, frame.head(5)], ignore_index=True)
    dirty = da.check_duplicates(doubled, id_column="id", key_columns=["loan_amnt", "int_rate"])
    assert dirty.severity is Severity.CRITICAL
    assert dirty.details["exact"] == 5
    assert dirty.details["repeated_ids"] == 5


def test_business_rules_severity_follows_share() -> None:
    frame = _prepared(1000)
    rules: list[da.BusinessRule] = [
        ("dti_negatif", lambda df: (df["dti"] < 0).fillna(value=False))
    ]
    finding, table = da.check_business_rules(frame, rules)
    assert finding.severity is Severity.OK
    frame.loc[:19, "dti"] = -1.0  # 2 % des lignes
    finding, table = da.check_business_rules(frame, rules)
    assert finding.severity is Severity.CRITICAL
    assert table.loc[0, "violations"] == 20


def test_lending_club_rules_cover_raw_file() -> None:
    raw = pd.DataFrame(
        {
            "dti": [10.0, 150.0],
            "annual_inc": [50000.0, 0.0],
            "fico_range_low": [700.0, 700.0],
            "fico_range_high": [704.0, 704.0],
            "int_rate": [10.0, 10.0],
            "loan_amnt": [10000.0, 10000.0],
            "revol_util": [50.0, 120.0],
            "installment": [322.67, 1000.0],
            "term": [" 36 months", " 36 months"],
            "issue_d": ["Jan-2015", "Jan-2015"],
            "last_pymnt_d": ["Jun-2015", "Dec-2014"],
        }
    )
    finding, table = da.check_business_rules(raw, da.lending_club_rules())
    assert finding.severity is Severity.CRITICAL  # 50 % des lignes touchées
    violated = set(table.loc[table["violations"] > 0, "regle"])
    assert {
        "dti_hors_bornes [0, 100]",
        "revenu_annuel_nul_ou_negatif",
        "utilisation_revolving_superieure_100",
    } <= violated
    assert "dernier_paiement_avant_emission" in violated
    assert "mensualite_incoherente_avec_annuite (>10 %)" in violated


def test_missing_detects_informative_mechanism() -> None:
    frame = _prepared(4000, seed=1)
    # manquant lié à la cible : on retire la valeur surtout chez les positifs
    mask = (frame["y"] == 1) & (np.random.default_rng(2).random(len(frame)) < 0.8)
    frame.loc[mask, "dti"] = np.nan
    finding, table = da.check_missing(frame, features=["dti", "loan_amnt"], target="y")
    assert finding.severity is Severity.WARNING
    assert finding.details["informative"] == ["dti"]
    row = table.set_index("colonne").loc["dti"]
    assert row["ratio_cible_manquant_vs_present"] > 1.25


def test_outliers_distributions_cardinality_vif() -> None:
    frame = _prepared(3000, seed=3)
    frame.loc[:99, "loan_amnt"] = 5e7  # 3 % d'aberrants massifs
    outliers, _ = da.check_outliers(frame, numeric=["loan_amnt", "int_rate"])
    assert outliers.severity is Severity.WARNING
    assert outliers.details["flagged"] == ["loan_amnt"]

    dist, _ = da.check_distributions(frame, numeric=["loan_amnt", "int_rate"])
    assert dist.severity is Severity.OK
    assert "loan_amnt" in dist.details["skewed"]

    card, _ = da.check_cardinality(frame, features=["constant", "grade", "id"])
    assert "constant" in card.details["near_constant"]
    assert "id" in card.details["high_cardinality"]

    vif, _ = da.check_multicollinearity(
        _prepared(3000, seed=6), numeric=["loan_amnt", "installment", "int_rate"]
    )
    assert vif.severity is Severity.WARNING
    assert set(vif.details["severe"]) == {"loan_amnt", "installment"}


def test_temporal_consistency_flags_gaps_and_small_months() -> None:
    frame = _prepared(3000, seed=4)
    ok, table = da.check_temporal_consistency(frame, timestamp="issue_d", min_rows=50)
    assert ok.severity is Severity.OK
    assert len(table) == 12
    gap = frame[frame["issue_d"].dt.month != 6]
    bad, table = da.check_temporal_consistency(gap, timestamp="issue_d", min_rows=50)
    assert bad.severity is Severity.CRITICAL
    assert bad.details["gaps"] == ["2015-06"]


def test_target_quality_and_full_audit(tmp_path) -> None:
    frame = _prepared(3000, seed=5)
    finding, table = da.check_target_quality(frame, target="y", timestamp="issue_d")
    assert finding.severity is Severity.OK
    assert list(table.columns) == ["annee", "lignes", "taux_positif"]
    rare = frame.assign(y=0)
    rare.loc[:9, "y"] = 1
    finding, _ = da.check_target_quality(rare, target="y", timestamp="issue_d")
    assert finding.severity is Severity.CRITICAL

    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    report = da.audit(
        frame,
        target="y",
        timestamp="issue_d",
        numeric_features=["loan_amnt", "int_rate", "dti", "installment"],
        categorical_features=["grade"],
        raw=frame,
        rules=[("dti_negatif", lambda df: (df["dti"] < 0).fillna(value=False))],
        duplicate_keys=["loan_amnt", "int_rate"],
        source_path=source,
    )
    assert report.exploitable
    assert report.findings[0].check == "fingerprint"
    assert len(report.findings[0].details["sha256"]) == 64
    assert set(report.tables) >= {
        "business_rules",
        "missing_values",
        "outliers",
        "multicollinearity",
    }
    assert "EXPLOITABLE" in report.summary()
    assert len(report.to_dataframe()) == len(report.findings)
    markdown = da.report_to_markdown(report)
    assert markdown.startswith("# Audit de qualité des données")
    assert "## multicollinearity" in markdown
    assert len(da.findings_to_records(report)) == len(report.findings)
