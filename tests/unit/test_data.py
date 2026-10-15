"""Tests de la couche données : générateur synthétique, préparation Bank Marketing, sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alm_monitoring.data import DataSource, KaggleSource, LocalCsvSource, synthetic
from alm_monitoring.data import bank_marketing as bm
from alm_monitoring.data.synthetic import DriftScenario
from alm_monitoring.exceptions import ConfigurationError, DataSourceError, SchemaValidationError

pytestmark = pytest.mark.unit


class TestSynthetic:
    def test_schema_and_determinism(self) -> None:
        a = synthetic.make_dataset(500, seed=7)
        b = synthetic.make_dataset(500, seed=7)
        pd.testing.assert_frame_equal(a, b)
        expected = {
            synthetic.TIMESTAMP,
            *synthetic.NUMERIC_FEATURES,
            *synthetic.CATEGORICAL_FEATURES,
            synthetic.TARGET,
            synthetic.PREDICTION,
        }
        assert set(a.columns) == expected
        assert a["y"].isin([0, 1]).all()
        assert a["score"].between(0, 1).all()
        assert a["date"].dt.day.eq(1).all()

    def test_periods_spread_dates(self) -> None:
        frame = synthetic.make_dataset(2_000, seed=1, periods=6, start="2023-01-01")
        assert frame["date"].nunique() == 6
        assert frame["date"].min() == pd.Timestamp("2023-01-01")

    def test_scenario_effects(self) -> None:
        base = synthetic.make_dataset(5_000, seed=0)
        drifted = synthetic.make_dataset(
            5_000,
            seed=0,
            scenario=DriftScenario(
                rate_shock=2.0, missing_rate=0.2, unseen_category_rate=0.1, numeric_shift=0.5
            ),
        )
        assert drifted["rate_spread"].mean() > base["rate_spread"].mean() + 1.5
        assert drifted["balance"].isna().mean() == pytest.approx(0.2, abs=0.02)
        assert (drifted["product"] == "structured_note").mean() == pytest.approx(0.1, abs=0.02)
        assert drifted["age"].mean() > base["age"].mean()

    def test_score_is_calibrated_without_drift(self) -> None:
        frame = synthetic.make_dataset(20_000, seed=3)
        assert frame["score"].mean() == pytest.approx(frame["y"].mean(), abs=0.01)

    @pytest.mark.parametrize(
        "kwargs",
        [{"missing_rate": 1.5}, {"label_noise": -0.1}, {"score_miscalibration": 0}],
    )
    def test_invalid_scenarios(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ConfigurationError):
            DriftScenario(**kwargs)

    def test_invalid_rows(self) -> None:
        with pytest.raises(ConfigurationError):
            synthetic.make_dataset(0)

    def test_model_spec_matches_dataset(self) -> None:
        spec = synthetic.synthetic_model_spec()
        frame = synthetic.make_dataset(10)
        features = spec["features"]
        assert isinstance(features, dict)
        for col in [*features["numeric"], *features["categorical"]]:
            assert col in frame.columns


def _raw_bank(n_per_month: int = 5) -> pd.DataFrame:
    months = ["may", "jun", "jul", "aug", "oct", "nov", "dec", "mar", "apr", "may"]
    rows = []
    for m in months:
        for i in range(n_per_month):
            rows.append(
                {
                    "age": 30 + i,
                    "job": "admin.",
                    "marital": "married",
                    "education": "university.degree",
                    "default": "no",
                    "housing": "yes",
                    "loan": "no",
                    "contact": "cellular",
                    "month": m,
                    "day_of_week": "mon",
                    "duration": 100,
                    "campaign": 1,
                    "pdays": 999,
                    "previous": 0,
                    "poutcome": "nonexistent",
                    "emp.var.rate": 1.1,
                    "cons.price.idx": 93.9,
                    "cons.conf.idx": -36.4,
                    "euribor3m": 4.857,
                    "nr.employed": 5191.0,
                    "y": "yes" if i % 2 else "no",
                }
            )
    return pd.DataFrame(rows)


class TestBankMarketing:
    def test_prepare(self) -> None:
        prepared = bm.prepare_bank_marketing(_raw_bank())
        assert "duration" not in prepared.columns
        assert "emp_var_rate" in prepared.columns
        assert prepared[bm.TARGET].dtype == "int8"
        assert prepared[bm.TARGET].mean() == pytest.approx(0.4)
        assert prepared[bm.TIMESTAMP].min() == pd.Timestamp("2008-05-01")
        # Le passage dec -> mar incrémente l'année.
        assert prepared[bm.TIMESTAMP].max() == pd.Timestamp("2009-05-01")
        assert all(prepared[c].dtype == "string" for c in bm.CATEGORICAL_FEATURES)
        assert all(np.issubdtype(prepared[c].dtype, np.number) for c in bm.NUMERIC_FEATURES)

    def test_missing_columns(self) -> None:
        with pytest.raises(SchemaValidationError) as exc:
            bm.prepare_bank_marketing(_raw_bank().drop(columns=["euribor3m", "job"]))
        assert exc.value.details["missing"] == ["euribor3m", "job"]

    def test_unknown_month(self) -> None:
        raw = _raw_bank()
        raw.loc[0, "month"] = "xyz"
        with pytest.raises(SchemaValidationError, match="month"):
            bm.prepare_bank_marketing(raw)

    def test_split(self) -> None:
        prepared = bm.prepare_bank_marketing(_raw_bank())
        ref, cur = bm.split_reference_current(prepared, cutoff="2009-01-01")
        assert (ref[bm.TIMESTAMP] < "2009-01-01").all()
        assert (cur[bm.TIMESTAMP] >= "2009-01-01").all()
        assert len(ref) + len(cur) == len(prepared)

    def test_split_default_cutoff(self) -> None:
        prepared = bm.prepare_bank_marketing(_raw_bank())
        with pytest.raises(SchemaValidationError, match="fenêtre vide"):
            # La fixture s'arrête en 2009-05, avant le cutoff par défaut du 2009-07-01.
            bm.split_reference_current(prepared)

    def test_split_errors(self) -> None:
        prepared = bm.prepare_bank_marketing(_raw_bank())
        with pytest.raises(SchemaValidationError, match="fenêtre vide"):
            bm.split_reference_current(prepared, cutoff="2030-01-01")
        with pytest.raises(SchemaValidationError, match="pas de colonne de date"):
            bm.split_reference_current(prepared.drop(columns=[bm.TIMESTAMP]))


class TestSources:
    def test_local_csv(self, tmp_path: Path) -> None:
        file = tmp_path / "x.csv"
        file.write_text("a;b\n1;2\n", encoding="utf-8")
        source = LocalCsvSource("x.csv", base_dir=tmp_path, sep=";")
        assert isinstance(source, DataSource)
        assert source.path == file.resolve()
        assert source.load().to_dict("records") == [{"a": 1, "b": 2}]

    def test_local_csv_errors(self, tmp_path: Path) -> None:
        with pytest.raises(DataSourceError, match="Impossible de lire"):
            LocalCsvSource("missing.csv", base_dir=tmp_path).load()
        (tmp_path / "empty.csv").write_text("", encoding="utf-8")
        with pytest.raises(DataSourceError):
            LocalCsvSource("empty.csv", base_dir=tmp_path).load()

    @pytest.mark.parametrize("dataset", ["noslash", "a/b/c", "owner/../name", "a b/c"])
    def test_kaggle_dataset_name_validated(self, dataset: str) -> None:
        with pytest.raises(DataSourceError):
            KaggleSource(dataset, "file.csv")

    def test_kaggle_load_with_fake_hub(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        (tmp_path / "bank.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        fake = types.ModuleType("kagglehub")
        fake.dataset_download = lambda _name: str(tmp_path)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kagglehub", fake)
        frame = KaggleSource("owner/name", "bank.csv").load()
        assert frame.shape == (1, 2)
        with pytest.raises(DataSourceError, match="Échec du téléchargement Kaggle"):
            KaggleSource("owner/name", "../escape.csv").load()
