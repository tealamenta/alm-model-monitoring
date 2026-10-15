"""Tests des routines numériques pures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alm_monitoring.exceptions import InsufficientDataError, KPIComputationError
from alm_monitoring.kpis import _stats

pytestmark = pytest.mark.unit


class TestCleanNumeric:
    def test_drops_nan_inf_and_strings(self) -> None:
        series = pd.Series([1.0, np.nan, np.inf, "x", 3])
        out = _stats.clean_numeric(series)
        assert out.tolist() == [1.0, 3.0]
        assert out.dtype == np.float64


class TestBinning:
    def test_edges_are_monotonic_and_infinite_at_ends(self) -> None:
        ref = np.linspace(0, 1, 1000)
        edges = _stats.reference_bin_edges(ref, 10)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        assert np.all(np.diff(edges) > 0)
        assert edges.size == 11

    def test_constant_reference_isolates_its_value(self) -> None:
        """Une référence constante ne doit PAS se réduire à une classe unique : la valeur
        est isolée pour qu'un courant différent soit détecté (régression corrigée)."""
        edges = _stats.reference_bin_edges(np.ones(50), 10)
        assert edges.size == 4  # -inf, 1-tol, 1+tol, +inf
        ref_freq = _stats.binned_frequencies(np.ones(50), edges)
        cur_freq = _stats.binned_frequencies(np.full(50, 2.0), edges)
        assert ref_freq.tolist() == [0.0, 1.0, 0.0]
        assert cur_freq.tolist() == [0.0, 0.0, 1.0]

    def test_sparse_counter_gets_one_bin_per_distinct_value(self) -> None:
        # 97 % de zéros : des bornes par quantiles s'effondreraient toutes sur 0 et un
        # décalage de la masse non nulle serait invisible pour le PSI (observé sur
        # `previous` dans les données Bank Marketing).
        ref = np.array([0.0] * 97 + [1.0, 2.0, 3.0])
        edges = _stats.reference_bin_edges(ref, 10)
        assert edges.tolist() == [-np.inf, 0.5, 1.5, 2.5, np.inf]
        cur = np.array([0.0] * 80 + [1.0] * 10 + [2.0] * 5 + [3.0] * 5)
        value = _stats.psi(
            _stats.binned_frequencies(ref, edges), _stats.binned_frequencies(cur, edges)
        )
        assert value > 0.25

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(KPIComputationError):
            _stats.reference_bin_edges(np.ones(5), 1)
        with pytest.raises(InsufficientDataError):
            _stats.reference_bin_edges(np.array([]), 5)

    def test_frequencies_sum_to_one(self) -> None:
        ref = np.random.default_rng(0).normal(size=500)
        edges = _stats.reference_bin_edges(ref, 10)
        freq = _stats.binned_frequencies(ref, edges)
        assert freq.sum() == pytest.approx(1.0)
        assert np.allclose(freq, 0.1, atol=0.01)

    def test_frequencies_require_values(self) -> None:
        with pytest.raises(InsufficientDataError):
            _stats.binned_frequencies(np.array([]), np.array([-np.inf, 0.0, np.inf]))


class TestCategoricalFrequencies:
    def test_union_of_levels_with_missing_as_level(self) -> None:
        ref = pd.Series(["a", "b", "a", None])
        cur = pd.Series(["a", "c"])
        ref_f, cur_f, levels = _stats.categorical_frequencies(ref, cur)
        assert levels == ["<NA>", "a", "b", "c"]
        assert ref_f.sum() == pytest.approx(1.0)
        assert cur_f.sum() == pytest.approx(1.0)
        assert cur_f[levels.index("b")] == 0.0

    def test_empty_series_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            _stats.categorical_frequencies(pd.Series([], dtype="string"), pd.Series(["a"]))


class TestPSI:
    def test_identical_distributions_give_zero(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        assert _stats.psi(p, p) == pytest.approx(0.0)

    def test_shift_increases_psi(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        q_small = np.array([0.45, 0.3, 0.25])
        q_big = np.array([0.1, 0.2, 0.7])
        assert 0 < _stats.psi(p, q_small) < _stats.psi(p, q_big)

    def test_empty_bin_is_smoothed_not_infinite(self) -> None:
        value = _stats.psi(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert np.isfinite(value)

    def test_misaligned_vectors_rejected(self) -> None:
        with pytest.raises(KPIComputationError):
            _stats.psi(np.array([1.0]), np.array([0.5, 0.5]))

    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=20),
        st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=20),
    )
    def test_psi_is_non_negative_and_symmetric(self, a: list[float], b: list[float]) -> None:
        n = min(len(a), len(b))
        p = np.array(a[:n]) / sum(a[:n])
        q = np.array(b[:n]) / sum(b[:n])
        assert _stats.psi(p, q) >= -1e-12
        assert _stats.psi(p, q) == pytest.approx(_stats.psi(q, p))


class TestJensenShannon:
    def test_bounds(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert _stats.jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)
        assert _stats.jensen_shannon(p, q) == pytest.approx(1.0)

    def test_nan_becomes_zero(self) -> None:
        zero = np.zeros(3)
        assert _stats.jensen_shannon(zero, zero) == 0.0

    def test_misaligned_vectors_rejected(self) -> None:
        with pytest.raises(KPIComputationError):
            _stats.jensen_shannon(np.array([1.0]), np.array([0.5, 0.5]))


class TestKS:
    def test_same_sample_statistic_zero(self) -> None:
        x = np.random.default_rng(0).normal(size=300)
        stat, p = _stats.ks_statistic(x, x)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_shifted_sample_detected(self) -> None:
        rng = np.random.default_rng(0)
        stat, p = _stats.ks_statistic(rng.normal(size=500), rng.normal(2.0, size=500))
        assert stat > 0.5
        assert p < 1e-6

    def test_empty_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            _stats.ks_statistic(np.array([]), np.ones(3))


class TestCramersV:
    def test_identical_distribution_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.choice(["a", "b", "c"], 2000))
        cur = pd.Series(rng.choice(["a", "b", "c"], 2000))
        v, p = _stats.cramers_v(ref, cur)
        assert v < 0.05
        assert 0.0 <= p <= 1.0

    def test_disjoint_levels_give_strong_association(self) -> None:
        ref = pd.Series(["a"] * 200 + ["b"] * 200)
        cur = pd.Series(["c"] * 400)
        v, p = _stats.cramers_v(ref, cur)
        assert v > 0.9
        assert p < 1e-6

    def test_single_level_returns_zero(self) -> None:
        assert _stats.cramers_v(pd.Series(["a"] * 10), pd.Series(["a"] * 10)) == (0.0, 1.0)


class TestECE:
    def test_perfectly_calibrated_is_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        prob = rng.uniform(size=50_000)
        y = (rng.uniform(size=prob.size) < prob).astype("float64")
        assert _stats.expected_calibration_error(y, prob) < 0.02

    def test_overconfident_is_large(self) -> None:
        y = np.zeros(100)
        prob = np.full(100, 0.9)
        assert _stats.expected_calibration_error(y, prob) == pytest.approx(0.9)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(KPIComputationError):
            _stats.expected_calibration_error(np.zeros(3), np.zeros(2))
        with pytest.raises(InsufficientDataError):
            _stats.expected_calibration_error(np.array([]), np.array([]))
