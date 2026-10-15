"""Propriétés mathématiques des métriques de calibration (ECE, biais de niveau).

Ces tests ne vérifient pas une valeur attendue sur un exemple, mais des **invariants**
qui doivent tenir quelles que soient les données : c'est la réponse à la question
« la formule est-elle juste ? » posée par toute revue de modèle.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from alm_monitoring.kpis import _stats
from alm_monitoring.kpis.performance import LevelBias

pytestmark = pytest.mark.unit

_N = st.integers(min_value=50, max_value=2_000)


def _calibrated_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Scores tirés uniformément, cibles tirées selon ces scores : calibration parfaite."""
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(size=n) < prob).astype("float64")
    return y, prob


class TestExpectedCalibrationError:
    @given(n=st.integers(min_value=20_000, max_value=40_000), seed=st.integers(0, 10_000))
    @settings(max_examples=15, deadline=None)
    def test_perfectly_calibrated_scores_give_near_zero_ece(self, n: int, seed: int) -> None:
        y, prob = _calibrated_sample(n, seed)
        # Bruit d'échantillonnage : ~ 1/sqrt(n/10) par classe, pondéré → bien < 0,02.
        assert _stats.expected_calibration_error(y, prob) < 0.02

    @given(
        prob=hnp.arrays(np.float64, st.integers(10, 500), elements=st.floats(0.0, 1.0)),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_bounded_in_unit_interval(self, prob: np.ndarray, data: st.DataObject) -> None:  # type: ignore[type-arg]
        y = np.asarray(
            data.draw(
                st.lists(st.sampled_from([0.0, 1.0]), min_size=len(prob), max_size=len(prob))
            )
        )
        ece = _stats.expected_calibration_error(y, prob)
        assert 0.0 <= ece <= 1.0

    @given(
        prob=hnp.arrays(np.float64, st.integers(10, 500), elements=st.floats(0.0, 1.0)),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_invariant_under_permutation(self, prob: np.ndarray, data: st.DataObject) -> None:  # type: ignore[type-arg]
        y = np.asarray(
            data.draw(
                st.lists(st.sampled_from([0.0, 1.0]), min_size=len(prob), max_size=len(prob))
            )
        )
        perm = np.random.default_rng(0).permutation(len(prob))
        assert _stats.expected_calibration_error(y, prob) == pytest.approx(
            _stats.expected_calibration_error(y[perm], prob[perm])
        )

    @given(shift=st.floats(min_value=0.0, max_value=0.3), n=_N)
    @settings(max_examples=50, deadline=None)
    def test_constant_level_shift_is_recovered(self, shift: float, n: int) -> None:
        """Si toutes les probabilités sont décalées de ``shift`` par rapport à la fréquence
        observée dans chaque classe, l'ECE vaut exactement ``shift`` (à l'arrondi près)."""
        rng = np.random.default_rng(1)
        true_prob = rng.uniform(0.0, 0.6, size=n)
        # Cibles « idéales » : on force la fréquence observée de chaque classe égale à la
        # moyenne de true_prob dans la classe en tirant y de façon déterministe.
        edges = np.linspace(0.0, 1.0, 11)
        pred = np.clip(true_prob + shift, 0.0, 1.0)
        idx = np.clip(np.digitize(pred, edges[1:-1], right=True), 0, 9)
        y = np.zeros(n)
        for b in np.unique(idx):
            mask = idx == b
            k = round(true_prob[mask].mean() * mask.sum())
            y[np.flatnonzero(mask)[:k]] = 1.0
        ece = _stats.expected_calibration_error(y, pred)
        expected = float(np.mean(np.abs(pred - true_prob)))  # = shift sauf aux bornes
        assert ece == pytest.approx(expected, abs=0.5 / n * 10 + 1e-9)

    def test_all_identical_scores_single_bin(self) -> None:
        y = np.array([1.0, 0.0, 0.0, 0.0])
        prob = np.full(4, 0.25)
        assert _stats.expected_calibration_error(y, prob) == pytest.approx(0.0)
        assert _stats.expected_calibration_error(y, np.full(4, 0.75)) == pytest.approx(0.5)

    def test_single_observed_class_is_handled(self) -> None:
        """Une seule classe observée : l'ECE reste défini (c'est l'AUC qui ne l'est pas)."""
        y = np.zeros(100)
        prob = np.full(100, 0.1)
        assert _stats.expected_calibration_error(y, prob) == pytest.approx(0.1)


class TestLevelBias:
    @given(
        prob=hnp.arrays(np.float64, st.integers(10, 500), elements=st.floats(0.0, 1.0)),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_equals_abs_difference_of_means_and_is_permutation_invariant(
        self,
        prob: np.ndarray,  # type: ignore[type-arg]
        data: st.DataObject,
    ) -> None:
        y = np.asarray(
            data.draw(
                st.lists(st.sampled_from([0.0, 1.0]), min_size=len(prob), max_size=len(prob))
            )
        )
        value = LevelBias.metric(y, prob)
        assert value == pytest.approx(abs(prob.mean() - y.mean()))
        perm = np.random.default_rng(0).permutation(len(prob))
        assert LevelBias.metric(y[perm], prob[perm]) == pytest.approx(value)

    @given(n=st.integers(min_value=20_000, max_value=40_000), seed=st.integers(0, 10_000))
    @settings(max_examples=15, deadline=None)
    def test_near_zero_when_calibrated(self, n: int, seed: int) -> None:
        y, prob = _calibrated_sample(n, seed)
        assert LevelBias.metric(y, prob) < 0.01

    def test_sign_is_kept_in_details_and_value_is_absolute(self) -> None:
        """Sous-estimation (prédit 0,14, observé 0,17) : valeur 0,03, signe négatif."""
        y = np.array([1.0] * 17 + [0.0] * 83)
        prob = np.full(100, 0.14)
        assert LevelBias.metric(y, prob) == pytest.approx(0.03)
        assert prob.mean() - y.mean() == pytest.approx(-0.03)
