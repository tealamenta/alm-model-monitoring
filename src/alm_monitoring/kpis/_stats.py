"""Routines numériques pures partagées par les KPI.

Tout ce qui se trouve ici opère sur des tableaux NumPy / des Series pandas et ignore
la configuration du monitoring, ce qui rend les fonctions faciles à tester
unitairement et à réutiliser.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance

from alm_monitoring.exceptions import InsufficientDataError, KPIComputationError
from alm_monitoring.types import FloatArray

_EPS = 1e-6


def clean_numeric(series: pd.Series[Any]) -> FloatArray:
    """Convertit en float64 et supprime les NaN / valeurs infinies."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    # ``np.asarray(..., dtype=np.float64)`` fige le type quelle que soit la version de numpy.
    return np.asarray(values[np.isfinite(values)], dtype=np.float64)


def reference_bin_edges(reference: FloatArray, bins: int) -> FloatArray:
    """Bornes de classes par quantiles, calculées sur la distribution de référence.

    Utiliser les quantiles de la référence (plutôt que des classes de largeur égale)
    rend le PSI robuste aux variables ALM à queue lourde telles que les encours ou les
    durées. Les bornes sont dédoublonnées et étendues à +/- l'infini pour que chaque
    valeur courante tombe dans une classe.

    Les variables à faible cardinalité (au plus ``bins`` valeurs de référence
    distinctes, p. ex. des compteurs comme ``previous``) reçoivent une classe par
    valeur distincte : des bornes par quantiles s'effondreraient sinon sur la valeur
    dominante et masqueraient tout décalage.
    """
    if bins < 2:  # noqa: PLR2004
        raise KPIComputationError("bins doit être >= 2", details={"bins": bins})
    if reference.size == 0:
        raise InsufficientDataError("La distribution de référence est vide")
    distinct = np.unique(reference)
    if distinct.size == 1:
        # Référence constante : une seule classe absorberait toute valeur courante et le
        # PSI vaudrait 0 même si la variable a complètement changé (cas limite détecté
        # par test_drift_edge_cases). On isole la valeur de référence dans sa propre
        # classe, encadrée par deux classes « au-dessous » et « au-dessus ».
        value = float(distinct[0])
        tol = max(abs(value), 1.0) * 1e-9
        inner = np.array([value - tol, value + tol])
    elif distinct.size <= bins:
        inner = (distinct[:-1] + distinct[1:]) / 2.0
    else:
        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        inner = np.unique(np.quantile(reference, quantiles))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def binned_frequencies(values: FloatArray, edges: FloatArray) -> FloatArray:
    """Fréquences relatives de ``values`` dans les classes définies par ``edges``."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        raise InsufficientDataError("Aucune valeur finie à répartir en classes")
    return np.asarray(counts.astype("float64") / total, dtype=np.float64)


def categorical_frequencies(
    reference: pd.Series[Any], current: pd.Series[Any]
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Fréquences relatives alignées sur l'union des catégories (NaN conservé comme niveau)."""
    ref = reference.astype("string").fillna("<NA>")
    cur = current.astype("string").fillna("<NA>")
    if ref.empty or cur.empty:
        raise InsufficientDataError("Série catégorielle vide")
    levels = sorted(set(ref.unique()) | set(cur.unique()))
    ref_freq = ref.value_counts(normalize=True).reindex(levels, fill_value=0.0)
    cur_freq = cur.value_counts(normalize=True).reindex(levels, fill_value=0.0)
    return (
        np.asarray(ref_freq.to_numpy(), dtype=np.float64),
        np.asarray(cur_freq.to_numpy(), dtype=np.float64),
        [str(level) for level in levels],
    )


def psi(expected: FloatArray, actual: FloatArray, *, eps: float = _EPS) -> float:
    """Population Stability Index entre deux vecteurs de fréquences alignés.

    Le lissage ``eps`` évite la division par zéro pour les classes vides. Règle
    empirique du secteur : < 0,10 stable, 0,10-0,25 décalage modéré, > 0,25 décalage
    significatif.
    """
    if expected.shape != actual.shape:
        raise KPIComputationError("Les vecteurs de fréquences doivent être alignés")
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def jensen_shannon(expected: FloatArray, actual: FloatArray) -> float:
    """*Distance* de Jensen-Shannon (base 2, bornée dans [0, 1])."""
    if expected.shape != actual.shape:
        raise KPIComputationError("Les vecteurs de fréquences doivent être alignés")
    with np.errstate(invalid="ignore", divide="ignore"):  # des vecteurs nuls donnent NaN
        value = float(distance.jensenshannon(expected, actual, base=2))
    return 0.0 if np.isnan(value) else value


def ks_statistic(reference: FloatArray, current: FloatArray) -> tuple[float, float]:
    """Statistique de Kolmogorov-Smirnov à deux échantillons et p-value associée."""
    if reference.size == 0 or current.size == 0:
        raise InsufficientDataError("Le test KS requiert des échantillons non vides")
    res = stats.ks_2samp(reference, current, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def cramers_v(reference: pd.Series[Any], current: pd.Series[Any]) -> tuple[float, float]:
    """V de Cramér corrigé du biais et p-value du khi-deux entre deux échantillons catégoriels."""
    ref = reference.astype("string").fillna("<NA>")
    cur = current.astype("string").fillna("<NA>")
    table = pd.crosstab(
        pd.concat([ref, cur], ignore_index=True),
        np.repeat(["reference", "current"], [len(ref), len(cur)]),
    )
    if table.shape[0] < 2:  # noqa: PLR2004 - une seule catégorie : aucune association possible
        return 0.0, 1.0
    chi2, p_value, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    # Correction du biais de Bergsma (2013)
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0
    return v, float(p_value)


def expected_calibration_error(y_true: FloatArray, y_prob: FloatArray, *, bins: int = 10) -> float:
    """Expected Calibration Error avec des classes de probabilité de largeur égale."""
    if y_true.shape != y_prob.shape:
        raise KPIComputationError("y_true et y_prob doivent avoir la même forme")
    if y_true.size == 0:
        raise InsufficientDataError("L'ECE requiert des échantillons non vides")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    n = y_true.size
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)
