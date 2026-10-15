"""Explicabilité du champion : importance par permutation et dépendance partielle.

Deux questions qu'une validation interne pose avant d'accepter un modèle comportemental :

* **Quelles variables comptent ?** L'importance par permutation mesure, **hors temps**, la
  perte de performance quand on mélange une variable : elle reflète ce que le modèle
  utilise réellement en production, contrairement aux importances internes des arbres.
* **Le modèle a-t-il un sens économique ?** La dépendance partielle montre comment la
  probabilité prédite réagit à une variable, toutes choses égales par ailleurs. Pour un
  modèle de remboursement anticipé, elle doit croître avec le taux du prêt (incitation
  à refinancer) et avec l'écart au marché ; un signe inverse serait un signal d'alerte.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import brier_score_loss, make_scorer

_LOG = logging.getLogger(__name__)
_RANDOM_STATE: Final[int] = 42


def permutation_importance_table(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    sample_size: int = 50_000,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Importance par permutation (dégradation du score de Brier), sur un échantillon hors temps.

    Returns:
        Table triée : ``feature, importance_moyenne, importance_ecart_type``. Une importance
        négative ou nulle signifie que la variable ne sert pas au modèle hors temps.
    """
    sample = (
        frame
        if len(frame) <= sample_size
        else frame.sample(n=sample_size, random_state=_RANDOM_STATE)
    )
    scorer = make_scorer(
        brier_score_loss, greater_is_better=False, response_method="predict_proba"
    )
    result = permutation_importance(
        estimator,
        sample[list(features)],
        sample[target].to_numpy(dtype="int8"),
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=_RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(features),
            "importance_moyenne": result.importances_mean,
            "importance_ecart_type": result.importances_std,
        }
    ).sort_values("importance_moyenne", ascending=False)
    _LOG.info("Importance par permutation calculée sur %d lignes", len(sample))
    return table.reset_index(drop=True)


def partial_dependence_table(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    columns: Sequence[str],
    grid_resolution: int = 20,
    sample_size: int = 20_000,
) -> pd.DataFrame:
    """Dépendance partielle de la probabilité prédite pour chaque colonne de ``columns``.

    Returns:
        Table longue ``feature, valeur, probabilite_moyenne`` (grille sur les quantiles 5-95 %).
    """
    sample = (
        frame
        if len(frame) <= sample_size
        else frame.sample(n=sample_size, random_state=_RANDOM_STATE)
    )
    x = sample[list(features)]
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in features:
            continue
        pdp = partial_dependence(
            estimator,
            x,
            [column],
            kind="average",
            grid_resolution=grid_resolution,
            percentiles=(0.05, 0.95),
            response_method="predict_proba",
        )
        values = np.asarray(pdp["grid_values"][0], dtype="float64")
        average = np.asarray(pdp["average"][0], dtype="float64")
        rows.extend(
            {"feature": column, "valeur": float(v), "probabilite_moyenne": float(p)}
            for v, p in zip(values, average, strict=True)
        )
    return pd.DataFrame(rows)


def economic_sign_check(pdp: pd.DataFrame, *, expected: dict[str, str]) -> pd.DataFrame:
    """Vérifie le sens économique de chaque dépendance partielle.

    ``expected`` associe une feature à ``"croissant"`` ou ``"decroissant"`` ; le sens observé
    est le signe de la pente de la régression de la probabilité sur la valeur.
    """
    rows: list[dict[str, Any]] = []
    for feature, direction in expected.items():
        sub = pdp[pdp["feature"] == feature]
        if len(sub) < 2:  # noqa: PLR2004
            rows.append(
                {
                    "feature": feature,
                    "attendu": direction,
                    "observe": "indisponible",
                    "conforme": None,
                }
            )
            continue
        slope = float(np.polyfit(sub["valeur"], sub["probabilite_moyenne"], 1)[0])
        observed = "croissant" if slope > 0 else "decroissant"
        rows.append(
            {
                "feature": feature,
                "attendu": direction,
                "observe": observed,
                "conforme": observed == direction,
            }
        )
    return pd.DataFrame(rows)
