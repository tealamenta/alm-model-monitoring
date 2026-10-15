"""Benchmark de modèles candidats avec validation hors temps.

Un modèle ALM n'est pas choisi sur la seule AUC : en gestion actif-passif, le niveau
prédit (par exemple 15 % de remboursements anticipés sur une cohorte) alimente
directement les projections de flux. Un modèle bien ordonné mais mal calibré fausse le
bilan projeté. Le benchmark compare donc chaque candidat sur quatre familles de mesures :

* **discrimination** : AUC ROC ;
* **précision probabiliste** : score de Brier, log-loss ;
* **calibration** : erreur de calibration attendue (ECE) et biais de niveau
  (moyenne prédite - taux observé) ;
* **stabilité** : écart entre la performance en développement (hors-pli sur la
  référence) et la performance *hors temps* sur la première année de surveillance.

Le protocole est volontairement strict :

1. les candidats sont entraînés **uniquement** sur la fenêtre de référence ;
2. les scores de référence sont **hors-pli** (validation croisée stratifiée) pour ne
   pas surestimer la performance de développement ;
3. la sélection se fait sur la fenêtre **hors temps** (``holdout``), jamais vue, qui
   reproduit les conditions de production.

Règle de sélection (:func:`select_champion`) : le champion est le modèle dont le score
de Brier hors temps est le plus faible parmi ceux dont l'ECE reste sous
``max_calibration_error`` ; à défaut, le mieux calibré. Cette règle est documentée
dans ``docs/methodologie_kpi.md`` et peut être adaptée.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.kpis._stats import expected_calibration_error

_LOG = logging.getLogger(__name__)

RANDOM_STATE: Final[int] = 42
DEFAULT_N_FOLDS: Final[int] = 5
DEFAULT_MAX_ECE: Final[float] = 0.025


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Mesures de performance d'un modèle sur une fenêtre."""

    rows: int
    positive_rate: float
    mean_prediction: float
    auc: float
    average_precision: float
    brier: float
    log_loss: float
    calibration_error: float

    @property
    def level_bias(self) -> float:
        """Biais de niveau : moyenne prédite - taux observé (positif = surestimation)."""
        return self.mean_prediction - self.positive_rate

    def to_dict(self) -> dict[str, float]:
        """Représentation sérialisable."""
        return {
            "rows": float(self.rows),
            "positive_rate": self.positive_rate,
            "mean_prediction": self.mean_prediction,
            "level_bias": self.level_bias,
            "auc": self.auc,
            "average_precision": self.average_precision,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
        }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Résultat complet d'un candidat."""

    name: str
    description: str
    in_time: ModelMetrics
    out_of_time: ModelMetrics
    estimator: BaseEstimator = field(repr=False, compare=False)
    oof_scores: np.ndarray = field(repr=False, compare=False)
    feature_set: str = "base"
    features: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Nom lisible : candidat + jeu de features."""
        return f"{self.name} [{self.feature_set}]"

    @property
    def auc_decay(self) -> float:
        """Perte relative d'AUC entre développement et hors temps."""
        return (self.in_time.auc - self.out_of_time.auc) / self.in_time.auc

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable (sans l'estimateur)."""
        return {
            "name": self.name,
            "feature_set": self.feature_set,
            "features": list(self.features),
            "description": self.description,
            "in_time": self.in_time.to_dict(),
            "out_of_time": self.out_of_time.to_dict(),
            "auc_decay": self.auc_decay,
        }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
    """Calcule toutes les mesures sur un couple (cible, probabilité)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_prob = np.clip(np.asarray(y_prob, dtype="float64"), 1e-9, 1 - 1e-9)
    return ModelMetrics(
        rows=int(y_true.size),
        positive_rate=float(y_true.mean()),
        mean_prediction=float(y_prob.mean()),
        auc=float(roc_auc_score(y_true, y_prob)),
        average_precision=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob)),
        calibration_error=float(expected_calibration_error(y_true, y_prob)),
    )


# ------------------------------------------------------------------ candidats


def build_candidates(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> dict[str, tuple[str, BaseEstimator]]:
    """Trois candidats de complexité croissante, tous robustes aux valeurs manquantes.

    * ``naive_prior`` : prédit le taux de base. Plancher de performance : tout modèle
      doit le battre nettement, sinon il n'apporte rien.
    * ``logistic_regression`` : référence auditable (coefficients lisibles), appréciée
      en validation de modèles bancaire.
    * ``hist_gradient_boosting`` : non linéaire, interactions automatiques, gestion
      native des catégories et des manquants.
    """
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    logistic_prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
                categorical,
            ),
        ]
    )
    hgb_prep = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
                ),
                categorical,
            ),
        ]
    )
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    return {
        "naive_prior": (
            "Taux de base (DummyClassifier prior) : plancher de performance",
            DummyClassifier(strategy="prior"),
        ),
        "logistic_regression": (
            "Imputation médiane + standardisation + one-hot + régression logistique L2",
            Pipeline(
                [
                    ("prep", logistic_prep),
                    ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        "hist_gradient_boosting": (
            "Encodage ordinal des catégories + HistGradientBoosting (early stopping)",
            Pipeline(
                [
                    ("prep", hgb_prep),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=600,
                            max_leaf_nodes=31,
                            min_samples_leaf=200,
                            l2_regularization=1.0,
                            categorical_features=categorical_mask,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=30,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    }


# ------------------------------------------------------------------- benchmark


def run_benchmark(
    reference: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    target: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    candidates: Mapping[str, tuple[str, BaseEstimator]] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    feature_set: str = "base",
) -> list[CandidateResult]:
    """Entraîne chaque candidat sur ``reference`` et l'évalue hors-pli puis sur ``holdout``.

    ``feature_set`` étiquette le jeu de features utilisé (``base``, ``engineered``…) pour
    comparer, à protocole identique, l'apport du feature engineering.
    """
    if n_folds < 2:  # noqa: PLR2004
        raise ConfigurationError("n_folds doit être >= 2", details={"n_folds": n_folds})
    features = [*numeric_features, *categorical_features]
    x_ref, y_ref = reference[features], reference[target].to_numpy(dtype="int8")
    x_hold, y_hold = holdout[features], holdout[target].to_numpy(dtype="int8")
    candidates = candidates or build_candidates(numeric_features, categorical_features)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results: list[CandidateResult] = []
    for name, (description, estimator) in candidates.items():
        _LOG.info("Benchmark : entraînement de %s [%s]", name, feature_set)
        oof = cross_val_predict(clone(estimator), x_ref, y_ref, cv=folds, method="predict_proba")[
            :, 1
        ]
        fitted = clone(estimator).fit(x_ref, y_ref)
        hold = fitted.predict_proba(x_hold)[:, 1]
        result = CandidateResult(
            name=name,
            description=description,
            in_time=compute_metrics(y_ref, oof),
            out_of_time=compute_metrics(y_hold, hold),
            estimator=fitted,
            oof_scores=oof,
            feature_set=feature_set,
            features=tuple(features),
        )
        _LOG.info(
            "%s : AUC hors-pli %.3f, AUC hors temps %.3f, Brier %.4f, ECE %.4f",
            name,
            result.in_time.auc,
            result.out_of_time.auc,
            result.out_of_time.brier,
            result.out_of_time.calibration_error,
        )
        results.append(result)
    return results


def select_champion(
    results: Sequence[CandidateResult], *, max_calibration_error: float = DEFAULT_MAX_ECE
) -> tuple[CandidateResult, str]:
    """Applique la règle de sélection et renvoie ``(champion, justification)``.

    Le modèle naïf n'est jamais champion : il sert de plancher.
    """
    eligible = [r for r in results if r.name != "naive_prior"]
    if not eligible:
        raise ConfigurationError("Aucun candidat non trivial dans le benchmark")
    calibrated = [r for r in eligible if r.out_of_time.calibration_error <= max_calibration_error]
    if calibrated:
        champion = min(calibrated, key=lambda r: (r.out_of_time.brier, -r.out_of_time.auc))
        reason = (
            f"meilleur score de Brier hors temps ({champion.out_of_time.brier:.4f}) parmi les "
            f"{len(calibrated)} candidat(s) dont l'ECE reste sous {max_calibration_error:.3f}"
        )
    else:
        champion = min(eligible, key=lambda r: r.out_of_time.calibration_error)
        reason = (
            f"aucun candidat sous le seuil d'ECE {max_calibration_error:.3f} ; retenu pour la "
            f"meilleure calibration hors temps ({champion.out_of_time.calibration_error:.4f})"
        )
    return champion, reason


def benchmark_table(results: Sequence[CandidateResult]) -> pd.DataFrame:
    """Tableau comparatif (une ligne par candidat) pour la documentation et les notebooks."""
    rows = [
        {
            "modele": r.label,
            "auc_dev": r.in_time.auc,
            "auc_hors_temps": r.out_of_time.auc,
            "pr_auc_hors_temps": r.out_of_time.average_precision,
            "perte_auc": r.auc_decay,
            "brier_hors_temps": r.out_of_time.brier,
            "log_loss_hors_temps": r.out_of_time.log_loss,
            "ece_hors_temps": r.out_of_time.calibration_error,
            "biais_niveau_hors_temps": r.out_of_time.level_bias,
        }
        for r in results
    ]
    return pd.DataFrame(rows).set_index("modele").round(4)
