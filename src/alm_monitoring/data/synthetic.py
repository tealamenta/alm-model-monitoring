"""Générateur de données synthétiques à drift contrôlable.

Utilisé par la suite de tests et par la démo lorsqu'aucun dataset réel n'est
disponible. Le schéma reproduit un modèle comportemental ALM simplifié : quelques
variables client et macro, une cible binaire et un score de probabilité calibré.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alm_monitoring.exceptions import ConfigurationError

NUMERIC_FEATURES = ("age", "balance", "rate_spread", "tenure_months")
CATEGORICAL_FEATURES = ("segment", "product", "channel")
TARGET = "y"
PREDICTION = "score"
TIMESTAMP = "date"

_SEGMENTS = ("retail", "affluent", "private")
_PRODUCTS = ("savings", "term_deposit", "current")
_CHANNELS = ("branch", "online", "phone")


@dataclass(frozen=True, slots=True)
class DriftScenario:
    """Perturbations appliquées à un dataset généré.

    Attributes:
        numeric_shift: Décalage (en écarts-types) ajouté à chaque variable numérique.
        rate_shock: Choc additif sur ``rate_spread`` (changement de régime ALM).
        missing_rate: Fraction de ``balance`` mise à NaN.
        unseen_category_rate: Fraction de ``product`` remplacée par un nouveau niveau.
        label_noise: Fraction de labels inversés (dégrade la performance).
        score_miscalibration: Facteur multiplicatif appliqué au logit du score.
        extra_categories: Noms des niveaux inédits injectés dans ``product``.
    """

    numeric_shift: float = 0.0
    rate_shock: float = 0.0
    missing_rate: float = 0.0
    unseen_category_rate: float = 0.0
    label_noise: float = 0.0
    score_miscalibration: float = 1.0
    extra_categories: tuple[str, ...] = field(default=("structured_note",))

    def __post_init__(self) -> None:
        for name in ("missing_rate", "unseen_category_rate", "label_noise"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} doit être dans [0, 1]", details={name: value})
        if self.score_miscalibration <= 0:
            raise ConfigurationError("score_miscalibration doit être > 0")


NO_DRIFT = DriftScenario()


def make_dataset(
    n_rows: int,
    *,
    seed: int = 0,
    scenario: DriftScenario = NO_DRIFT,
    start: str = "2024-01-01",
    periods: int = 1,
) -> pd.DataFrame:
    """Génère ``n_rows`` observations réparties uniformément sur ``periods`` mois.

    La cible est tirée d'un modèle logistique des variables, de sorte que la colonne
    de score est bien calibrée lorsque ``scenario.score_miscalibration == 1``.
    """
    if n_rows <= 0:
        raise ConfigurationError(
            "n_rows doit être strictement positif", details={"n_rows": n_rows}
        )
    rng = np.random.default_rng(seed)

    age = rng.normal(45, 12, n_rows).clip(18, 95)
    balance = rng.lognormal(9.5, 1.0, n_rows)
    rate_spread = rng.normal(1.0, 0.5, n_rows) + scenario.rate_shock
    tenure = rng.integers(1, 360, n_rows).astype("float64")

    if scenario.numeric_shift:
        age = age + scenario.numeric_shift * age.std()
        balance = balance * (1 + scenario.numeric_shift)
        tenure = tenure + scenario.numeric_shift * tenure.std()

    segment = rng.choice(_SEGMENTS, n_rows, p=(0.7, 0.25, 0.05))
    product = rng.choice(_PRODUCTS, n_rows, p=(0.5, 0.3, 0.2)).astype(object)
    channel = rng.choice(_CHANNELS, n_rows, p=(0.4, 0.45, 0.15))

    # Coefficients dimensionnés pour que le « vrai » modèle atteigne une AUC d'environ
    # 0,90 sur des données propres.
    logit = (
        -5.5
        + 0.12 * (age - 45)
        + 1.5 * (np.log(balance) - 9.5)
        + 3.0 * rate_spread
        + 1.5 * (segment == "affluent")
        + 3.0 * (segment == "private")
        - 1.5 * (channel == "phone")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < prob).astype("int8")

    if scenario.label_noise:
        flip = rng.random(n_rows) < scenario.label_noise
        y = np.where(flip, 1 - y, y).astype("int8")

    score_logit = logit * scenario.score_miscalibration
    score = 1.0 / (1.0 + np.exp(-score_logit))

    if scenario.missing_rate:
        balance = balance.copy()
        balance[rng.random(n_rows) < scenario.missing_rate] = np.nan

    if scenario.unseen_category_rate:
        mask = rng.random(n_rows) < scenario.unseen_category_rate
        product[mask] = rng.choice(scenario.extra_categories, mask.sum())

    months = pd.date_range(start=start, periods=periods, freq="MS")
    dates = rng.choice(months.to_numpy(), n_rows)

    return pd.DataFrame(
        {
            TIMESTAMP: pd.to_datetime(dates),
            "age": age,
            "balance": balance,
            "rate_spread": rate_spread,
            "tenure_months": tenure,
            "segment": pd.array(segment, dtype="string"),
            "product": pd.array(product, dtype="string"),
            "channel": pd.array(channel, dtype="string"),
            TARGET: y,
            PREDICTION: score,
        }
    )


def synthetic_model_spec() -> dict[str, object]:
    """Section « model » d'une configuration de monitoring compatible avec :func:`make_dataset`."""
    return {
        "name": "synthetic_deposit_model",
        "version": "1.0.0",
        "task": "binary_classification",
        "target": TARGET,
        "prediction": PREDICTION,
        "timestamp": TIMESTAMP,
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
    }
