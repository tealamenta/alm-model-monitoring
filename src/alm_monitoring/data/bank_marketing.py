"""Préparation du dataset UCI/Kaggle *Bank Marketing* comme proxy ALM.

Le dataset recense des appels de marketing direct (mai 2008 - nov. 2010) et indique si
le client a souscrit un dépôt à terme. Il sert ici de substitut à un modèle
comportemental ALM (la propension au dépôt à terme alimente le modèle de dépôts sans
échéance).

Pourquoi c'est un bon banc d'essai de monitoring : la période couvre la crise de 2008
et le régime de taux zéro qui a suivi. Avec une fenêtre de développement de mai 2008 à
juin 2009 (le cycle de baisse des taux) et une fenêtre de monitoring de juillet 2009 à
novembre 2010, les variables macro (``euribor3m``, ``emp_var_rate``, ``nr_employed``)
continuent de sortir du support de référence tandis que le ciblage de la campagne
change (le taux de souscription bondit de ~7 % à ~46 %), ce qui est exactement le
changement de régime qu'un framework de monitoring ALM doit détecter.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from alm_monitoring.exceptions import SchemaValidationError

MONTH_ORDER: Final[tuple[str, ...]] = (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
)  # fmt: skip
FIRST_YEAR: Final[int] = 2008

TARGET: Final[str] = "y"
PREDICTION: Final[str] = "score"
TIMESTAMP: Final[str] = "date"
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "poutcome",
)
# `duration` est la durée de l'appel : inconnue avant l'appel et quasi-fuite du
# résultat, elle est donc retirée (comme le recommandent les auteurs du dataset).
LEAKY_COLUMNS: Final[tuple[str, ...]] = ("duration",)

_RAW_REQUIRED: Final[frozenset[str]] = frozenset(
    {"age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
     "day_of_week", "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
     "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"}
)  # fmt: skip


def prepare_bank_marketing(raw: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes, encode la cible et reconstruit une date mensuelle.

    Le fichier brut n'a pas de colonne d'année, mais les lignes sont en ordre
    chronologique : l'année est donc retrouvée en l'incrémentant chaque fois que
    l'indice du mois diminue.

    Raises:
        SchemaValidationError: Si des colonnes brutes obligatoires sont absentes.
    """
    missing = sorted(_RAW_REQUIRED - set(raw.columns))
    if missing:
        raise SchemaValidationError(
            "Des colonnes manquent dans le fichier Bank Marketing brut",
            details={"missing": missing},
        )
    df = raw.rename(columns=lambda c: c.replace(".", "_")).copy()
    df[TARGET] = (df[TARGET].astype("string").str.lower() == "yes").astype("int8")
    df[TIMESTAMP] = _reconstruct_dates(df["month"])
    df = df.drop(columns=list(LEAKY_COLUMNS), errors="ignore")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")
    return df


def _reconstruct_dates(month: pd.Series[str]) -> pd.Series[pd.Timestamp]:
    idx = month.astype("string").str.lower().map({m: i for i, m in enumerate(MONTH_ORDER)})
    if idx.isna().any():
        raise SchemaValidationError("Libellé de month inconnu dans les données Bank Marketing")
    month_idx = idx.to_numpy(dtype="int64")
    year_increment = np.concatenate(([0], (np.diff(month_idx) < 0).astype("int64")))
    years = FIRST_YEAR + np.cumsum(year_increment)
    dates = pd.to_datetime({"year": years, "month": month_idx + 1, "day": np.ones_like(years)})
    return pd.Series(dates, index=month.index, name=TIMESTAMP)


DEFAULT_CUTOFF: Final[str] = "2009-07-01"


def split_reference_current(
    df: pd.DataFrame, *, cutoff: str = DEFAULT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scinde sur la date reconstruite : avant ``cutoff`` la référence, après le courant.

    Le cutoff par défaut garde l'ensemble du cycle de baisse des taux S2 2008 - S1 2009
    dans la fenêtre de référence afin que les variables macro aient du support lors de
    l'ajustement du modèle ; une référence limitée à 2008 les laisse quasi constantes
    et tout modèle extrapole.
    """
    if TIMESTAMP not in df.columns:
        raise SchemaValidationError("Le dataset préparé n'a pas de colonne de date")
    cut = pd.Timestamp(cutoff)
    mask = df[TIMESTAMP] < cut
    reference = df.loc[mask].reset_index(drop=True)
    current = df.loc[~mask].reset_index(drop=True)
    if reference.empty or current.empty:
        raise SchemaValidationError(
            "Le cutoff laisse une fenêtre vide", details={"cutoff": cutoff}
        )
    return reference, current
