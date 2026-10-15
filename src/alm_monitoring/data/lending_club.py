"""Préparation du jeu de données *Lending Club* pour un modèle ALM de remboursement anticipé.

Le fichier ``accepted_2007_to_2018Q4.csv`` (Kaggle, ``wordsforthewise/lending-club``)
décrit 2,26 millions de prêts à la consommation américains émis entre 2007 et 2018,
avec leur statut final. Il sert ici de support à un **modèle comportemental ALM de
remboursement anticipé (prepayment)** : la probabilité qu'un prêt soit intégralement
remboursé avant son échéance contractuelle. En gestion actif-passif, ce comportement
pilote l'écoulement des actifs, donc les projections de flux et le risque de taux (IRRBB).

Trois choix méthodologiques structurent ce module ; ils sont détaillés dans
``docs/methodologie_kpi.md`` :

1. **Cible à horizon fixe.** Un prêt est *remboursé par anticipation* s'il est soldé dans
   les ``HORIZON_MONTHS`` mois suivant son émission, strictement avant son terme. Définir
   la cible sans horizon ("remboursé avant terme") introduit un biais de censure : les
   cohortes récentes n'ont pas encore eu le temps de prépayer, et les prêts encore en
   cours sont exclus, ce qui gonfle artificiellement le taux observé.
2. **Cohortes observables uniquement.** Une cohorte n'entre dans le périmètre que si son
   horizon est entièrement écoulé à la date d'arrêté du fichier (``SNAPSHOT_DATE``).
3. **Features connues à l'émission uniquement.** Les colonnes renseignées *après*
   l'octroi (paiements reçus, dernier paiement, capital restant dû, recouvrements…)
   révèlent le résultat : elles constituent une fuite de données et sont explicitement
   bannies par :data:`POST_ORIGINATION_COLUMNS` (voir :mod:`alm_monitoring.leakage`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from alm_monitoring.exceptions import DataSourceError, SchemaValidationError

_LOG = logging.getLogger(__name__)

TARGET: Final[str] = "prepaid_12m"
PREDICTION: Final[str] = "score"
TIMESTAMP: Final[str] = "issue_d"
HORIZON_MONTHS: Final[int] = 12
#: Date d'arrêté du fichier Kaggle : dernière date de paiement observée (mars 2019).
SNAPSHOT_DATE: Final[str] = "2019-03-01"

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico",
    "emp_length_years",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "mort_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "inq_last_6mths",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "grade",
    "home_ownership",
    "purpose",
    "verification_status",
    "application_type",
)
#: Attributs opérationnels de la plateforme (mode de listage du prêt sur la place de
#: marché) : ils ne décrivent pas l'emprunteur, dérivent au gré de la politique interne de
#: Lending Club et domineraient la validation adversariale ; exclus des features.
OPERATIONAL_COLUMNS: Final[tuple[str, ...]] = ("initial_list_status",)

#: Colonnes renseignées après l'octroi : elles décrivent le *résultat* du prêt et sont
#: donc interdites comme variables explicatives (fuite de données).
POST_ORIGINATION_COLUMNS: Final[tuple[str, ...]] = (
    "loan_status",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "hardship_flag",
    "hardship_type",
    "hardship_status",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_amount",
    "funded_amnt_inv",
)

#: Colonnes brutes nécessaires à la préparation (features + construction de la cible).
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
    "initial_list_status",
    "last_pymnt_d",
    "earliest_cr_line",
)

#: Features dérivées (voir :func:`engineer_features`) : connues à l'émission, lisibles par
#: le métier, évaluées hors temps dans le benchmark avant d'être retenues.
ENGINEERED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "loan_to_income",
    "installment_to_income",
    "rate_spread",
    "credit_history_years",
    "mort_acc_missing",
    "term_x_rate",
)
ENGINEERED_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("fico_band",)

_LONG_TERM_MONTHS: Final[int] = 60
_FULLY_PAID: Final[frozenset[str]] = frozenset(
    {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
)
_CATEGORICAL_RAW: Final[dict[str, str]] = dict.fromkeys(
    (
        "term",
        "grade",
        "emp_length",
        "home_ownership",
        "verification_status",
        "loan_status",
        "purpose",
        "application_type",
        "initial_list_status",
    ),
    "category",
)
_FLOAT_RAW: Final[tuple[str, ...]] = (
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
)
_EMP_LENGTH: Final[dict[str, float]] = {
    "< 1 year": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


def load_lending_club(
    path: Path | str,
    *,
    columns: Iterable[str] = RAW_COLUMNS,
    chunksize: int = 250_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Lit le fichier brut (CSV, éventuellement gzippé, ou Parquet) avec des types compacts.

    Le fichier complet pèse 1,6 Go décompressé ; la lecture par morceaux avec des types
    ``float32`` / ``category`` ramène l'empreinte mémoire sous 600 Mo. Seules les
    colonnes ``columns`` sont chargées.

    Args:
        path: Chemin du fichier ``accepted_2007_to_2018Q4.csv[.gz]`` ou d'un Parquet.
        columns: Colonnes à conserver.
        chunksize: Taille des morceaux lus.
        max_rows: Si renseigné, arrête la lecture après ce nombre de lignes (tests, démo).

    Raises:
        DataSourceError: Si le fichier est illisible ou ne contient pas les colonnes.
    """
    file = Path(path)
    wanted = list(columns)
    if not file.exists():
        raise DataSourceError("Fichier Lending Club introuvable", details={"path": str(file)})
    try:
        if file.suffix == ".parquet":
            frame = pd.read_parquet(file, columns=wanted)
            return frame.head(max_rows) if max_rows else frame
        parts: list[pd.DataFrame] = []
        read = 0
        for raw_chunk in pd.read_csv(
            file, usecols=wanted, dtype=_CATEGORICAL_RAW, chunksize=chunksize, low_memory=False
        ):
            chunk = raw_chunk[raw_chunk["issue_d"].notna()].copy()
            if "id" in chunk.columns:  # identifiants mêlant entiers et chaînes
                chunk["id"] = chunk["id"].astype("string")
            for col in _FLOAT_RAW:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
            parts.append(chunk)
            read += len(chunk)
            if max_rows and read >= max_rows:
                break
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataSourceError(
            "Lecture du fichier Lending Club impossible", details={"path": str(file)}
        ) from exc
    frame = pd.concat(parts, ignore_index=True)
    for col in _CATEGORICAL_RAW:
        if col in frame.columns:
            frame[col] = frame[col].astype("category")
    _LOG.info("Lending Club : %d lignes chargées depuis %s", len(frame), file.name)
    return frame.head(max_rows) if max_rows else frame


def prepare_lending_club(
    raw: pd.DataFrame,
    *,
    horizon_months: int = HORIZON_MONTHS,
    snapshot_date: str = SNAPSHOT_DATE,
    clean: bool = True,
) -> pd.DataFrame:
    """Construit les features à l'émission et la cible de remboursement anticipé.

    Étapes :

    * parsing des dates (``issue_d``, ``last_pymnt_d`` au format ``Dec-2015``) ;
    * features dérivées : ``term_months``, ``fico`` (milieu de fourchette),
      ``emp_length_years`` ;
    * cible ``prepaid_12m`` = prêt soldé (*Fully Paid*) au plus tard ``horizon_months``
      mois après l'émission **et** au moins un mois avant le terme contractuel ;
    * suppression des cohortes dont l'horizon n'est pas écoulé à ``snapshot_date`` ;
    * suppression de toutes les colonnes post-octroi (:data:`POST_ORIGINATION_COLUMNS`) ;
    * si ``clean`` : neutralisation des valeurs impossibles (:func:`clean_impossible_values`).

    Raises:
        SchemaValidationError: Si des colonnes brutes indispensables manquent.
    """
    required = {"issue_d", "loan_status", "last_pymnt_d", "term"} | set(_FLOAT_RAW)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise SchemaValidationError(
            "Colonnes brutes Lending Club manquantes", details={"missing": missing}
        )
    df = raw.copy()
    df[TIMESTAMP] = pd.to_datetime(df["issue_d"].astype("string"), format="%b-%Y", errors="coerce")
    last_pymnt = pd.to_datetime(
        df["last_pymnt_d"].astype("string"), format="%b-%Y", errors="coerce"
    )
    df = df[df[TIMESTAMP].notna()].copy()
    last_pymnt = last_pymnt.loc[df.index]

    df["term_months"] = df["term"].astype("string").str.extract(r"(\d+)")[0].astype("float32")
    df["fico"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2.0).astype("float32")
    df["emp_length_years"] = df["emp_length"].astype("string").map(_EMP_LENGTH).astype("float32")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")
    if clean:
        df = clean_impossible_values(df)

    months_active = (
        (last_pymnt.dt.year - df[TIMESTAMP].dt.year) * 12
        + (last_pymnt.dt.month - df[TIMESTAMP].dt.month)
    ).astype("float32")
    fully_paid = df["loan_status"].astype("string").isin(_FULLY_PAID)
    df[TARGET] = (
        fully_paid & (months_active <= horizon_months) & (months_active < df["term_months"] - 1)
    ).astype("int8")

    cutoff = pd.Timestamp(snapshot_date) - pd.DateOffset(months=horizon_months)
    observable = df[TIMESTAMP] <= cutoff
    dropped = int((~observable).sum())
    if dropped:
        _LOG.info(
            "%d prêts écartés : horizon de %d mois non écoulé à la date d'arrêté %s",
            dropped,
            horizon_months,
            snapshot_date,
        )
    df = df[observable]

    keep = ["id", TIMESTAMP, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "earliest_cr_line"]
    present = [c for c in keep if c in df.columns]
    return df[present].reset_index(drop=True)


#: Bornes de vraisemblance métier ; hors de ces bornes la valeur est remplacée par NaN
#: (les modèles retenus tolèrent les manquants) plutôt que supprimée (on garderait un
#: biais de sélection) ou plafonnée (on inventerait une valeur).
_PLAUSIBLE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "dti": (0.0, 100.0),
    "annual_inc": (1.0, 5_000_000.0),
    "revol_util": (0.0, 100.0),
    "fico": (300.0, 850.0),
    "int_rate": (0.01, 40.0),
    "loan_amnt": (1.0, float("inf")),
}


def clean_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise les valeurs impossibles identifiées par l'audit (0,6 % des lignes).

    Politique : **NaN, pas suppression ni plafonnement**. Supprimer les lignes introduirait
    un biais de sélection (les revenus nuls sont peut-être des profils particuliers) ;
    plafonner inventerait une valeur. Le nombre de valeurs neutralisées est journalisé.
    """
    out = df.copy()
    replaced: dict[str, int] = {}
    for col, (low, high) in _PLAUSIBLE_BOUNDS.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        mask = (values < low) | (values > high)
        count = int(mask.sum())
        if count:
            out.loc[mask, col] = np.nan
            replaced[col] = count
    if replaced:
        _LOG.info("Valeurs impossibles neutralisées (NaN) : %s", replaced)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées métier (toutes connues à l'émission).

    * ``loan_to_income`` et ``installment_to_income`` : taux d'effort, lisibles par un
      analyste crédit ;
    * ``rate_spread`` : écart entre le taux du prêt et le taux moyen des prêts émis le même
      mois, proxy du taux de marché à l'octroi. C'est **la** variable d'un modèle de
      remboursement anticipé : plus un prêt est cher par rapport au marché, plus
      l'incitation à refinancer est forte. Calculé par mois d'émission, donc disponible
      en production dès la clôture du mois ;
    * ``credit_history_years`` : ancienneté du premier compte de crédit à l'émission ;
    * ``mort_acc_missing`` : indicateur du manquant informatif détecté par l'audit ;
    * ``term_x_rate`` : interaction durée x taux (coût total du crédit) ;
    * ``fico_band`` : classes FICO usuelles (lisibilité, robustesse).
    """
    out = df.copy()
    income = pd.to_numeric(out["annual_inc"], errors="coerce").where(lambda s: s > 0)
    out["loan_to_income"] = (out["loan_amnt"] / income).astype("float32")
    out["installment_to_income"] = (out["installment"] / (income / 12.0)).astype("float32")
    month = out[TIMESTAMP].dt.to_period("M")
    market = out.groupby(month)["int_rate"].transform("mean")
    out["rate_spread"] = (out["int_rate"] - market).astype("float32")
    if "earliest_cr_line" in out.columns:
        first_credit = pd.to_datetime(
            out["earliest_cr_line"].astype("string"), format="%b-%Y", errors="coerce"
        )
        out["credit_history_years"] = ((out[TIMESTAMP] - first_credit).dt.days / 365.25).astype(
            "float32"
        )
    else:
        out["credit_history_years"] = np.nan
    out["mort_acc_missing"] = out["mort_acc"].isna().astype("float32")
    out["term_x_rate"] = (out["term_months"] * out["int_rate"] / 100.0).astype("float32")
    out["fico_band"] = pd.cut(
        out["fico"],
        bins=[0, 660, 700, 740, 780, 900],
        labels=["<660", "660-699", "700-739", "740-779", "780+"],
    ).astype("string")
    return out


DEFAULT_REFERENCE_PERIOD: Final[tuple[str, str]] = ("2013-01-01", "2015-12-31")
DEFAULT_CURRENT_PERIOD: Final[tuple[str, str]] = ("2016-01-01", "2017-12-31")


def split_reference_current(
    df: pd.DataFrame,
    *,
    reference_period: tuple[str, str] = DEFAULT_REFERENCE_PERIOD,
    current_period: tuple[str, str] = DEFAULT_CURRENT_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpe temporelle stricte : fenêtre de référence puis fenêtre de surveillance.

    Les deux fenêtres sont disjointes et ordonnées (la fin de la référence précède le
    début de la surveillance), ce qui garantit l'absence de fuite temporelle : aucun prêt
    "du futur" n'entre dans l'entraînement. La vérification formelle est faite par
    :func:`alm_monitoring.leakage.check_temporal_split`.

    Raises:
        SchemaValidationError: Si les fenêtres se chevauchent ou sont vides.
    """
    if TIMESTAMP not in df.columns:
        raise SchemaValidationError("Le jeu préparé n'a pas de colonne de date")
    ref_start, ref_end = (pd.Timestamp(d) for d in reference_period)
    cur_start, cur_end = (pd.Timestamp(d) for d in current_period)
    if ref_end >= cur_start:
        raise SchemaValidationError(
            "La fenêtre de référence doit se terminer avant la fenêtre de surveillance",
            details={"reference_end": str(ref_end), "current_start": str(cur_start)},
        )
    ts = df[TIMESTAMP]
    reference = df.loc[(ts >= ref_start) & (ts <= ref_end)].reset_index(drop=True)
    current = df.loc[(ts >= cur_start) & (ts <= cur_end)].reset_index(drop=True)
    if reference.empty or current.empty:
        raise SchemaValidationError(
            "Une des fenêtres est vide",
            details={"reference_rows": len(reference), "current_rows": len(current)},
        )
    return reference, current


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par année d'émission (volume, taux de prepayment, profil moyen)."""
    year = df[TIMESTAMP].dt.year.rename("annee")
    return (
        df.groupby(year)
        .agg(
            prets=("id", "size"),
            taux_prepayment=(TARGET, "mean"),
            taux_moyen=("int_rate", "mean"),
            montant_moyen=("loan_amnt", "mean"),
            fico_moyen=("fico", "mean"),
            dti_median=("dti", "median"),
            part_60_mois=("term_months", lambda s: float(np.mean(s == _LONG_TERM_MONTHS))),
        )
        .round(3)
    )
