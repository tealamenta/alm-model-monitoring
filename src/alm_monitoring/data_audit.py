"""Audit de qualité des données et de la cible avant entraînement.

Le monitoring ne vaut que si les données sont exploitables : un KPI de dérive calculé sur
un fichier dupliqué, des revenus à 110 millions ou une cible incohérente ne mesure rien.
Cet audit répond à la question « peut-on entraîner et surveiller un modèle sur ces
données ? » par un verdict argumenté, constat par constat.

Dix contrôles, chacun renvoyant un :class:`AuditFinding` gradué (``ok`` / ``warning`` /
``critical``) :

1. **Empreinte** du fichier (hash SHA-256, lignes, colonnes) : traçabilité du résultat.
2. **Doublons** : lignes identiques, identifiants répétés, doublons « sémantiques » (mêmes
   montant, taux, durée, date, revenu : même dossier saisi deux fois).
3. **Règles métier** : valeurs impossibles (DTI négatif ou > 100, revenu nul ou > 5 M$,
   FICO hors [300, 850], taux d'utilisation > 100 %, dernier paiement avant l'émission,
   mensualité incohérente avec la formule d'annuité).
4. **Manquants et leur mécanisme** : taux par colonne et, surtout, si le fait d'être
   manquant est lié à la cible (ratio des taux de cible manquant / renseigné). Un
   manquant informatif (MNAR) doit être conservé comme signal, pas imputé en silence.
5. **Valeurs aberrantes robustes** : score z médian/MAD (insensible aux extrêmes) et
   règle IQR ; part des observations au-delà de 5 MAD.
6. **Forme des distributions** : asymétrie et aplatissement, avec recommandation de
   transformation pour les modèles linéaires.
7. **Cardinalité** : colonnes constantes ou quasi constantes (sans information),
   catégorielles à très forte cardinalité (sur-apprentissage).
8. **Multicolinéarité** : facteurs d'inflation de la variance (VIF) sur un échantillon.
9. **Cohérence temporelle** : volumes par mois, trous, mois trop petits.
10. **Qualité de la cible** : équilibre des classes par période, statuts ambigus,
    incohérences entre statut et montants (ground truth).

Tout est calculé sur un échantillon borné (``sample_size``) quand le volume le permet,
pour rester rapide sur des millions de lignes.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from alm_monitoring.types import Severity

_LOG = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE: Final[int] = 200_000
_RANDOM_STATE: Final[int] = 42
_MAD_THRESHOLD: Final[float] = 5.0
_IQR_FACTOR: Final[float] = 3.0
_NEAR_CONSTANT_SHARE: Final[float] = 0.99
_HIGH_CARDINALITY: Final[int] = 50
_VIF_WARNING: Final[float] = 5.0
_VIF_CRITICAL: Final[float] = 10.0
_SKEW_WARNING: Final[float] = 2.0
_MNAR_RATIO_WARNING: Final[float] = 0.8  # taux cible manquant / renseigné hors [0,8 ; 1,25]
_MIN_MONTH_ROWS: Final[int] = 1000
_RULE_WARNING_SHARE: Final[float] = 0.001
_RULE_CRITICAL_SHARE: Final[float] = 0.01
_IMBALANCE_WARNING: Final[float] = 0.05  # classe positive sous 5 % : déséquilibre fort
_IMBALANCE_CRITICAL: Final[float] = 0.01


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """Constat d'un contrôle."""

    check: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Aplatit le constat."""
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            **self.details,
        }


@dataclass(frozen=True, slots=True)
class DataAuditReport:
    """Ensemble des constats et verdict d'exploitabilité."""

    findings: tuple[AuditFinding, ...]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def overall_severity(self) -> Severity:
        """Gravité maximale."""
        return max((f.severity for f in self.findings), default=Severity.OK)

    @property
    def exploitable(self) -> bool:
        """Verdict : exploitable tant qu'aucun constat n'est critique."""
        return self.overall_severity < Severity.CRITICAL

    def to_dataframe(self) -> pd.DataFrame:
        """Table de synthèse (une ligne par constat)."""
        return pd.DataFrame([f.to_record() for f in self.findings])

    def summary(self) -> str:
        """Résumé lisible."""
        verdict = "EXPLOITABLE" if self.exploitable else "NON EXPLOITABLE EN L'ÉTAT"
        lines = [f"Audit de qualité : {verdict} ({self.overall_severity.value.upper()})"]
        lines.extend(f"- [{f.severity.value}] {f.check}: {f.message}" for f in self.findings)
        return "\n".join(lines)


# ----------------------------------------------------------------- utilitaires


def _sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    return frame if len(frame) <= size else frame.sample(n=size, random_state=_RANDOM_STATE)


def _share_severity(share: float) -> Severity:
    if share >= _RULE_CRITICAL_SHARE:
        return Severity.CRITICAL
    if share >= _RULE_WARNING_SHARE:
        return Severity.WARNING
    return Severity.OK


# ------------------------------------------------------------------ contrôles


def data_fingerprint(
    path: Path | str, *, rows: int | None = None, columns: int | None = None
) -> AuditFinding:
    """Empreinte SHA-256 du fichier (lu par blocs) pour tracer l'origine des résultats."""
    file = Path(path)
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return AuditFinding(
        check="fingerprint",
        severity=Severity.OK,
        message=f"{file.name} : sha256 {digest.hexdigest()[:16]}…",
        details={
            "file": file.name,
            "sha256": digest.hexdigest(),
            "bytes": file.stat().st_size,
            "rows": rows,
            "columns": columns,
        },
    )


def check_duplicates(
    frame: pd.DataFrame, *, id_column: str | None = None, key_columns: Sequence[str] = ()
) -> AuditFinding:
    """Lignes identiques, identifiants répétés et doublons sémantiques sur ``key_columns``."""
    n = len(frame)
    exact = int(frame.duplicated().sum())
    repeated_ids = (
        int(frame[id_column].duplicated().sum()) if id_column and id_column in frame else 0
    )
    keys = [c for c in key_columns if c in frame.columns]
    semantic = int(frame.duplicated(subset=keys).sum()) if keys else 0
    worst = max(exact, repeated_ids) / n if n else 0.0
    severity = (
        Severity.CRITICAL
        if worst >= _RULE_CRITICAL_SHARE
        else Severity.WARNING
        if worst > 0
        else Severity.OK
    )
    if semantic and severity is Severity.OK:
        severity = Severity.WARNING if semantic / n >= _RULE_WARNING_SHARE else Severity.OK
    return AuditFinding(
        check="duplicates",
        severity=severity,
        message=(
            f"{exact} ligne(s) identique(s), {repeated_ids} identifiant(s) répété(s), "
            f"{semantic} doublon(s) sémantique(s) sur {keys or 'aucune clé'}"
        ),
        details={
            "exact": exact,
            "repeated_ids": repeated_ids,
            "semantic": semantic,
            "keys": keys,
            "rows": n,
        },
    )


BusinessRule = tuple[str, Callable[[pd.DataFrame], "pd.Series[bool]"]]


def lending_club_rules() -> list[BusinessRule]:
    """Règles d'impossibilité métier pour le fichier Lending Club brut (préparé ou non)."""

    def _installment_mismatch(df: pd.DataFrame) -> pd.Series[bool]:
        rate = pd.to_numeric(df["int_rate"], errors="coerce") / 1200.0
        term = pd.to_numeric(
            df.get("term_months", df["term"].astype(str).str.extract(r"(\d+)")[0]), errors="coerce"
        )
        principal = pd.to_numeric(df["loan_amnt"], errors="coerce")
        annuity = principal * rate / (1 - (1 + rate) ** (-term))
        relative = (pd.to_numeric(df["installment"], errors="coerce") - annuity).abs() / annuity
        return (relative > 0.10).fillna(False)

    def _payment_before_issue(df: pd.DataFrame) -> pd.Series[bool]:
        issue = pd.to_datetime(df["issue_d"].astype(str), format="%b-%Y", errors="coerce")
        last = pd.to_datetime(df["last_pymnt_d"].astype(str), format="%b-%Y", errors="coerce")
        return (last < issue).fillna(False)

    rules: list[BusinessRule] = [
        (
            "dti_hors_bornes [0, 100]",
            lambda df: ((df["dti"] < 0) | (df["dti"] > 100)).fillna(False),
        ),
        ("revenu_annuel_nul_ou_negatif", lambda df: (df["annual_inc"] <= 0).fillna(False)),
        ("revenu_annuel_superieur_5M", lambda df: (df["annual_inc"] > 5_000_000).fillna(False)),
        (
            "fico_hors_bornes [300, 850]",
            lambda df: ((df["fico_range_low"] < 300) | (df["fico_range_high"] > 850)).fillna(
                False
            ),
        ),
        (
            "taux_hors_bornes (0, 40]",
            lambda df: ((df["int_rate"] <= 0) | (df["int_rate"] > 40)).fillna(False),
        ),
        ("montant_nul_ou_negatif", lambda df: (df["loan_amnt"] <= 0).fillna(False)),
        (
            "utilisation_revolving_superieure_100",
            lambda df: (df["revol_util"] > 100).fillna(False),
        ),
        ("mensualite_incoherente_avec_annuite (>10 %)", _installment_mismatch),
    ]
    rules.append(("dernier_paiement_avant_emission", _payment_before_issue))
    return rules


def check_business_rules(
    frame: pd.DataFrame, rules: Sequence[BusinessRule]
) -> tuple[AuditFinding, pd.DataFrame]:
    """Compte les violations de chaque règle ; la gravité suit la part de lignes touchées."""
    n = len(frame)
    rows = []
    for name, rule in rules:
        try:
            violations = int(rule(frame).sum())
        except KeyError:
            continue
        rows.append(
            {"regle": name, "violations": violations, "part": violations / n if n else 0.0}
        )
    table = pd.DataFrame(rows).sort_values("violations", ascending=False).reset_index(drop=True)
    worst = float(table["part"].max()) if len(table) else 0.0
    total = int(table["violations"].sum()) if len(table) else 0
    finding = AuditFinding(
        check="business_rules",
        severity=_share_severity(worst),
        message=(
            f"{total} violation(s) sur {len(table)} règles ; pire règle : {worst:.2%} des lignes"
        ),
        details={"total_violations": total, "worst_share": worst, "rules": len(table)},
    )
    return finding, table


def check_missing(
    frame: pd.DataFrame, *, features: Sequence[str], target: str | None = None
) -> tuple[AuditFinding, pd.DataFrame]:
    """Taux de manquants et mécanisme : le manquant est-il lié à la cible (MNAR) ?"""
    rows = []
    y = (
        pd.to_numeric(frame[target], errors="coerce")
        if target and target in frame.columns
        else None
    )
    for col in features:
        if col not in frame.columns:
            continue
        missing = frame[col].isna()
        rate = float(missing.mean())
        ratio = None
        if y is not None and 0 < missing.sum() < len(frame):
            present_rate = float(y[~missing].mean())
            ratio = float(y[missing].mean() / present_rate) if present_rate > 0 else None
        rows.append(
            {"colonne": col, "taux_manquant": rate, "ratio_cible_manquant_vs_present": ratio}
        )
    table = pd.DataFrame(rows).sort_values("taux_manquant", ascending=False).reset_index(drop=True)
    informative = table.loc[
        table["ratio_cible_manquant_vs_present"].notna()
        & (table["taux_manquant"] >= _RULE_WARNING_SHARE)
        & (
            (table["ratio_cible_manquant_vs_present"] < _MNAR_RATIO_WARNING)
            | (table["ratio_cible_manquant_vs_present"] > 1 / _MNAR_RATIO_WARNING)
        ),
        "colonne",
    ].tolist()
    heavy = table.loc[table["taux_manquant"] > 0.2, "colonne"].tolist()
    severity = Severity.CRITICAL if heavy else Severity.WARNING if informative else Severity.OK
    finding = AuditFinding(
        check="missing_values",
        severity=severity,
        message=(
            f"{len(heavy)} colonne(s) au-delà de 20 % de manquants ; "
            f"{len(informative)} manquant(s) informatif(s) lié(s) à la cible : "
            f"{informative or 'aucun'}"
        ),
        details={
            "heavy": heavy,
            "informative": informative,
            "max_rate": float(table["taux_manquant"].max()) if len(table) else 0.0,
        },
    )
    return finding, table


def check_outliers(
    frame: pd.DataFrame, *, numeric: Sequence[str], sample_size: int = DEFAULT_SAMPLE_SIZE
) -> tuple[AuditFinding, pd.DataFrame]:
    """Part d'observations extrêmes par colonne : score z robuste (médiane/MAD) et règle IQR."""
    sample = _sample(frame, sample_size)
    rows = []
    for col in numeric:
        if col not in sample.columns:
            continue
        values = pd.to_numeric(sample[col], errors="coerce").dropna().to_numpy(dtype="float64")
        if values.size < 10:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        robust_share = (
            float(np.mean(np.abs(values - median) > _MAD_THRESHOLD * mad)) if mad > 0 else 0.0
        )
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        iqr_share = (
            float(np.mean((values < q1 - _IQR_FACTOR * iqr) | (values > q3 + _IQR_FACTOR * iqr)))
            if iqr > 0
            else 0.0
        )
        rows.append(
            {
                "colonne": col,
                "part_aberrante_mad5": robust_share,
                "part_aberrante_iqr3": iqr_share,
                "mediane": median,
                "max": float(values.max()),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("part_aberrante_mad5", ascending=False)
        .reset_index(drop=True)
    )
    worst = float(table["part_aberrante_mad5"].max()) if len(table) else 0.0
    flagged = table.loc[table["part_aberrante_mad5"] > 0.02, "colonne"].tolist()
    finding = AuditFinding(
        check="outliers",
        severity=Severity.WARNING if flagged else Severity.OK,
        message=(
            f"{len(flagged)} colonne(s) avec plus de 2 % d'observations au-delà de 5 MAD : "
            f"{flagged or 'aucune'}"
        ),
        details={"flagged": flagged, "worst_share": worst},
    )
    return finding, table


def check_distributions(
    frame: pd.DataFrame, *, numeric: Sequence[str], sample_size: int = DEFAULT_SAMPLE_SIZE
) -> tuple[AuditFinding, pd.DataFrame]:
    """Asymétrie et aplatissement ; recommandation de transformation pour un modèle linéaire."""
    sample = _sample(frame, sample_size)
    rows = []
    for col in numeric:
        if col not in sample.columns:
            continue
        values = pd.to_numeric(sample[col], errors="coerce").dropna().to_numpy(dtype="float64")
        if values.size < 10 or np.ptp(values) == 0:
            continue
        skew, kurt = float(stats.skew(values)), float(stats.kurtosis(values))
        if abs(skew) > _SKEW_WARNING and values.min() >= 0:
            advice = "log1p ou classes de quantiles (modèle linéaire)"
        elif abs(skew) > _SKEW_WARNING:
            advice = "classes de quantiles (modèle linéaire)"
        else:
            advice = "aucune"
        rows.append(
            {
                "colonne": col,
                "asymetrie": skew,
                "aplatissement": kurt,
                "transformation_conseillee": advice,
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("asymetrie", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    skewed = table.loc[table["asymetrie"].abs() > _SKEW_WARNING, "colonne"].tolist()
    finding = AuditFinding(
        check="distributions",
        severity=Severity.OK,
        message=(
            f"{len(skewed)} colonne(s) fortement asymétrique(s) (|skew| > 2) : "
            f"{skewed or 'aucune'} ; sans impact sur les arbres, à transformer pour la logistique"
        ),
        details={"skewed": skewed},
    )
    return finding, table


def check_cardinality(
    frame: pd.DataFrame, *, features: Sequence[str]
) -> tuple[AuditFinding, pd.DataFrame]:
    """Colonnes constantes ou quasi constantes, catégorielles à forte cardinalité."""
    rows = []
    for col in features:
        if col not in frame.columns:
            continue
        counts = frame[col].value_counts(dropna=False, normalize=True)
        rows.append(
            {
                "colonne": col,
                "modalites": int(frame[col].nunique(dropna=False)),
                "part_modalite_dominante": float(counts.iloc[0]) if len(counts) else 1.0,
                "type": "numeric" if pd.api.types.is_numeric_dtype(frame[col]) else "categorical",
            }
        )
    table = pd.DataFrame(rows)
    constant = table.loc[
        table["part_modalite_dominante"] >= _NEAR_CONSTANT_SHARE, "colonne"
    ].tolist()
    high = table.loc[
        (table["type"] == "categorical") & (table["modalites"] > _HIGH_CARDINALITY), "colonne"
    ].tolist()
    finding = AuditFinding(
        check="cardinality",
        severity=Severity.WARNING if constant or high else Severity.OK,
        message=(
            f"{len(constant)} colonne(s) quasi constante(s) : {constant or 'aucune'} ; "
            f"{len(high)} catégorielle(s) à forte cardinalité : {high or 'aucune'}"
        ),
        details={"near_constant": constant, "high_cardinality": high},
    )
    return finding, table


def check_multicollinearity(
    frame: pd.DataFrame, *, numeric: Sequence[str], sample_size: int = 50_000
) -> tuple[AuditFinding, pd.DataFrame]:
    """Facteurs d'inflation de la variance (VIF) à partir de la matrice de corrélation."""
    cols = [c for c in numeric if c in frame.columns]
    sample = _sample(frame, sample_size)[cols].apply(pd.to_numeric, errors="coerce").dropna()
    sample = sample.loc[:, sample.std() > 0]
    if sample.shape[1] < 2:
        return AuditFinding(
            "multicollinearity", Severity.OK, "moins de deux colonnes numériques", {}
        ), pd.DataFrame()
    # VIF_j = 1 / (1 - R²_j), R²_j = qualité de la régression de la colonne j sur les
    # autres (moindres carrés). Une colinéarité exacte donne R² = 1, donc un VIF infini.
    matrix = sample.to_numpy(dtype="float64")
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    vif: list[float] = []
    for j in range(matrix.shape[1]):
        y = matrix[:, j]
        x = np.delete(matrix, j, axis=1)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coef
        r2 = 1.0 - float(residual.var()) / float(y.var())
        vif.append(float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    table = (
        pd.DataFrame({"colonne": list(sample.columns), "vif": vif})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )
    severe = table.loc[table["vif"] >= _VIF_CRITICAL, "colonne"].tolist()
    moderate = table.loc[
        (table["vif"] >= _VIF_WARNING) & (table["vif"] < _VIF_CRITICAL), "colonne"
    ].tolist()
    finding = AuditFinding(
        check="multicollinearity",
        severity=Severity.WARNING if severe or moderate else Severity.OK,
        message=(
            f"VIF >= 10 : {severe or 'aucune'} ; VIF entre 5 et 10 : {moderate or 'aucune'} "
            "(sans impact sur les arbres, à traiter pour la logistique)"
        ),
        details={"severe": severe, "moderate": moderate},
    )
    return finding, table


def check_temporal_consistency(
    frame: pd.DataFrame, *, timestamp: str, min_rows: int = _MIN_MONTH_ROWS
) -> tuple[AuditFinding, pd.DataFrame]:
    """Volumes mensuels, mois manquants et mois trop petits pour être surveillés."""
    ts = pd.to_datetime(frame[timestamp], errors="coerce")
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full, fill_value=0)
    table = pd.DataFrame({"mois": monthly.index.astype(str), "lignes": monthly.to_numpy()})
    gaps = table.loc[table["lignes"] == 0, "mois"].tolist()
    small = table.loc[(table["lignes"] > 0) & (table["lignes"] < min_rows), "mois"].tolist()
    unparsed = int(ts.isna().sum())
    severity = (
        Severity.CRITICAL if unparsed > 0 or gaps else Severity.WARNING if small else Severity.OK
    )
    finding = AuditFinding(
        check="temporal_consistency",
        severity=severity,
        message=(
            f"{len(table)} mois de {table['mois'].iloc[0]} à {table['mois'].iloc[-1]} ; "
            f"{len(gaps)} mois vide(s), {len(small)} mois sous {min_rows} lignes, "
            f"{unparsed} date(s) illisible(s)"
        ),
        details={"gaps": gaps, "small_months": small, "unparsed_dates": unparsed},
    )
    return finding, table


def check_target_quality(
    prepared: pd.DataFrame,
    *,
    target: str,
    timestamp: str,
    raw: pd.DataFrame | None = None,
) -> tuple[AuditFinding, pd.DataFrame]:
    """Équilibre des classes par année, statuts ambigus et incohérences statut / montants."""
    year = pd.to_datetime(prepared[timestamp]).dt.year
    table = (
        prepared.groupby(year)[target]
        .agg(lignes="size", taux_positif="mean")
        .rename_axis("annee")
        .reset_index()
    )
    overall = float(prepared[target].mean())
    details: dict[str, Any] = {
        "positive_rate": overall,
        "min_year_rate": float(table["taux_positif"].min()),
        "max_year_rate": float(table["taux_positif"].max()),
    }
    severity = (
        Severity.CRITICAL
        if overall < _IMBALANCE_CRITICAL
        else Severity.WARNING
        if overall < _IMBALANCE_WARNING
        else Severity.OK
    )
    notes = [
        f"taux positif {overall:.1%} "
        f"(déséquilibre {'fort' if overall < _IMBALANCE_WARNING else 'modéré'}), "
        f"de {details['min_year_rate']:.1%} à {details['max_year_rate']:.1%} selon l'année"
    ]
    if raw is not None and {"loan_status", "out_prncp", "total_pymnt", "recoveries"} <= set(
        raw.columns
    ):
        status = raw["loan_status"].astype(str)
        ambiguous = int(status.str.contains("Does not meet the credit policy").sum())
        paid_with_balance = int(((status == "Fully Paid") & (raw["out_prncp"] > 0)).sum())
        recoveries_over = int(
            ((status == "Charged Off") & (raw["recoveries"] > raw["total_pymnt"])).sum()
        )
        share = (paid_with_balance + recoveries_over) / len(raw)
        details.update(
            {
                "ambiguous_status": ambiguous,
                "fully_paid_with_balance": paid_with_balance,
                "recoveries_over_payments": recoveries_over,
            }
        )
        notes.append(
            f"{ambiguous} statut(s) ambigu(s), {paid_with_balance} prêt(s) soldé(s) avec "
            f"capital restant dû, {recoveries_over} recouvrement(s) incohérent(s)"
        )
        severity = max(severity, _share_severity(share))
    finding = AuditFinding(
        check="target_quality", severity=severity, message=" ; ".join(notes), details=details
    )
    return finding, table


# ---------------------------------------------------------------------- audit


def audit(
    prepared: pd.DataFrame,
    *,
    target: str,
    timestamp: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    raw: pd.DataFrame | None = None,
    rules: Sequence[BusinessRule] = (),
    id_column: str | None = "id",
    duplicate_keys: Sequence[str] = (),
    source_path: Path | str | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> DataAuditReport:
    """Exécute tous les contrôles et renvoie un :class:`DataAuditReport` avec ses tables."""
    features = [*numeric_features, *categorical_features]
    findings: list[AuditFinding] = []
    tables: dict[str, pd.DataFrame] = {}
    if source_path is not None:
        findings.append(
            data_fingerprint(
                source_path,
                rows=len(raw if raw is not None else prepared),
                columns=(raw if raw is not None else prepared).shape[1],
            )
        )
    findings.append(
        check_duplicates(
            raw if raw is not None else prepared, id_column=id_column, key_columns=duplicate_keys
        )
    )
    if rules and raw is not None:
        finding, tables["business_rules"] = check_business_rules(raw, rules)
        findings.append(finding)
    for name, (finding, table) in {
        "missing_values": check_missing(prepared, features=features, target=target),
        "outliers": check_outliers(prepared, numeric=numeric_features, sample_size=sample_size),
        "distributions": check_distributions(
            prepared, numeric=numeric_features, sample_size=sample_size
        ),
        "cardinality": check_cardinality(prepared, features=features),
        "multicollinearity": check_multicollinearity(prepared, numeric=numeric_features),
        "temporal_consistency": check_temporal_consistency(prepared, timestamp=timestamp),
        "target_quality": check_target_quality(
            prepared, target=target, timestamp=timestamp, raw=raw
        ),
    }.items():
        findings.append(finding)
        tables[name] = table
    report = DataAuditReport(findings=tuple(findings), tables=tables)
    _LOG.info("Audit de qualité terminé : %s", report.overall_severity.value)
    return report


def report_to_markdown(
    report: DataAuditReport, *, title: str = "Audit de qualité des données"
) -> str:
    """Rapport Markdown : verdict, constats, puis chaque table."""
    lines = [f"# {title}", "", report.summary(), ""]
    for name, table in report.tables.items():
        if table.empty:
            continue
        lines += [f"## {name}", "", table.head(40).to_markdown(index=False, floatfmt=".4f"), ""]
    return "\n".join(lines)


def findings_to_records(report: DataAuditReport) -> list[Mapping[str, Any]]:
    """Constats sérialisables (JSON)."""
    return [f.to_record() for f in report.findings]
