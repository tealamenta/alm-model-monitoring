"""Audit de fuite de données (*data leakage*) avant entraînement et avant monitoring.

Une fuite de données survient lorsqu'une information indisponible au moment de la
prédiction entre dans le modèle : le score est alors artificiellement excellent en
développement et s'effondre en production. Pour un modèle ALM de remboursement anticipé,
le risque est très concret : le fichier source contient les paiements reçus *après*
l'octroi, qui révèlent directement le résultat.

L'audit combine quatre contrôles complémentaires, chacun produisant un
:class:`LeakageFinding` avec un niveau de gravité :

1. **Liste noire de colonnes** (:func:`check_forbidden_columns`) : aucune colonne
   post-octroi ne doit figurer parmi les features.
2. **Découpe temporelle** (:func:`check_temporal_split`) : la dernière date de la
   référence précède strictement la première date de la fenêtre de surveillance.
3. **Pouvoir discriminant unitaire** (:func:`check_single_feature_power`) : une feature
   qui, seule, sépare presque parfaitement la cible (AUC > ``max_single_auc``) est
   suspecte — un signal aussi fort est rarement légitime pour un comportement humain.
4. **Validation adversariale** (:func:`adversarial_validation`) : un classifieur tente
   de distinguer les lignes de référence des lignes courantes. Une AUC proche de 0,5
   signifie que les deux fenêtres sont indiscernables ; une AUC élevée signale soit
   une fuite (une feature encode la période), soit une dérive réelle. Ce contrôle est
   donc **informatif** : il désigne les features à examiner, il ne tranche pas seul.

Tout est calculé sur un échantillon borné (``sample_size``) pour rester rapide sur des
millions de lignes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from alm_monitoring.exceptions import SchemaValidationError
from alm_monitoring.types import Severity

_LOG = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE: Final[int] = 50_000
DEFAULT_MAX_SINGLE_AUC: Final[float] = 0.90
DEFAULT_ADVERSARIAL_WARNING: Final[float] = 0.70
DEFAULT_ADVERSARIAL_CRITICAL: Final[float] = 0.90
_RANDOM_STATE: Final[int] = 42


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """Résultat d'un contrôle de fuite."""

    check: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Aplatit le résultat (table de synthèse, JSON)."""
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            **{f"detail_{k}": v for k, v in sorted(self.details.items())},
        }


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Ensemble des constats d'un audit."""

    findings: tuple[LeakageFinding, ...]

    @property
    def overall_severity(self) -> Severity:
        """Gravité maximale observée."""
        return max((f.severity for f in self.findings), default=Severity.OK)

    @property
    def passed(self) -> bool:
        """Vrai si aucun constat n'atteint le niveau critique."""
        return self.overall_severity < Severity.CRITICAL

    def to_dataframe(self) -> pd.DataFrame:
        """Table de synthèse (une ligne par constat)."""
        return pd.DataFrame([f.to_record() for f in self.findings])

    def summary(self) -> str:
        """Résumé lisible, une ligne par constat."""
        lines = [f"Audit de fuite : {self.overall_severity.value.upper()}"]
        lines.extend(f"- [{f.severity.value}] {f.check}: {f.message}" for f in self.findings)
        return "\n".join(lines)


# --------------------------------------------------------------------- contrôles


def check_forbidden_columns(features: Sequence[str], forbidden: Sequence[str]) -> LeakageFinding:
    """Vérifie qu'aucune colonne interdite (post-octroi) n'est utilisée comme feature."""
    leaked = sorted(set(features) & set(forbidden))
    if leaked:
        return LeakageFinding(
            check="forbidden_columns",
            severity=Severity.CRITICAL,
            message=f"{len(leaked)} feature(s) post-octroi détectée(s)",
            details={"columns": leaked},
        )
    return LeakageFinding(
        check="forbidden_columns",
        severity=Severity.OK,
        message="aucune colonne post-octroi parmi les features",
        details={"n_features": len(features)},
    )


def check_temporal_split(
    reference: pd.DataFrame, current: pd.DataFrame, timestamp: str
) -> LeakageFinding:
    """Vérifie que la référence précède strictement la fenêtre de surveillance."""
    if timestamp not in reference.columns or timestamp not in current.columns:
        raise SchemaValidationError("Colonne de date absente", details={"timestamp": timestamp})
    ref_max = pd.to_datetime(reference[timestamp]).max()
    cur_min = pd.to_datetime(current[timestamp]).min()
    if ref_max >= cur_min:
        return LeakageFinding(
            check="temporal_split",
            severity=Severity.CRITICAL,
            message="les fenêtres se chevauchent : fuite temporelle possible",
            details={"reference_max": str(ref_max), "current_min": str(cur_min)},
        )
    return LeakageFinding(
        check="temporal_split",
        severity=Severity.OK,
        message="la référence précède strictement la surveillance",
        details={"reference_max": str(ref_max), "current_min": str(cur_min)},
    )


def check_single_feature_power(
    frame: pd.DataFrame,
    *,
    target: str,
    numeric_features: Sequence[str],
    max_single_auc: float = DEFAULT_MAX_SINGLE_AUC,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> LeakageFinding:
    """Mesure l'AUC de chaque feature numérique prise isolément contre la cible.

    L'AUC est symétrisée (``max(auc, 1 - auc)``) : une relation fortement négative est
    aussi suspecte qu'une relation fortement positive.
    """
    sample = _sample(frame, sample_size)
    y = pd.to_numeric(sample[target], errors="coerce")
    scores: dict[str, float] = {}
    for col in numeric_features:
        if col not in sample.columns:
            continue
        x = pd.to_numeric(sample[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 2 or y[mask].nunique() < 2:  # noqa: PLR2004
            continue
        auc = float(roc_auc_score(y[mask], x[mask]))
        scores[col] = max(auc, 1.0 - auc)
    suspicious = {c: round(a, 4) for c, a in scores.items() if a > max_single_auc}
    top = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5])
    if suspicious:
        return LeakageFinding(
            check="single_feature_power",
            severity=Severity.CRITICAL,
            message=f"{len(suspicious)} feature(s) séparent presque parfaitement la cible",
            details={"suspicious": suspicious, "threshold": max_single_auc},
        )
    return LeakageFinding(
        check="single_feature_power",
        severity=Severity.OK,
        message="aucune feature ne sépare la cible à elle seule",
        details={"top_auc": {c: round(a, 4) for c, a in top.items()}, "threshold": max_single_auc},
    )


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    warning_auc: float = DEFAULT_ADVERSARIAL_WARNING,
    critical_auc: float = DEFAULT_ADVERSARIAL_CRITICAL,
) -> LeakageFinding:
    """Entraîne un classifieur à distinguer référence et surveillance (validation croisée).

    L'AUC hors-pli et l'importance (par permutation simplifiée : AUC univariée) des
    features les plus discriminantes sont renvoyées pour orienter l'analyse.
    """
    ref = _sample(reference, sample_size // 2).assign(_window=0)
    cur = _sample(current, sample_size // 2).assign(_window=1)
    both = pd.concat([ref, cur], ignore_index=True)
    cols = [c for c in [*numeric_features, *categorical_features] if c in both.columns]
    x = both[cols].copy()
    for col in categorical_features:
        if col in x.columns:
            x[col] = x[col].astype("category").cat.codes.astype("float32")
    x = x.apply(pd.to_numeric, errors="coerce")
    y = both["_window"].to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=4, random_state=_RANDOM_STATE
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=_RANDOM_STATE)
    proba = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    univariate = {}
    for col in cols:
        v = x[col]
        mask = v.notna()
        if mask.sum() > 1 and len(np.unique(y[mask])) == 2:  # noqa: PLR2004
            a = float(roc_auc_score(y[mask], v[mask]))
            univariate[col] = round(max(a, 1.0 - a), 4)
    top = dict(sorted(univariate.items(), key=lambda kv: kv[1], reverse=True)[:5])
    # Ce contrôle ne distingue pas une fuite d'une dérive légitime : il ne dépasse donc
    # jamais le niveau WARNING à lui seul (cf. docstring du module). Le niveau
    # ``critical_auc`` est conservé dans les détails pour qualifier l'ampleur.
    if auc >= warning_auc:
        severity = Severity.WARNING
        message = (
            "les fenêtres sont fortement séparables"
            if auc >= critical_auc
            else "les fenêtres sont nettement distinguables"
        ) + " : dérive à documenter, fuite à exclure feature par feature"
    else:
        severity, message = Severity.OK, "référence et surveillance sont peu distinguables"
    return LeakageFinding(
        check="adversarial_validation",
        severity=severity,
        message=f"{message} (AUC adversariale = {auc:.3f})",
        details={
            "auc": round(auc, 4),
            "top_features": top,
            "n_rows": len(both),
            "warning_auc": warning_auc,
            "critical_auc": critical_auc,
        },
    )


def audit(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target: str,
    timestamp: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    forbidden_columns: Sequence[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> LeakageReport:
    """Exécute les quatre contrôles et renvoie un :class:`LeakageReport`."""
    features = [*numeric_features, *categorical_features]
    findings = [
        check_forbidden_columns(features, forbidden_columns),
        check_temporal_split(reference, current, timestamp),
        check_single_feature_power(
            reference, target=target, numeric_features=numeric_features, sample_size=sample_size
        ),
        adversarial_validation(
            reference,
            current,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            sample_size=sample_size,
        ),
    ]
    report = LeakageReport(findings=tuple(findings))
    _LOG.info("Audit de fuite terminé : %s", report.overall_severity.value)
    return report


def _sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(frame) <= size:
        return frame
    return frame.sample(n=size, random_state=_RANDOM_STATE)
