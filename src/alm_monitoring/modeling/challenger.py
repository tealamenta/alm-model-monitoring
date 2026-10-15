"""Champion-challenger : décider, sur des fenêtres de production, de remplacer un modèle.

C'est l'équivalent, pour un modèle comportemental, du test A/B : on ne peut pas
« randomiser » des clients, mais on peut faire tourner le **challenger** en parallèle du
**champion** sur les mêmes fenêtres mensuelles et comparer, fenêtre par fenêtre, les
métriques qui comptent. La décision suit une règle écrite à l'avance, pas une impression
sur une courbe :

* le challenger est **promu** s'il améliore le score de Brier sur au moins ``min_share``
  des fenêtres **et** n'aggrave pas l'erreur de calibration en moyenne ;
* il est **rejeté** s'il est moins bon sur Brier dans la majorité des fenêtres ;
* sinon la décision est **indéterminée** : on prolonge la période d'observation.

Le résultat est une table par fenêtre (prête pour un dataset Dataiku) et une décision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from alm_monitoring.exceptions import SchemaValidationError
from alm_monitoring.modeling.benchmark import compute_metrics

_LOG = logging.getLogger(__name__)
DEFAULT_MIN_SHARE: Final[float] = 2 / 3


@dataclass(frozen=True, slots=True)
class ChallengerDecision:
    """Décision et éléments chiffrés."""

    decision: str  # "promouvoir" | "rejeter" | "indetermine"
    reason: str
    windows: int
    share_better_brier: float
    mean_delta_brier: float
    mean_delta_calibration: float

    def to_dict(self) -> dict[str, Any]:
        """Représentation sérialisable."""
        return {
            "decision": self.decision,
            "reason": self.reason,
            "windows": self.windows,
            "share_better_brier": self.share_better_brier,
            "mean_delta_brier": self.mean_delta_brier,
            "mean_delta_calibration": self.mean_delta_calibration,
        }


def compare_by_window(
    frame: pd.DataFrame,
    *,
    target: str,
    timestamp: str,
    champion_score: str,
    challenger_score: str,
    freq: str = "MS",
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Métriques du champion et du challenger par fenêtre temporelle.

    Returns:
        Une ligne par fenêtre : ``window_start, rows, auc_*, brier_*, calibration_*,
        level_bias_*, delta_brier, delta_calibration`` (delta = challenger - champion).
    """
    for column in (target, timestamp, champion_score, challenger_score):
        if column not in frame.columns:
            raise SchemaValidationError("Colonne absente", details={"column": column})
    ts = pd.to_datetime(frame[timestamp], errors="coerce")
    rows: list[dict[str, Any]] = []
    for start, chunk in frame.groupby(ts.dt.to_period(freq[0] if freq in {"MS", "QS"} else freq)):
        if len(chunk) < min_rows:
            continue
        y = chunk[target].to_numpy(dtype="int8")
        champ = compute_metrics(y, chunk[champion_score].to_numpy())
        chall = compute_metrics(y, chunk[challenger_score].to_numpy())
        rows.append(
            {
                "window_start": pd.Period(start).to_timestamp(),
                "rows": len(chunk),
                "auc_champion": champ.auc,
                "auc_challenger": chall.auc,
                "brier_champion": champ.brier,
                "brier_challenger": chall.brier,
                "calibration_champion": champ.calibration_error,
                "calibration_challenger": chall.calibration_error,
                "level_bias_champion": champ.level_bias,
                "level_bias_challenger": chall.level_bias,
                "delta_brier": chall.brier - champ.brier,
                "delta_calibration": chall.calibration_error - champ.calibration_error,
            }
        )
    if not rows:
        raise SchemaValidationError(
            "Aucune fenêtre assez grande pour comparer", details={"min_rows": min_rows}
        )
    return pd.DataFrame(rows)


def decide(table: pd.DataFrame, *, min_share: float = DEFAULT_MIN_SHARE) -> ChallengerDecision:
    """Applique la règle de décision à la table de :func:`compare_by_window`."""
    better = (table["delta_brier"] < 0).mean()
    mean_brier = float(table["delta_brier"].mean())
    mean_cal = float(table["delta_calibration"].mean())
    n = len(table)
    if better >= min_share and mean_cal <= 0:
        decision = "promouvoir"
        reason = (
            f"Brier amélioré sur {better:.0%} des {n} fenêtres (seuil {min_share:.0%}) et "
            f"calibration non dégradée ({mean_cal:+.4f})"
        )
    elif better < 0.5:  # noqa: PLR2004
        decision = "rejeter"
        reason = f"Brier dégradé sur {1 - better:.0%} des {n} fenêtres"
    else:
        decision = "indetermine"
        reason = (
            f"Brier amélioré sur {better:.0%} des fenêtres mais "
            f"{'calibration dégradée' if mean_cal > 0 else 'sous le seuil'} : "
            "prolonger l'observation"
        )
    _LOG.info("Champion-challenger : %s (%s)", decision, reason)
    return ChallengerDecision(
        decision=decision,
        reason=reason,
        windows=n,
        share_better_brier=float(better),
        mean_delta_brier=mean_brier,
        mean_delta_calibration=mean_cal,
    )
