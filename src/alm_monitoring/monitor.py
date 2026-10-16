"""L'orchestrateur :class:`ModelMonitor`.

Utilisation::

    monitor = ModelMonitor.from_yaml("configs/monitoring.yaml")
    report = monitor.run(reference_df, current_df)
    monitor.alerts.process(report)

Le moniteur est indépendant du modèle : il ne connaît que le contrat de données déclaré
dans la configuration (colonnes de variables, cible, prédiction), ce qui permet
d'appliquer le même cadre à tous les modèles ALM actuels et futurs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from alm_monitoring._version import __version__
from alm_monitoring.alerts import AlertManager
from alm_monitoring.config import MonitoringConfig, Thresholds
from alm_monitoring.exceptions import AlmMonitoringError, SchemaValidationError
from alm_monitoring.kpis import KPI, KPIResult, MonitoringContext, build_kpis
from alm_monitoring.report import MonitoringReport
from alm_monitoring.types import KPICategory, Severity

_LOG = logging.getLogger(__name__)

#: Nom du pseudo-KPI émis pour une fenêtre temporelle vide ou trop petite. Il garantit
#: qu'une absence de données apparaît dans ``monitoring_kpis`` comme une **erreur**, et
#: non comme un trou silencieux que ni le dashboard ni les règles ne verraient.
DATA_AVAILABILITY_KPI = "data_availability"


class ModelMonitor:
    """Exécute les KPI configurés sur une paire de datasets (référence, courant)."""

    def __init__(
        self,
        config: MonitoringConfig,
        *,
        kpis: Sequence[KPI] | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.config = config
        self.kpis: list[KPI] = list(kpis) if kpis is not None else build_kpis(config.kpis)
        self.alerts = alert_manager or AlertManager(config.alerts)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelMonitor:
        """Construit un moniteur à partir d'un fichier de configuration YAML."""
        return cls(MonitoringConfig.from_yaml(path))

    # ------------------------------------------------------------------ run

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MonitoringReport:
        """Calcule chaque KPI et renvoie un :class:`MonitoringReport`.

        Le dataset de référence doit respecter le contrat de données (il est sous le
        contrôle de l'équipe). Le dataset courant est validé par les KPI de qualité des
        données plutôt que rejeté, afin qu'un flux amont défaillant produise un *rapport*
        et non un plantage. Même logique pour le volume : un dataset courant vide ou plus
        petit que ``min_sample_size`` produit un rapport réduit au pseudo-KPI
        :data:`DATA_AVAILABILITY_KPI` en gravité ERROR (une seule ligne explicite, plutôt
        qu'une erreur par KPI ou une exception), identique à celui des fenêtres
        temporelles de :meth:`run_over_time`.

        Raises:
            SchemaValidationError: Si le dataset de référence viole le contrat.
        """
        self._validate_reference(reference)
        if len(current) < self.config.min_sample_size:
            _LOG.warning(
                "Dataset courant : %d lignes < min_sample_size %d, rapport d'indisponibilité",
                len(current),
                self.config.min_sample_size,
            )
            return self._availability_report(reference, len(current), metadata or {})
        ctx = MonitoringContext(
            reference=reference,
            current=current,
            model=self.config.model,
            min_sample_size=self.config.min_sample_size,
        )
        results: list[KPIResult] = []
        for kpi in self.kpis:
            results.extend(self._run_kpi(kpi, ctx))
        report = MonitoringReport(
            model_name=self.config.model.name,
            model_version=self.config.model.version,
            results=tuple(results),
            reference_rows=len(reference),
            current_rows=len(current),
            metadata=self._trace_metadata(metadata),
        )
        _LOG.info(
            "Exécution de monitoring %s pour %s v%s : %s (%d résultats)",
            report.run_id,
            report.model_name,
            report.model_version,
            report.overall_severity.value,
            len(results),
        )
        return report

    def run_over_time(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        freq: str = "MS",
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[MonitoringReport]:
        """Produit un rapport par tranche temporelle de ``current``.

        Les tranches sont définies en rééchantillonnant la colonne d'horodatage configurée
        à la fréquence ``freq`` (alias de fréquence pandas, défaut : début de mois). Une
        tranche **vide ou plus petite que** ``min_sample_size`` ne produit pas les KPI
        (ils seraient instables) mais un rapport réduit à un pseudo-KPI
        :data:`DATA_AVAILABILITY_KPI` en gravité ERROR : une absence de données doit être
        visible dans l'historique, le dashboard et les règles, jamais confondue avec une
        absence de dérive.

        Raises:
            SchemaValidationError: Si le dataset de référence viole le contrat, ou si
                aucune colonne d'horodatage n'est configurée ou présente.
        """
        # La référence est un contrat : rejetée à l'entrée, une fois, avant tout découpage,
        # et non au détour de la première fenêtre (même si toutes sont trop petites).
        self._validate_reference(reference)
        ts_col = self.config.model.timestamp
        if not ts_col:
            raise SchemaValidationError(
                "Une colonne d'horodatage doit être configurée pour le découpage temporel"
            )
        if ts_col not in current.columns:
            raise SchemaValidationError(
                "Colonne d'horodatage absente du dataset courant", details={"column": ts_col}
            )
        timestamps = pd.to_datetime(current[ts_col], errors="coerce")
        if timestamps.isna().all():
            raise SchemaValidationError(
                "La colonne d'horodatage ne peut pas être analysée", details={"column": ts_col}
            )
        # Regroupement de type resample sur des POSITIONS de lignes, jamais sur les
        # étiquettes de l'index : un index dupliqué (concat sans ignore_index, jointures)
        # ferait sinon sélectionner plusieurs lignes par étiquette avec ``.loc`` et
        # dupliquerait des observations dans les fenêtres. Les lignes NaT sont écartées.
        positions = pd.Series(
            np.arange(len(current)), index=pd.DatetimeIndex(timestamps.to_numpy())
        )
        unparsed = int(timestamps.isna().sum())
        if unparsed:
            _LOG.warning(
                "%d ligne(s) sans horodatage analysable écartée(s) du découpage temporel",
                unparsed,
            )
        for period_start, index_values in positions.groupby(pd.Grouper(freq=freq), sort=True):
            chunk = current.iloc[index_values.to_numpy()]
            slice_meta = {
                **dict(metadata or {}),
                "window_start": pd.Timestamp(cast("Any", period_start)).isoformat(),
                "window_rows": len(chunk),
                "unparsed_timestamp_rows": unparsed,
            }
            yield self.run(reference, chunk, metadata=slice_meta)

    # -------------------------------------------------------------- utilitaires

    def _trace_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        """Métadonnées de traçabilité ajoutées à chaque rapport.

        ``config_fingerprint`` identifie la configuration effective (seuils, features,
        paramètres) et ``package_version`` le code de calcul : avec ``model_version`` et
        ``window_start``, un résultat est entièrement reproductible et auditable.
        """
        return {
            "config_fingerprint": self.config.fingerprint(),
            "package_version": __version__,
            "min_sample_size": self.config.min_sample_size,
            **dict(metadata or {}),
        }

    def _availability_report(
        self, reference: pd.DataFrame, rows: int, metadata: Mapping[str, Any]
    ) -> MonitoringReport:
        """Rapport ERROR pour une fenêtre sans données exploitables."""
        minimum = self.config.min_sample_size
        if rows == 0:
            message = "Aucune ligne dans la fenêtre : flux amont absent ou filtré"
        else:
            message = f"{rows} ligne(s) < min_sample_size ({minimum}) : KPI non calculés"
        result = KPIResult(
            kpi=DATA_AVAILABILITY_KPI,
            category=KPICategory.DATA_QUALITY,
            column=None,
            value=float(rows),
            severity=Severity.ERROR,
            thresholds=Thresholds(warning=float(minimum), critical=float(minimum)),
            details={"n_current": rows, "min_sample_size": minimum},
            message=message,
        )
        return MonitoringReport(
            model_name=self.config.model.name,
            model_version=self.config.model.version,
            results=(result,),
            reference_rows=len(reference),
            current_rows=rows,
            metadata=self._trace_metadata(metadata),
        )

    def _run_kpi(self, kpi: KPI, ctx: MonitoringContext) -> list[KPIResult]:
        if not kpi.supports(ctx.model.task):
            _LOG.debug("KPI %s ignoré : tâche %s non prise en charge", kpi.name, ctx.model.task)
            return []
        try:
            return kpi.compute(ctx)
        except AlmMonitoringError as exc:
            _LOG.warning("Échec du KPI %s : %s", kpi.name, exc)
            return [kpi.error_result(str(exc))]
        except Exception as exc:  # défensif : un KPI bogué ne doit pas interrompre l'exécution
            _LOG.exception("Échec inattendu dans le KPI %s", kpi.name)
            return [kpi.error_result(f"Erreur inattendue : {type(exc).__name__}")]

    def _validate_reference(self, reference: pd.DataFrame) -> None:
        if reference.empty:
            raise SchemaValidationError("Le dataset de référence est vide")
        model = self.config.model
        required = {*model.features.all, model.prediction}
        missing = sorted(required - set(reference.columns))
        if missing:
            raise SchemaValidationError(
                "Des colonnes obligatoires manquent dans le dataset de référence",
                details={"missing": missing},
            )
        non_numeric = [
            c
            for c in [*model.features.numeric, model.prediction]
            if not pd.api.types.is_numeric_dtype(reference[c])
        ]
        if non_numeric:
            raise SchemaValidationError(
                "Des colonnes numériques de référence ont un type non numérique",
                details={"columns": non_numeric},
            )
