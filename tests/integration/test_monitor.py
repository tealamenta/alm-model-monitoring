"""Tests de bout en bout de l'orchestrateur ModelMonitor."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alm_monitoring import ModelMonitor, MonitoringConfig, __version__
from alm_monitoring.alerts import AlertManager
from alm_monitoring.config import AlertConfig
from alm_monitoring.exceptions import KPIComputationError, SchemaValidationError
from alm_monitoring.kpis import KPI, KPIResult, MonitoringContext
from alm_monitoring.monitor import DATA_AVAILABILITY_KPI
from alm_monitoring.types import KPICategory, Severity

pytestmark = pytest.mark.integration


class TestRun:
    def test_stable_window_is_ok(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        monitor = ModelMonitor(synthetic_config)
        report = monitor.run(reference_df, current_stable_df, metadata={"source": "test"})
        assert report.overall_severity is Severity.OK, [r.to_record() for r in report.breaches()]
        assert report.reference_rows == len(reference_df)
        assert report.current_rows == len(current_stable_df)
        assert report.metadata["source"] == "test"
        assert {"config_fingerprint", "package_version", "min_sample_size"} <= set(report.metadata)
        assert {r.kpi for r in report.results} == {k.name for k in monitor.kpis}
        assert monitor.alerts.process(report) is None

    def test_drifted_window_is_critical_everywhere(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_drifted_df: pd.DataFrame,
    ) -> None:
        report = ModelMonitor(synthetic_config).run(reference_df, current_drifted_df)
        assert report.overall_severity is Severity.CRITICAL
        worst = report.summary()["category_worst_severity"]
        assert all(v == "critical" for v in worst.values()), worst

    def test_alerts_are_dispatched(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_drifted_df: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monitor = ModelMonitor(synthetic_config)
        report = monitor.run(reference_df, current_drifted_df)
        with caplog.at_level(logging.CRITICAL, logger="alm_monitoring.alerts"):
            payload = monitor.alerts.process(report)
        assert payload is not None and payload["severity"] == "critical"
        assert "CRITICAL" in caplog.text

    def test_reference_validation(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        monitor = ModelMonitor(synthetic_config)
        with pytest.raises(SchemaValidationError, match="vide"):
            monitor.run(reference_df.iloc[0:0], current_stable_df)
        with pytest.raises(SchemaValidationError, match="colonnes obligatoires manquent") as exc:
            monitor.run(reference_df.drop(columns=["age", "score"]), current_stable_df)
        assert exc.value.details["missing"] == ["age", "score"]
        with pytest.raises(SchemaValidationError, match="non numérique"):
            monitor.run(reference_df.assign(age="x"), current_stable_df)

    @pytest.mark.parametrize("rows", [0, 1, 99])
    def test_small_or_empty_current_is_one_availability_error(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
        rows: int,
    ) -> None:
        """Même régime que les fenêtres temporelles : une ligne ERROR, pas une exception ni
        une erreur par KPI."""
        report = ModelMonitor(synthetic_config).run(
            reference_df, current_stable_df.iloc[:rows], metadata={"source": "t"}
        )
        assert report.overall_severity is Severity.ERROR
        assert [r.kpi for r in report.results] == [DATA_AVAILABILITY_KPI]
        assert report.results[0].value == rows
        assert report.current_rows == rows
        assert report.metadata["source"] == "t"
        assert "config_fingerprint" in report.metadata
        frame = report.to_dataframe()
        assert len(frame) == 1
        assert frame.loc[0, "severity"] == "error"

    def test_broken_current_feed_yields_report_not_crash(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        broken = current_stable_df.drop(columns=["age", "score", "y"])
        broken["balance"] = "oops"
        report = ModelMonitor(synthetic_config).run(reference_df, broken)
        assert report.overall_severity is Severity.ERROR
        schema = next(r for r in report.results if r.kpi == "schema_conformity")
        assert schema.severity is Severity.CRITICAL
        assert schema.value == 3.0
        assert all(
            r.severity is Severity.ERROR
            for r in report.results
            if r.category is KPICategory.PERFORMANCE
        )

    def test_kpi_exceptions_are_contained(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        class Explodes(KPI):
            name = "explodes"
            category = KPICategory.DATA_DRIFT

            def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
                raise KPIComputationError("numerical issue")

        class Bug(KPI):
            name = "bug"
            category = KPICategory.DATA_DRIFT

            def compute(self, ctx: MonitoringContext) -> list[KPIResult]:
                raise ZeroDivisionError

        monitor = ModelMonitor(synthetic_config, kpis=[Explodes(), Bug()])
        report = monitor.run(reference_df, current_stable_df)
        by_kpi = {r.kpi: r for r in report.results}
        assert by_kpi["explodes"].message == "numerical issue"
        assert by_kpi["bug"].message == "Erreur inattendue : ZeroDivisionError"
        assert report.overall_severity is Severity.ERROR

    def test_unsupported_kpis_are_skipped(
        self, reference_df: pd.DataFrame, current_stable_df: pd.DataFrame
    ) -> None:
        from alm_monitoring.data.synthetic import synthetic_model_spec

        raw = {"model": synthetic_model_spec(), "kpis": [{"name": "rmse"}, {"name": "psi"}]}
        monitor = ModelMonitor(MonitoringConfig.from_dict(raw))
        report = monitor.run(reference_df, current_stable_df)
        assert {r.kpi for r in report.results} == {"psi"}

    def test_custom_alert_manager(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        manager = AlertManager(AlertConfig(min_severity=Severity.CRITICAL), [])
        monitor = ModelMonitor(synthetic_config, alert_manager=manager)
        assert monitor.alerts is manager


class TestRunOverTime:
    @pytest.fixture
    def monthly_current(self) -> pd.DataFrame:
        from alm_monitoring.data.synthetic import DriftScenario, make_dataset

        stable = make_dataset(1_500, seed=5, start="2024-02-01", periods=3)
        shocked = make_dataset(
            600, seed=6, start="2024-05-01", periods=1, scenario=DriftScenario(rate_shock=2.0)
        )
        tiny = make_dataset(20, seed=7, start="2024-06-01", periods=1)
        return pd.concat([stable, shocked, tiny], ignore_index=True)

    def test_small_window_yields_availability_error_not_silence(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        monthly_current: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Une fenêtre trop petite doit apparaître dans l'historique comme une ERREUR."""
        monitor = ModelMonitor(synthetic_config)
        with caplog.at_level(logging.WARNING, logger="alm_monitoring.monitor"):
            reports = list(monitor.run_over_time(reference_df, monthly_current, metadata={"k": 1}))
        starts = [r.metadata["window_start"][:10] for r in reports]
        assert starts == ["2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
        assert all(r.metadata["k"] == 1 for r in reports)
        assert "20 lignes < min_sample_size" in caplog.text
        severities = [r.overall_severity for r in reports]
        assert severities[:3] == [Severity.OK] * 3
        assert severities[3] is Severity.CRITICAL
        # Le mois de juin (20 lignes) n'est pas ignoré : un seul pseudo-KPI en ERROR.
        tiny = reports[4]
        assert tiny.overall_severity is Severity.ERROR
        assert [r.kpi for r in tiny.results] == [DATA_AVAILABILITY_KPI]
        assert tiny.results[0].value == 20
        assert tiny.results[0].details == {"n_current": 20, "min_sample_size": 100}
        assert tiny.current_rows == 20
        history = reports[0].concat(reports)
        assert history["run_id"].nunique() == 5
        assert set(history.loc[history["kpi"] == DATA_AVAILABILITY_KPI, "severity"]) == {"error"}

    def test_empty_calendar_month_yields_availability_error(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
    ) -> None:
        """Un mois sans aucune ligne entre deux mois servis est signalé, pas sauté."""
        from alm_monitoring.data.synthetic import make_dataset

        feb = make_dataset(800, seed=11, start="2024-02-01", periods=1)
        apr = make_dataset(800, seed=12, start="2024-04-01", periods=1)
        reports = list(
            ModelMonitor(synthetic_config).run_over_time(
                reference_df, pd.concat([feb, apr], ignore_index=True)
            )
        )
        by_start = {r.metadata["window_start"][:10]: r for r in reports}
        assert set(by_start) == {"2024-02-01", "2024-03-01", "2024-04-01"}
        march = by_start["2024-03-01"]
        assert march.overall_severity is Severity.ERROR
        assert march.results[0].kpi == DATA_AVAILABILITY_KPI
        assert march.results[0].value == 0
        assert "Aucune ligne" in (march.results[0].message or "")

    def test_reports_carry_traceability_metadata(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        """Chaque ligne porte l'empreinte de configuration et la version du code."""
        report = ModelMonitor(synthetic_config).run(reference_df, current_stable_df)
        fingerprint = synthetic_config.fingerprint()
        assert report.metadata["config_fingerprint"] == fingerprint
        assert report.metadata["package_version"] == __version__
        frame = report.to_dataframe()
        assert set(frame["config_fingerprint"]) == {fingerprint}
        assert set(frame["package_version"]) == {__version__}
        assert list(frame.columns[:7]) == [
            "run_id",
            "run_at",
            "window_start",
            "model_name",
            "model_version",
            "config_fingerprint",
            "package_version",
        ]

    def test_duplicated_index_does_not_duplicate_rows(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        monthly_current: pd.DataFrame,
    ) -> None:
        """Un index non unique (concat sans ignore_index) ne doit pas gonfler les fenêtres.

        Régression : la sélection par étiquettes (``.loc``) renvoyait toutes les lignes
        portant une étiquette dupliquée et comptait des observations plusieurs fois.
        """
        monitor = ModelMonitor(synthetic_config)
        clean = list(monitor.run_over_time(reference_df, monthly_current))
        # Index volontairement dégénéré : les mêmes étiquettes 0..9 répétées.
        dirty = monthly_current.copy()
        dirty.index = pd.Index(np.arange(len(dirty)) % 10)
        assert dirty.index.has_duplicates
        dup = list(monitor.run_over_time(reference_df, dirty))
        assert [r.current_rows for r in dup] == [r.current_rows for r in clean]
        assert sum(r.current_rows for r in dup) == len(monthly_current)
        for a, b in zip(clean, dup, strict=True):
            va = {(r.kpi, r.column): r.value for r in a.results}
            vb = {(r.kpi, r.column): r.value for r in b.results}
            assert va.keys() == vb.keys()
            for key, value in va.items():
                if value is None:
                    assert vb[key] is None
                else:
                    assert vb[key] == pytest.approx(value), key

    def test_unparsable_timestamps_are_excluded_and_counted(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        monthly_current: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        dirty = monthly_current.copy()
        dirty["date"] = dirty["date"].astype(str)  # flux brut : la date arrive en texte
        dirty.loc[dirty.index[-7:], "date"] = "pas-une-date"
        monitor = ModelMonitor(synthetic_config)
        with caplog.at_level(logging.WARNING, logger="alm_monitoring.monitor"):
            reports = list(monitor.run_over_time(reference_df, dirty))
        assert sum(r.current_rows for r in reports) == len(monthly_current) - 7
        assert all(r.metadata["unparsed_timestamp_rows"] == 7 for r in reports)
        assert "7 ligne(s) sans horodatage" in caplog.text

    def test_invalid_reference_is_rejected_before_any_window(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        monthly_current: pd.DataFrame,
    ) -> None:
        """Référence vide ou hors contrat : exception à l'entrée, jamais un rapport
        data_availability, même si toutes les fenêtres sont trop petites."""
        monitor = ModelMonitor(synthetic_config)
        tiny = monthly_current.head(5)  # toutes les fenêtres < min_sample_size
        with pytest.raises(SchemaValidationError, match="vide"):
            next(monitor.run_over_time(reference_df.iloc[0:0], tiny))
        with pytest.raises(SchemaValidationError, match="colonnes obligatoires"):
            next(monitor.run_over_time(reference_df.drop(columns=["age"]), tiny))

    def test_quarterly_frequency(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        monthly_current: pd.DataFrame,
    ) -> None:
        reports = list(
            ModelMonitor(synthetic_config).run_over_time(reference_df, monthly_current, freq="QS")
        )
        assert [r.metadata["window_start"][:10] for r in reports] == ["2024-01-01", "2024-04-01"]

    def test_timestamp_errors(
        self,
        synthetic_config: MonitoringConfig,
        reference_df: pd.DataFrame,
        current_stable_df: pd.DataFrame,
    ) -> None:
        monitor = ModelMonitor(synthetic_config)
        with pytest.raises(SchemaValidationError, match="absente"):
            next(monitor.run_over_time(reference_df, current_stable_df.drop(columns=["date"])))
        with pytest.raises(SchemaValidationError, match="analysée"):
            next(monitor.run_over_time(reference_df, current_stable_df.assign(date="n/a")))
        no_ts = synthetic_config.model_copy(
            update={"model": synthetic_config.model.model_copy(update={"timestamp": None})}
        )
        with pytest.raises(
            SchemaValidationError, match="colonne d'horodatage doit être configurée"
        ):
            next(ModelMonitor(no_ts).run_over_time(reference_df, current_stable_df))
