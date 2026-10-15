"""Tests du schéma de configuration pydantic et du chargement YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alm_monitoring.config import (
    AlertConfig,
    FeatureSpec,
    KPISpec,
    LoggingDispatcherConfig,
    ModelSpec,
    MonitoringConfig,
    Thresholds,
    WebhookDispatcherConfig,
)
from alm_monitoring.data.synthetic import synthetic_model_spec
from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.types import Severity, TaskType

pytestmark = pytest.mark.unit


def _minimal(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"model": synthetic_model_spec(), "kpis": [{"name": "psi"}]}
    raw.update(overrides)
    return raw


class TestThresholds:
    def test_requires_one_level(self) -> None:
        with pytest.raises(ValueError, match="au moins l'un"):
            Thresholds()

    def test_frozen(self) -> None:
        t = Thresholds(warning=0.1)
        with pytest.raises(ValueError):
            t.warning = 0.2  # type: ignore[misc]


class TestKPISpec:
    @pytest.mark.parametrize("name", ["Psi", "1psi", "psi-x", ""])
    def test_name_pattern(self, name: str) -> None:
        with pytest.raises(ValueError):
            KPISpec(name=name)

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            KPISpec.model_validate({"name": "psi", "threshold": {"warning": 1}})


class TestFeatureSpec:
    def test_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="à la fois numériques et catégorielles"):
            FeatureSpec(numeric=["a"], categorical=["a"])

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValueError, match="numériques en double"):
            FeatureSpec(numeric=["a", "a"])
        with pytest.raises(ValueError, match="catégorielles en double"):
            FeatureSpec(categorical=["b", "b"])

    def test_all_preserves_order(self) -> None:
        assert FeatureSpec(numeric=["a"], categorical=["b"]).all == ["a", "b"]


class TestModelSpec:
    def test_reserved_columns(self) -> None:
        raw = synthetic_model_spec()
        raw["features"]["numeric"].append("score")  # type: ignore[index]
        with pytest.raises(ValueError, match="ne peuvent pas être des variables"):
            ModelSpec.model_validate(raw)

    def test_target_differs_from_prediction(self) -> None:
        raw = synthetic_model_spec()
        raw["prediction"] = raw["target"]
        with pytest.raises(ValueError, match="doivent être distinctes"):
            ModelSpec.model_validate(raw)

    def test_task_enum(self) -> None:
        spec = ModelSpec.model_validate(synthetic_model_spec())
        assert spec.task is TaskType.BINARY_CLASSIFICATION
        assert spec.timestamp == "date"


class TestAlertConfig:
    def test_defaults_to_logging(self) -> None:
        cfg = AlertConfig()
        assert cfg.min_severity is Severity.WARNING
        assert isinstance(cfg.dispatchers[0], LoggingDispatcherConfig)

    def test_error_not_allowed_as_min(self) -> None:
        with pytest.raises(ValueError, match="ne peut pas valoir 'error'"):
            AlertConfig(min_severity=Severity.ERROR)

    def test_discriminated_union(self) -> None:
        cfg = AlertConfig.model_validate(
            {"dispatchers": [{"type": "webhook", "url": "https://x.example"}]}
        )
        assert isinstance(cfg.dispatchers[0], WebhookDispatcherConfig)
        with pytest.raises(ValueError):
            AlertConfig.model_validate({"dispatchers": [{"type": "smoke_signal"}]})

    def test_webhook_constraints(self) -> None:
        with pytest.raises(ValueError):
            WebhookDispatcherConfig(url="https://x.example", timeout_s=0)
        with pytest.raises(ValueError):
            WebhookDispatcherConfig(url="https://x.example", auth_token_env="lower_case")
        ok = WebhookDispatcherConfig(url="https://x.example", auth_token_env="ALM_TOKEN")
        assert ok.allow_private_hosts is False


class TestMonitoringConfig:
    def test_minimal(self) -> None:
        cfg = MonitoringConfig.from_dict(_minimal())
        assert cfg.min_sample_size == 100
        assert [k.name for k in cfg.enabled_kpis] == ["psi"]

    def test_requires_one_kpi(self) -> None:
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_dict(_minimal(kpis=[]))

    def test_duplicate_kpis(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            MonitoringConfig.from_dict(_minimal(kpis=[{"name": "psi"}, {"name": "psi"}]))
        assert "déclarations de KPI en double" in exc.value.details["error"]

    def test_min_sample_size_floor(self) -> None:
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_dict(_minimal(min_sample_size=5))

    def test_enabled_kpis_filter(self) -> None:
        cfg = MonitoringConfig.from_dict(
            _minimal(kpis=[{"name": "psi"}, {"name": "auc", "enabled": False}])
        )
        assert [k.name for k in cfg.enabled_kpis] == ["psi"]

    def test_from_yaml_string(self) -> None:
        cfg = MonitoringConfig.from_yaml_string(
            """
            model:
              name: m
              version: "1"
              task: regression
              target: t
              prediction: p
              features: {numeric: [x]}
            kpis: [{name: rmse}]
            """
        )
        assert cfg.model.task is TaskType.REGRESSION

    @pytest.mark.parametrize("text", ["- not a mapping", "just a string", "key: [unclosed"])
    def test_from_yaml_string_errors(self, text: str) -> None:
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_yaml_string(text)

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("model:\n  name: m\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Configuration de monitoring invalide"):
            MonitoringConfig.from_yaml(path)

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Impossible de lire"):
            MonitoringConfig.from_yaml(tmp_path / "nope.yaml")

    def test_from_yaml_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("model: [", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="pas du YAML valide"):
            MonitoringConfig.from_yaml(path)

    def test_from_yaml_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="doit être un mapping"):
            MonitoringConfig.from_yaml(path)


@pytest.mark.parametrize("name", ["bank_marketing.yaml", "synthetic.yaml"])
def test_shipped_configs_are_valid(name: str) -> None:
    from alm_monitoring.monitor import ModelMonitor

    root = Path(__file__).resolve().parents[2]
    monitor = ModelMonitor.from_yaml(root / "configs" / name)
    assert monitor.kpis


class TestFingerprint:
    """L'empreinte identifie la configuration *effective*, pas le texte du fichier."""

    def test_stable_across_formatting_and_comments(
        self, synthetic_config: MonitoringConfig
    ) -> None:
        text = (Path(__file__).parents[2] / "configs" / "synthetic.yaml").read_text("utf-8")
        reformatted = "# commentaire ajouté\n" + text.replace("\n\n", "\n")
        assert MonitoringConfig.from_yaml_string(reformatted).fingerprint() == (
            synthetic_config.fingerprint()
        )
        assert len(synthetic_config.fingerprint()) == 12

    def test_changes_when_a_threshold_changes(self, synthetic_config: MonitoringConfig) -> None:
        raw = synthetic_config.model_dump()
        psi = next(k for k in raw["kpis"] if k["name"] == "psi")
        psi["thresholds"] = {"warning": 0.11, "critical": 0.25}
        assert MonitoringConfig.from_dict(raw).fingerprint() != synthetic_config.fingerprint()

    def test_changes_when_min_sample_size_changes(
        self, synthetic_config: MonitoringConfig
    ) -> None:
        raw = synthetic_config.model_dump()
        raw["min_sample_size"] = synthetic_config.min_sample_size + 1
        assert MonitoringConfig.from_dict(raw).fingerprint() != synthetic_config.fingerprint()
