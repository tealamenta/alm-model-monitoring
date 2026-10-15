"""Construit une ``MonitoringConfig`` à partir des paramètres du formulaire de la recette."""

from __future__ import annotations

from typing import Any

from alm_monitoring.config import MonitoringConfig


def build_config_from_params(params: dict[str, Any]) -> MonitoringConfig:
    """Traduit les champs du formulaire en configuration validée (pydantic)."""
    task = params.get("task", "binary_classification")
    kpis: list[dict[str, Any]] = [
        {"name": "schema_conformity"},
        {
            "name": "missing_rate",
            "thresholds": {
                "warning": params.get("missing_warning", 0.05),
                "critical": params.get("missing_critical", 0.20),
            },
        },
        {"name": "unseen_categories_rate"},
        {"name": "out_of_range_rate"},
        {
            "name": "psi",
            "thresholds": {
                "warning": params.get("psi_warning", 0.10),
                "critical": params.get("psi_critical", 0.25),
            },
        },
        {"name": "js_distance"},
        {"name": "ks_test"},
        {"name": "cramers_v"},
        {"name": "prediction_drift"},
        {"name": "performance_degradation"},
    ]
    if task == "binary_classification":
        kpis += [
            {
                "name": "auc",
                "thresholds": {
                    "warning": params.get("auc_warning", 0.70),
                    "critical": params.get("auc_critical", 0.65),
                },
            },
            {"name": "brier_score"},
            {"name": "calibration_error"},
            {
                "name": "level_bias",
                "thresholds": {
                    "warning": params.get("level_bias_warning", 0.02),
                    "critical": params.get("level_bias_critical", 0.04),
                },
            },
        ]
    else:
        kpis += [{"name": "rmse"}, {"name": "mae"}, {"name": "r2"}]
    raw = {
        "model": {
            "name": params["model_name"],
            "version": str(params.get("model_version", "1.0")),
            "task": task,
            "target": params["target"],
            "prediction": params["prediction"],
            "timestamp": params.get("timestamp") or None,
            "features": {
                "numeric": list(params.get("numeric_features", [])),
                "categorical": list(params.get("categorical_features", []) or []),
            },
        },
        "min_sample_size": int(params.get("min_sample_size", 500)),
        "kpis": kpis,
        "alerts": {"min_severity": "warning", "dispatchers": [{"type": "logging"}]},
    }
    return MonitoringConfig.from_dict(raw)
