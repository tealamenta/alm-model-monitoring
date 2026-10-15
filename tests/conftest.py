"""Fixtures partagées de la suite de tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alm_monitoring.config import ModelSpec, MonitoringConfig
from alm_monitoring.data import synthetic
from alm_monitoring.data.synthetic import DriftScenario
from alm_monitoring.kpis import MonitoringContext

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="session")
def reference_df() -> pd.DataFrame:
    return synthetic.make_dataset(3_000, seed=1)


@pytest.fixture(scope="session")
def current_stable_df() -> pd.DataFrame:
    return synthetic.make_dataset(2_000, seed=2, start="2024-02-01")


@pytest.fixture(scope="session")
def current_drifted_df() -> pd.DataFrame:
    scenario = DriftScenario(
        numeric_shift=1.0,
        rate_shock=1.5,
        missing_rate=0.30,
        unseen_category_rate=0.10,
        label_noise=0.30,
        score_miscalibration=3.0,
    )
    return synthetic.make_dataset(2_000, seed=3, scenario=scenario, start="2024-02-01")


@pytest.fixture(scope="session")
def model_spec() -> ModelSpec:
    return ModelSpec.model_validate(synthetic.synthetic_model_spec())


@pytest.fixture(scope="session")
def synthetic_config() -> MonitoringConfig:
    return MonitoringConfig.from_yaml(CONFIGS / "synthetic.yaml")


@pytest.fixture
def ctx_stable(
    reference_df: pd.DataFrame, current_stable_df: pd.DataFrame, model_spec: ModelSpec
) -> MonitoringContext:
    return MonitoringContext(reference=reference_df, current=current_stable_df, model=model_spec)


@pytest.fixture
def ctx_drifted(
    reference_df: pd.DataFrame, current_drifted_df: pd.DataFrame, model_spec: ModelSpec
) -> MonitoringContext:
    return MonitoringContext(reference=reference_df, current=current_drifted_df, model=model_spec)
