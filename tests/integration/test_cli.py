"""Tests de la CLI (``alm-monitor``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alm_monitoring.cli import EXIT_CODES, EXIT_USAGE, main
from alm_monitoring.data import synthetic
from alm_monitoring.data.synthetic import DriftScenario
from alm_monitoring.types import Severity

pytestmark = pytest.mark.integration

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml"


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    synthetic.make_dataset(2_000, seed=1).to_csv(tmp_path / "data/reference.csv", index=False)
    synthetic.make_dataset(1_000, seed=2, start="2024-02-01").to_csv(
        tmp_path / "data/stable.csv", index=False
    )
    synthetic.make_dataset(
        1_000, seed=3, start="2024-02-01", periods=3, scenario=DriftScenario(rate_shock=2.0)
    ).to_csv(tmp_path / "data/drifted.csv", index=False)
    return tmp_path


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "alm-monitor" in capsys.readouterr().out


def test_list_kpis(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-kpis"]) == 0
    out = capsys.readouterr().out
    assert "psi" in out
    assert "plus bas = pire" in out
    assert "auc" in out


def test_validate_config_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", str(CONFIG)]) == 0
    assert "OK: synthetic_deposit_model" in capsys.readouterr().out


def test_validate_config_bad(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {}\nkpis: [{name: nope}]\n", encoding="utf-8")
    assert main(["validate-config", str(bad)]) == EXIT_USAGE
    assert "erreur [CONFIGURATION_ERROR]" in capsys.readouterr().err


def test_validate_config_unknown_kpi(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        json.dumps({"model": synthetic.synthetic_model_spec(), "kpis": [{"name": "nope"}]}),
        encoding="utf-8",
    )
    assert main(["validate-config", str(bad)]) == EXIT_USAGE
    assert "UNKNOWN_KPI" in capsys.readouterr().err


def test_run_ok_writes_outputs(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--config", str(CONFIG),
            "--reference", "data/reference.csv",
            "--current", "data/stable.csv",
            "--output", "reports/latest.json",
            "--history", "reports/history.csv",
        ]
    )  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "fenêtre complète: OK" in out
    payload = json.loads((workdir / "reports/latest.json").read_text(encoding="utf-8"))
    assert payload["overall_severity"] == "ok"
    history = pd.read_csv(workdir / "reports/history.csv")
    assert len(history) == payload["n_results"]


def test_run_by_period_critical_and_history_append(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "run",
        "--config", str(CONFIG),
        "--reference", "data/reference.csv",
        "--current", "data/drifted.csv",
        "--by-period", "MS",
        "--history", "reports/history.csv",
        "--output", "reports/latest.json",
        "--no-alerts",
    ]  # fmt: skip
    assert main(args) == 2
    out = capsys.readouterr().out
    assert out.count("CRITICAL") == 3
    payload = json.loads((workdir / "reports/latest.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 3
    first = pd.read_csv(workdir / "reports/history.csv")
    assert main(args) == 2
    second = pd.read_csv(workdir / "reports/history.csv")
    assert len(second) == 2 * len(first)
    assert list(second.columns) == list(first.columns)


def test_run_by_period_too_small_is_a_monitoring_error(workdir: Path) -> None:
    """Une fenêtre sous min_sample_size n'est pas une erreur d'usage : c'est un résultat
    ERROR (data_availability) dans l'historique, et le code de sortie correspondant."""
    synthetic.make_dataset(30, seed=9).to_csv(workdir / "data/tiny.csv", index=False)
    code = main(
        ["run", "--config", str(CONFIG), "--reference", "data/reference.csv",
         "--current", "data/tiny.csv", "--by-period", "MS", "--no-alerts",
         "--output", "out/tiny.json"]
    )  # fmt: skip
    assert code == EXIT_CODES[Severity.ERROR]
    payload = json.loads((workdir / "out/tiny.json").read_text())
    assert payload["overall_severity"] == "error"
    assert [r["kpi"] for r in payload["results"]] == ["data_availability"]


def test_run_error_exit_code(workdir: Path) -> None:
    broken = pd.read_csv(workdir / "data/stable.csv").drop(columns=["y"])
    broken.to_csv(workdir / "data/broken.csv", index=False)
    code = main(
        ["run", "--config", str(CONFIG), "--reference", "data/reference.csv",
         "--current", "data/broken.csv", "--no-alerts"]
    )  # fmt: skip
    assert code == 3


def test_run_rejects_paths_outside_data_dir(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = workdir.parent / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    code = main(
        ["run", "--config", str(CONFIG), "--reference", str(outside),
         "--current", "data/stable.csv", "--data-dir", "data"]
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "UNSAFE_PATH" in capsys.readouterr().err


def test_run_alert_dispatch_failure_is_reported(
    workdir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from alm_monitoring import alerts
    from alm_monitoring.exceptions import AlertDispatchError

    def failing(self: alerts.AlertManager, report: object) -> None:
        raise AlertDispatchError("down")

    monkeypatch.setattr(alerts.AlertManager, "process", failing)
    code = main(
        ["-v", "run", "--config", str(CONFIG), "--reference", "data/reference.csv",
         "--current", "data/drifted.csv"]
    )  # fmt: skip
    assert code == 2
    assert "échec de l'envoi des alertes" in capsys.readouterr().err
