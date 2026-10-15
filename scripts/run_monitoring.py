"""Exécute le cadre de monitoring mois par mois sur un modèle (Lending Club par défaut).

Produit, sous ``reports/`` :

* ``history.csv``    - résultats des KPI au format long, une ligne par (fenêtre, KPI, colonne) ;
* ``latest.json``    - le rapport complet de la fenêtre la plus récente ;
* ``summary.md``     - un court condensé Markdown (statut global par fenêtre + principaux
  dépassements) ;
* ``figures/*.html`` - graphiques Plotly lorsque l'extra ``viz`` est installé.

Utilisation::

    python scripts/run_monitoring.py [--config configs/lending_club_prepayment.yaml]
                                     [--data data/processed/lending_club]
                                     [--out reports/lending_club] [--freq MS] [--no-figures]

    python scripts/run_monitoring.py --config configs/bank_marketing.yaml \
        --data data/processed/bank_marketing --out reports/bank_marketing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

from alm_monitoring.config import MonitoringConfig
from alm_monitoring.exceptions import AlmMonitoringError
from alm_monitoring.monitor import ModelMonitor
from alm_monitoring.report import MonitoringReport
from alm_monitoring.security import resolve_within
from alm_monitoring.types import Severity

MAX_LISTED = 6
TOP_DRIFTING = 3
MODEL_KPIS: tuple[str, ...] = ("prediction_drift", "auc", "calibration_error", "level_bias")


def _figure_kpis(history: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Figures à produire : les features au PSI maximal le plus élevé, puis les KPI modèle."""
    psi = history.loc[history["kpi"] == "psi"].groupby("column")["value"].max()
    top = [(str(c), "psi") for c in psi.sort_values(ascending=False).head(TOP_DRIFTING).index]
    figures: list[tuple[str, str | None]] = [("psi", column) for column, _ in top]
    figures.extend((kpi, None) for kpi in MODEL_KPIS if kpi in set(history["kpi"]))
    return figures


def _write_summary(reports: list[MonitoringReport], history: pd.DataFrame, path: Path) -> None:
    lines = ["# Synthèse du monitoring", ""]
    lines.append("| fenêtre | lignes | statut | dépassements (>= warning) |")
    lines.append("|---|---:|---|---|")
    for report in reports:
        breaches = report.breaches(Severity.WARNING)
        shown = [f"{b.kpi}{'·' + b.column if b.column else ''}" for b in breaches[:MAX_LISTED]]
        label = ", ".join(shown) + (" ..." if len(breaches) > MAX_LISTED else "")
        lines.append(
            f"| {report.metadata['window_start'][:7]} | {report.current_rows} "
            f"| **{report.overall_severity.value}** | {label} |"
        )
    worst = (
        history.loc[history["severity"].isin(["warning", "critical"])]
        .groupby(["kpi", "column"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    lines += ["", "## KPI les plus souvent en dépassement", ""]
    for key, count in worst.items():
        kpi, column = cast("tuple[str, Any]", key)
        name = kpi if pd.isna(column) else f"{kpi} · {column}"
        lines.append(f"- {name} : {count} fenêtre(s)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(history: pd.DataFrame, latest: MonitoringReport, out: Path) -> int:
    try:
        from alm_monitoring import viz  # noqa: PLC0415
    except ImportError:
        return 0
    try:
        import plotly  # noqa: F401, PLC0415
    except ImportError:
        logging.warning("plotly n'est pas installé ; figures ignorées (installer l'extra `viz`)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    viz.severity_heatmap(history).write_html(out / "heatmap.html", include_plotlyjs="cdn")
    written += 1
    viz.severity_summary(latest.to_dataframe()).write_html(
        out / "latest_summary.html", include_plotlyjs="cdn"
    )
    written += 1
    for kpi, column in _figure_kpis(history):
        try:
            fig = viz.kpi_timeseries(history, kpi, column=column)
        except AlmMonitoringError:
            continue
        suffix = f"_{column}" if column else ""
        fig.write_html(out / f"{kpi}{suffix}.html", include_plotlyjs="cdn")
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée ; le code de sortie reflète la sévérité de la dernière fenêtre (0/1/2/3)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lending_club_prepayment.yaml")
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed/lending_club"))
    parser.add_argument("--out", type=Path, default=Path("reports/lending_club"))
    parser.add_argument(
        "--freq", default="MS", help="alias de fréquence pandas (MS, QS, W-MON...)"
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-alerts", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("alm_monitoring.monitor").setLevel(logging.WARNING)

    try:
        config = MonitoringConfig.from_yaml(args.config)
        data_dir = resolve_within(Path.cwd(), args.data)
        reference = pd.read_csv(data_dir / "reference.csv")
        current = pd.read_csv(data_dir / "current.csv")
        monitor = ModelMonitor(config)
        reports = list(monitor.run_over_time(reference, current, freq=args.freq))
        if not reports:
            print("erreur : aucune fenêtre assez grande pour être surveillée", file=sys.stderr)
            return 10
        if not args.no_alerts:
            for report in reports:
                monitor.alerts.process(report)
    except (AlmMonitoringError, OSError, pd.errors.ParserError) as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 10

    args.out.mkdir(parents=True, exist_ok=True)
    history = MonitoringReport.concat(reports)
    history.to_csv(args.out / "history.csv", index=False)
    latest = reports[-1]
    (args.out / "latest.json").write_text(latest.to_json(), encoding="utf-8")
    _write_summary(reports, history, args.out / "summary.md")
    figures = 0 if args.no_figures else _write_figures(history, latest, args.out / "figures")

    for report in reports:
        n_breach = len(report.breaches(Severity.WARNING))
        print(
            f"{report.metadata['window_start'][:7]}  rows={report.current_rows:>5}  "
            f"{report.overall_severity.value:<8}  dépassements={n_breach}"
        )
    print(
        f"-> {args.out}/history.csv ({len(history)} lignes), latest.json, summary.md, "
        f"{figures} figure(s)"
    )
    return {Severity.OK: 0, Severity.WARNING: 1, Severity.CRITICAL: 2, Severity.ERROR: 3}[
        latest.overall_severity
    ]


if __name__ == "__main__":
    sys.exit(main())
