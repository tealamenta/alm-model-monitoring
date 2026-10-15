"""Interface en ligne de commande (``alm-monitor``).

Les codes de sortie reflètent la sévérité globale afin que la CLI puisse être enchaînée
dans des ordonnanceurs (cron, Airflow, une étape « Execute shell » Dataiku) : 0 OK,
1 WARNING, 2 CRITICAL, 3 ERROR (un KPI n'a pas pu être calculé), 10 erreur d'usage, de
configuration ou de données.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from alm_monitoring import __version__
from alm_monitoring.config import MonitoringConfig
from alm_monitoring.data.sources import LocalCsvSource
from alm_monitoring.exceptions import AlertDispatchError, AlmMonitoringError
from alm_monitoring.kpis import iter_kpi_metadata
from alm_monitoring.monitor import ModelMonitor
from alm_monitoring.report import MonitoringReport
from alm_monitoring.security import resolve_within
from alm_monitoring.types import Severity

EXIT_CODES: dict[Severity, int] = {
    Severity.OK: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
    Severity.ERROR: 3,
}
EXIT_USAGE = 10


def build_parser() -> argparse.ArgumentParser:
    """Crée l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="alm-monitor", description="Cadre de monitoring des modèles ALM"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="journalisation de débogage")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="calcule les KPI sur une paire référence/courant")
    run.add_argument("--config", required=True, type=Path, help="configuration YAML")
    run.add_argument("--reference", required=True, type=Path, help="CSV de référence")
    run.add_argument("--current", required=True, type=Path, help="CSV courant")
    run.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="répertoire dans lequel les chemins CSV doivent rester (défaut : cwd)",
    )
    run.add_argument("--output", type=Path, help="écrit le rapport JSON à cet emplacement")
    run.add_argument(
        "--history", type=Path, help="écrit/complète le CSV au format long à cet emplacement"
    )
    run.add_argument(
        "--by-period",
        metavar="FREQ",
        help="découpe la fenêtre courante par alias de fréquence pandas (ex. MS) "
        "sur la colonne d'horodatage",
    )
    run.add_argument("--no-alerts", action="store_true", help="ignore l'envoi des alertes")

    sub.add_parser("list-kpis", help="affiche le catalogue des KPI")

    validate = sub.add_parser("validate-config", help="valide une configuration YAML")
    validate.add_argument("config", type=Path)
    return parser


def _cmd_list_kpis() -> int:
    for meta in iter_kpi_metadata():
        thresholds = f"warn={meta['default_warning']} crit={meta['default_critical']}"
        direction = "plus haut = pire" if meta["higher_is_worse"] else "plus bas = pire"
        sys.stdout.write(
            f"{meta['name']:<26}{meta['category']:<18}{direction:<18}{thresholds}\n"
            f"{'':<26}{meta['description']}\n"
        )
    return 0


def _cmd_validate(path: Path) -> int:
    config = MonitoringConfig.from_yaml(path)
    ModelMonitor(config)  # instancie aussi les KPI, ce qui détecte les mauvais paramètres
    sys.stdout.write(
        f"OK: {config.model.name} v{config.model.version}, {len(config.enabled_kpis)} KPI\n"
    )
    return 0


def _write_outputs(
    reports: list[MonitoringReport], output: Path | None, history: Path | None, base_dir: Path
) -> None:
    if output is not None:
        target = resolve_within(base_dir, output)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in reports] if len(reports) > 1 else reports[0].to_dict()
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if history is not None:
        target = resolve_within(base_dir, history)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = MonitoringReport.concat(reports)
        frame.to_csv(
            target, mode="a" if target.exists() else "w", header=not target.exists(), index=False
        )


def _cmd_run(args: argparse.Namespace) -> int:
    monitor = ModelMonitor.from_yaml(args.config)
    reference = LocalCsvSource(args.reference, base_dir=args.data_dir).load()
    current = LocalCsvSource(args.current, base_dir=args.data_dir).load()
    if args.by_period:
        reports = list(monitor.run_over_time(reference, current, freq=args.by_period))
        if not reports:  # défensif : impossible dès qu'un horodatage est analysable
            sys.stderr.write("Aucune tranche temporelle dans le dataset courant\n")
            return EXIT_USAGE
    else:
        reports = [monitor.run(reference, current)]

    _write_outputs(reports, args.output, args.history, Path.cwd())

    worst = max(r.overall_severity for r in reports)
    for report in reports:
        summary = report.summary()
        window = summary["metadata"].get("window_start", "fenêtre complète")
        sys.stdout.write(
            f"{window}: {summary['overall_severity'].upper()} ({summary['severity_counts']})\n"
        )
    if not args.no_alerts:
        for report in reports:
            try:
                monitor.alerts.process(report)
            except AlertDispatchError as exc:
                sys.stderr.write(f"échec de l'envoi des alertes : {exc}\n")
    return EXIT_CODES[worst]


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée ; renvoie le code de sortie du processus."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "list-kpis":
            return _cmd_list_kpis()
        if args.command == "validate-config":
            return _cmd_validate(args.config)
        return _cmd_run(args)
    except AlmMonitoringError as exc:
        sys.stderr.write(f"erreur [{exc.code}] : {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
