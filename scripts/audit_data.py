"""Audit de qualité des données Lending Club : verdict d'exploitabilité avant entraînement.

Exécute les dix contrôles de :mod:`alm_monitoring.data_audit` sur le fichier brut et le
jeu préparé, puis écrit :

* ``reports/data_audit_lending_club.json`` : constats sérialisés ;
* ``reports/data_audit_lending_club.md`` : rapport lisible (verdict + tables).

Le code de sortie vaut 0 si les données sont exploitables, 2 sinon : le pipeline
``make pipeline`` s'arrête avant l'entraînement en cas de verdict négatif.

Usage::

    python scripts/audit_data.py [--raw data/raw/accepted_2007_to_2018Q4.csv.gz] [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alm_monitoring import data_audit as da
from alm_monitoring.data import lending_club as lc
from alm_monitoring.exceptions import AlmMonitoringError
from alm_monitoring.security import resolve_within

_LOG = logging.getLogger("audit_data")
DUPLICATE_KEYS = (
    "loan_amnt",
    "int_rate",
    "term",
    "issue_d",
    "annual_inc",
    "dti",
    "fico_range_low",
)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    )
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--name", default="lending_club")
    args = parser.parse_args(argv)
    try:
        raw_path = resolve_within(Path.cwd(), args.raw)
        reports = resolve_within(Path.cwd(), args.reports)
        reports.mkdir(parents=True, exist_ok=True)
        raw = lc.load_lending_club(raw_path, max_rows=args.max_rows)
        # L'audit porte sur les données AVANT nettoyage : il doit voir les valeurs impossibles.
        prepared = lc.prepare_lending_club(raw, clean=False)
        report = da.audit(
            prepared,
            target=lc.TARGET,
            timestamp=lc.TIMESTAMP,
            numeric_features=lc.NUMERIC_FEATURES,
            categorical_features=lc.CATEGORICAL_FEATURES,
            raw=raw,
            rules=da.lending_club_rules(),
            duplicate_keys=DUPLICATE_KEYS,
            source_path=raw_path,
        )
    except AlmMonitoringError as exc:
        _LOG.error("Échec : %s", exc)
        return 10
    _LOG.info("\n%s", report.summary())
    (reports / f"data_audit_{args.name}.json").write_text(
        json.dumps(da.findings_to_records(report), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (reports / f"data_audit_{args.name}.md").write_text(
        da.report_to_markdown(report, title=f"Audit de qualité des données - {args.name}"),
        encoding="utf-8",
    )
    _LOG.info("Rapports écrits dans %s", reports)
    return 0 if report.exploitable else 2


if __name__ == "__main__":
    sys.exit(main())
