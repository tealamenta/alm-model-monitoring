"""Télécharge les données brutes dans ``data/raw/``.

* ``--dataset lending-club`` (cas principal) : ``wordsforthewise/lending-club`` sur
  Kaggle, fichier ``accepted_2007_to_2018Q4.csv.gz`` (≈ 375 Mo). Nécessite ``kagglehub``
  et des identifiants Kaggle. Sans identifiants, l'échantillon stratifié
  ``data/sample/lending_club_sample.csv.gz`` livré avec le dépôt permet de tout rejouer
  en mode ``--quick``.
* ``--dataset bank-marketing`` (second cas) : Kaggle si possible, sinon le miroir UCI
  (sans identifiants, zip imbriqué traité de façon sûre).

Utilisation::

    python scripts/download_data.py --dataset lending-club
    python scripts/download_data.py --dataset bank-marketing [--source auto|kaggle|uci]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from alm_monitoring.data.sources import KaggleSource, UCIBankMarketingSource
from alm_monitoring.exceptions import DataSourceError

KAGGLE_DATASET = "henriqueyamahata/bank-marketing"
KAGGLE_FILE = "bank-additional-full.csv"
OUTPUT_NAME = "bank-additional-full.csv"
LENDING_CLUB_DATASET = "wordsforthewise/lending-club"
LENDING_CLUB_FILE = "accepted_2007_to_2018Q4.csv.gz"


def fetch_lending_club(raw_dir: Path) -> Path:
    """Télécharge le fichier Lending Club via ``kagglehub`` et le copie dans ``raw_dir``."""
    if not _kaggle_available():
        raise DataSourceError(
            "Lending Club nécessite kagglehub et des identifiants Kaggle "
            "(~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY)"
        )
    import shutil  # noqa: PLC0415

    import kagglehub  # noqa: PLC0415

    from alm_monitoring.security import resolve_within  # noqa: PLC0415

    root = Path(kagglehub.dataset_download(LENDING_CLUB_DATASET))
    source = resolve_within(root, LENDING_CLUB_FILE)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / LENDING_CLUB_FILE
    shutil.copyfile(source, target)
    return target


def _kaggle_available() -> bool:
    has_credentials = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    )
    try:
        import kagglehub  # noqa: F401, PLC0415
    except ImportError:
        return False
    return has_credentials


def fetch(source: str, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    """Renvoie le DataFrame brut et le nom de la source effectivement utilisée."""
    if source in {"auto", "kaggle"} and _kaggle_available():
        try:
            frame = KaggleSource(KAGGLE_DATASET, KAGGLE_FILE, sep=";").load()
        except DataSourceError as exc:
            if source == "kaggle":
                raise
            logging.warning("Échec du téléchargement Kaggle (%s) ; repli sur UCI", exc)
        else:
            return frame, "kaggle"
    if source == "kaggle":
        raise DataSourceError(
            "Kaggle demandé mais kagglehub/les identifiants ne sont pas disponibles"
        )
    return UCIBankMarketingSource(raw_dir).load(), "uci"


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset", choices=["lending-club", "bank-marketing"], default="lending-club"
    )
    parser.add_argument("--source", choices=["auto", "kaggle", "uci"], default="auto")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.dataset == "lending-club":
            target = fetch_lending_club(args.raw_dir)
            print(f"source=kaggle -> {target}")
            return 0
        frame, used = fetch(args.source, args.raw_dir)
    except DataSourceError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    target = args.raw_dir / OUTPUT_NAME
    # La source UCI met déjà en cache à ce chemin exact ; les données Kaggle y sont persistées.
    if used != "uci":
        frame.to_csv(target, sep=";", index=False)
    print(f"source={used} rows={len(frame)} cols={frame.shape[1]} -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
