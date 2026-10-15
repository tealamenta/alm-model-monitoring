"""Sources de données : CSV local, Kaggle (via ``kagglehub``) et le miroir UCI.

Toutes les sources implémentent le minuscule protocole :class:`DataSource`, de sorte
que le reste du framework ignore d'où provient un DataFrame. L'adaptateur Dataiku de
:mod:`alm_monitoring.dataiku` implémente le même protocole au-dessus des datasets
Dataiku.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import requests

from alm_monitoring.exceptions import DataSourceError
from alm_monitoring.security import resolve_within, validate_url

_LOG = logging.getLogger(__name__)
_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # plafond strict (200 Mio) contre les bombes zip


@runtime_checkable
class DataSource(Protocol):
    """Tout ce qui peut produire un DataFrame."""

    def load(self) -> pd.DataFrame:
        """Renvoie le dataset sous forme de DataFrame."""
        ...


class LocalCsvSource:
    """Lit un fichier CSV situé dans un répertoire de base autorisé."""

    def __init__(self, path: Path | str, *, base_dir: Path | str, sep: str = ",") -> None:
        self._path = resolve_within(base_dir, path)
        self._sep = sep

    @property
    def path(self) -> Path:
        """Chemin résolu et validé."""
        return self._path

    def load(self) -> pd.DataFrame:  # noqa: D102
        try:
            return pd.read_csv(self._path, sep=self._sep)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataSourceError(
                "Impossible de lire le CSV", details={"path": str(self._path)}
            ) from exc


class KaggleSource:
    """Télécharge un fichier d'un dataset Kaggle avec ``kagglehub``.

    Requiert des identifiants Kaggle (``~/.kaggle/kaggle.json`` ou les variables
    d'environnement ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``). La dépendance optionnelle
    est importée paresseusement pour que la bibliothèque de base reste légère.
    """

    def __init__(self, dataset: str, filename: str, *, sep: str = ",") -> None:
        if dataset.count("/") != 1 or any(ch in dataset for ch in " ..\\"):
            raise DataSourceError(
                "Le dataset Kaggle doit être de la forme 'owner/name'",
                details={"dataset": dataset},
            )
        self._dataset = dataset
        self._filename = filename
        self._sep = sep

    def load(self) -> pd.DataFrame:  # noqa: D102
        try:
            import kagglehub  # noqa: PLC0415 - dépendance optionnelle
        except ImportError as exc:  # pragma: no cover - exercé uniquement sans l'extra
            raise DataSourceError(
                "kagglehub n'est pas installé ; installer avec "
                "`pip install alm-model-monitoring[data]`"
            ) from exc
        try:
            root = Path(kagglehub.dataset_download(self._dataset))
            file = resolve_within(root, self._filename)
            return pd.read_csv(file, sep=self._sep)
        except Exception as exc:
            raise DataSourceError(
                "Échec du téléchargement Kaggle", details={"dataset": self._dataset}
            ) from exc


class UCIBankMarketingSource:
    """Télécharge *bank-additional-full.csv* depuis le dépôt UCI (sans identifiants).

    L'archive est un zip contenant un autre zip. Les deux sont extraits en mémoire avec
    des plafonds de taille et une validation des noms de membres (protection contre le
    zip-slip), et le CSV résultant est mis en cache sur disque dans ``cache_dir``.
    """

    URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    TARGET_MEMBER = "bank-additional/bank-additional-full.csv"

    def __init__(self, cache_dir: Path | str, *, timeout_s: float = 60.0) -> None:
        self._cache_dir = Path(cache_dir)
        self._timeout = timeout_s

    @property
    def cached_file(self) -> Path:
        """Emplacement du CSV mis en cache."""
        return self._cache_dir / "bank-additional-full.csv"

    def load(self) -> pd.DataFrame:  # noqa: D102
        if not self.cached_file.exists():
            self._download()
        return pd.read_csv(self.cached_file, sep=";")

    def _download(self) -> None:
        validate_url(self.URL, allowed_hosts=["archive.ics.uci.edu"])
        _LOG.info("Téléchargement du dataset Bank Marketing depuis UCI")
        try:
            response = requests.get(self.URL, timeout=self._timeout, stream=True)
            response.raise_for_status()
            payload = _read_capped(response.iter_content(chunk_size=1 << 16))
        except requests.RequestException as exc:
            raise DataSourceError("Échec du téléchargement UCI") from exc
        csv_bytes = self._extract(payload)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_file.write_bytes(csv_bytes)

    def _extract(self, outer_bytes: bytes) -> bytes:
        try:
            with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
                inner_name = _find_member(outer, "bank-additional.zip")
                inner_bytes = _read_member(outer, inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                csv_name = _find_member(inner, self.TARGET_MEMBER)
                return _read_member(inner, csv_name)
        except zipfile.BadZipFile as exc:
            raise DataSourceError("L'archive téléchargée est corrompue") from exc


def _read_capped(chunks: Iterator[bytes]) -> bytes:
    buffer = io.BytesIO()
    for chunk in chunks:
        buffer.write(chunk)
        if buffer.tell() > _MAX_DOWNLOAD_BYTES:
            raise DataSourceError("Le téléchargement dépasse le plafond de taille")
    return buffer.getvalue()


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    for info in archive.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise DataSourceError(
                "Chemin non sûr pour un membre de l'archive", details={"member": name}
            )
        if name.endswith(suffix):
            return name
    raise DataSourceError("Membre attendu introuvable dans l'archive", details={"suffix": suffix})


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _MAX_DOWNLOAD_BYTES:
        raise DataSourceError(
            "Un membre de l'archive dépasse le plafond de taille", details={"member": name}
        )
    return archive.read(name)
