"""Utilitaires de sécurité partagés par les chargeurs de données, les alertes et la configuration.

Ces utilitaires mettent en œuvre les garde-fous défensifs du cadre :

* ``resolve_within`` empêche la traversée de répertoires lorsqu'un chemin provient de la
  configuration ou d'un paramètre de recette Dataiku.
* ``validate_url`` impose HTTPS et une liste blanche d'hôtes optionnelle avant tout appel
  HTTP sortant (webhooks d'alerte).
* ``redact`` retire des dictionnaires les valeurs ressemblant à des secrets avant leur
  journalisation.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from alm_monitoring.exceptions import UnsafePathError, UnsafeURLError

_SECRET_KEY_PATTERN = re.compile(r"(secret|token|password|passwd|api[_-]?key|authorization)", re.I)
_REDACTED = "***"


def resolve_within(base_dir: Path | str, candidate: Path | str) -> Path:
    """Résout ``candidate`` et vérifie qu'il se trouve à l'intérieur de ``base_dir``.

    Les liens symboliques sont résolus avant le contrôle d'inclusion, de sorte qu'un lien
    pointant hors du répertoire de base est lui aussi rejeté.

    Args:
        base_dir: Répertoire dans lequel le chemin résultant doit rester.
        candidate: Chemin absolu ou relatif à valider.

    Returns:
        Le chemin entièrement résolu.

    Raises:
        UnsafePathError: Si le chemin résolu sort de ``base_dir``.
    """
    base = Path(base_dir).resolve()
    path = Path(candidate)
    resolved = (path if path.is_absolute() else base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise UnsafePathError(
            "Le chemin sort du répertoire de base autorisé",
            details={"base_dir": str(base), "candidate": str(candidate)},
        ) from exc
    return resolved


def _is_private_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", "localhost.localdomain"}


def validate_url(
    url: str,
    *,
    allowed_hosts: Iterable[str] | None = None,
    allow_private: bool = False,
) -> str:
    """Valide une URL sortante avant son utilisation par un client HTTP.

    Args:
        url: URL candidate.
        allowed_hosts: Si fourni, le nom d'hôte doit correspondre exactement à l'une de
            ces entrées.
        allow_private: Autorise les hôtes loopback / de réseau privé (utile pour les tests
            et les nœuds Dataiku on-premise). Désactivé par défaut pour limiter le SSRF.

    Returns:
        L'URL validée, inchangée.

    Raises:
        UnsafeURLError: Si le schéma n'est pas HTTPS, si l'hôte est absent, si l'hôte est
            privé alors que ``allow_private`` est faux, ou si l'hôte n'est pas dans la
            liste blanche.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UnsafeURLError(
            "Seules les URL HTTPS sont autorisées", details={"scheme": parts.scheme}
        )
    host = parts.hostname
    if not host:
        raise UnsafeURLError("L'URL n'a pas d'hôte")
    if parts.username or parts.password:
        raise UnsafeURLError("Les identifiants intégrés dans les URL ne sont pas autorisés")
    if not allow_private and _is_private_host(host):
        raise UnsafeURLError(
            "Les hôtes privés ou loopback ne sont pas autorisés", details={"host": host}
        )
    if allowed_hosts is not None:
        allowed = {h.lower() for h in allowed_hosts}
        if host.lower() not in allowed:
            raise UnsafeURLError(
                "L'hôte ne figure pas dans la liste blanche", details={"host": host}
            )
    return url


def redact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Renvoie une copie de ``payload`` où les valeurs ressemblant à des secrets sont masquées.

    Les clés sont comparées, sans tenir compte de la casse, aux noms de secrets usuels.
    Les mappings imbriqués sont traités récursivement.
    """
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if _SECRET_KEY_PATTERN.search(key):
            result[key] = _REDACTED
        elif isinstance(value, Mapping):
            result[key] = redact(value)
        else:
            result[key] = value
    return result
