"""Politique d'alerte et dispatchers.

L':class:`AlertManager` décide *si* un rapport justifie une alerte (selon la sévérité
minimale configurée) et diffuse l'alerte vers un ou plusieurs dispatchers. Les
dispatchers sont volontairement de petits adaptateurs, afin que de nouvelles destinations
(déclenchement d'un scénario Dataiku, Teams, PagerDuty...) puissent être ajoutées sans
toucher à la politique.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import requests

from alm_monitoring.config import (
    AlertConfig,
    DispatcherConfig,
    LoggingDispatcherConfig,
    WebhookDispatcherConfig,
)
from alm_monitoring.exceptions import AlertDispatchError, ConfigurationError
from alm_monitoring.report import MonitoringReport
from alm_monitoring.security import redact, validate_url
from alm_monitoring.types import Severity

_LOG = logging.getLogger(__name__)
_MAX_BREACHES_IN_PAYLOAD = 50


def build_alert_payload(report: MonitoringReport, min_severity: Severity) -> dict[str, Any]:
    """Construit le corps JSON commun à tous les dispatchers."""
    breaches = report.breaches(min_severity)
    return {
        "event": "alm_monitoring.alert",
        "severity": report.overall_severity.value,
        "model": {"name": report.model_name, "version": report.model_version},
        "run_id": report.run_id,
        "run_at": report.run_at.isoformat(),
        "n_breaches": len(breaches),
        "breaches": [
            {
                "kpi": r.kpi,
                "category": r.category.value,
                "column": r.column,
                "value": r.value,
                "severity": r.severity.value,
                "message": r.message,
            }
            for r in breaches[:_MAX_BREACHES_IN_PAYLOAD]
        ],
        "metadata": redact(report.metadata),
    }


class AlertDispatcher(ABC):
    """Remet une charge utile d'alerte à une destination."""

    @abstractmethod
    def dispatch(self, payload: dict[str, Any]) -> None:
        """Envoie la charge utile ; lève :class:`AlertDispatchError` en cas d'échec."""


class LoggingDispatcher(AlertDispatcher):
    """Écrit les alertes dans le système de logging standard (toujours sûr, sans réseau)."""

    def __init__(self, logger_name: str = "alm_monitoring.alerts") -> None:
        self._logger = logging.getLogger(logger_name)

    def dispatch(self, payload: dict[str, Any]) -> None:  # noqa: D102
        level = (
            logging.CRITICAL if payload["severity"] == Severity.CRITICAL.value else logging.WARNING
        )
        self._logger.log(
            level,
            "Modèle %s v%s : %s (%d dépassement(s)) run_id=%s",
            payload["model"]["name"],
            payload["model"]["version"],
            payload["severity"].upper(),
            payload["n_breaches"],
            payload["run_id"],
        )


class WebhookDispatcher(AlertDispatcher):
    """Envoie la charge utile en JSON (POST) vers un webhook HTTPS.

    Contrôles de sécurité :

    * URL validée : HTTPS uniquement, pas d'identifiants intégrés, liste blanche d'hôtes
      optionnelle, hôtes privés refusés sauf autorisation explicite (atténuation SSRF).
    * Bearer token lu depuis une variable d'environnement, jamais depuis la configuration.
    * Délai d'attente explicite ; vérification TLS toujours active.
    """

    def __init__(
        self,
        url: str,
        *,
        allowed_hosts: Sequence[str] | None = None,
        allow_private_hosts: bool = False,
        timeout_s: float = 5.0,
        auth_token_env: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._url = validate_url(
            url, allowed_hosts=allowed_hosts, allow_private=allow_private_hosts
        )
        self._timeout = timeout_s
        self._auth_token_env = auth_token_env
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "alm-model-monitoring"}
        if self._auth_token_env:
            token = os.environ.get(self._auth_token_env)
            if not token:
                raise ConfigurationError(
                    "La variable d'environnement du token d'authentification du webhook "
                    "n'est pas définie",
                    details={"env_var": self._auth_token_env},
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def dispatch(self, payload: dict[str, Any]) -> None:  # noqa: D102
        try:
            response = self._session.post(
                self._url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                verify=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AlertDispatchError(
                "Échec de l'envoi au webhook", details={"url_host": _host_only(self._url)}
            ) from exc


def _host_only(url: str) -> str:
    return urlsplit(url).hostname or ""


def build_dispatcher(config: DispatcherConfig) -> AlertDispatcher:
    """Instancie le dispatcher décrit par une entrée de configuration."""
    if isinstance(config, LoggingDispatcherConfig):
        return LoggingDispatcher(config.logger_name)
    if isinstance(config, WebhookDispatcherConfig):
        return WebhookDispatcher(
            config.url,
            allowed_hosts=config.allowed_hosts,
            allow_private_hosts=config.allow_private_hosts,
            timeout_s=config.timeout_s,
            auth_token_env=config.auth_token_env,
        )
    raise ConfigurationError(  # pragma: no cover - exhaustif par construction
        f"Type de dispatcher non pris en charge : {type(config).__name__}"
    )


class AlertManager:
    """Applique la politique d'escalade d'une :class:`AlertConfig` aux rapports de monitoring."""

    def __init__(
        self, config: AlertConfig, dispatchers: Sequence[AlertDispatcher] | None = None
    ) -> None:
        self._config = config
        self._dispatchers: list[AlertDispatcher] = (
            list(dispatchers)
            if dispatchers is not None
            else [build_dispatcher(c) for c in config.dispatchers]
        )

    @property
    def dispatchers(self) -> list[AlertDispatcher]:
        """Dispatchers configurés (vue en lecture seule)."""
        return list(self._dispatchers)

    def should_alert(self, report: MonitoringReport) -> bool:
        """Indique si la sévérité globale du rapport atteint le minimum configuré."""
        return report.overall_severity >= self._config.min_severity

    def process(self, report: MonitoringReport) -> dict[str, Any] | None:
        """Envoie une alerte si nécessaire et renvoie la charge utile envoyée (ou ``None``).

        Chaque dispatcher est tenté même si un précédent a échoué ; les échecs sont
        agrégés dans une seule :class:`AlertDispatchError`.
        """
        if not self.should_alert(report):
            _LOG.info(
                "Rapport %s sous le seuil d'alerte (%s)", report.run_id, report.overall_severity
            )
            return None
        payload = build_alert_payload(report, self._config.min_severity)
        failures: list[str] = []
        for dispatcher in self._dispatchers:
            try:
                dispatcher.dispatch(payload)
            except AlertDispatchError as exc:
                _LOG.error("Échec du dispatcher %s : %s", type(dispatcher).__name__, exc)
                failures.append(type(dispatcher).__name__)
        if failures:
            raise AlertDispatchError(
                "Un ou plusieurs dispatchers d'alerte ont échoué",
                details={"dispatchers": failures},
            )
        return payload
