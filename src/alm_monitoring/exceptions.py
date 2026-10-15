"""Hiérarchie des exceptions du cadre de monitoring ALM.

Toute erreur levée par la bibliothèque dérive de :class:`AlmMonitoringError`, de sorte
que les appelants (scripts, recettes Dataiku, scénarios) peuvent intercepter une seule
classe de base tout en réagissant à des modes de défaillance précis. Chaque exception
porte un ``code`` lisible par machine, stable d'une version à l'autre et pouvant être
exposé sans risque dans les logs ou les charges utiles d'alerte.
"""

from __future__ import annotations

from typing import Any


class AlmMonitoringError(Exception):
    """Classe de base de toutes les erreurs du cadre.

    Attributes:
        code: Identifiant d'erreur stable et lisible par machine.
        details: Contexte structuré optionnel (ne contient jamais de secrets).
    """

    code: str = "ALM_MONITORING_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def __str__(self) -> str:
        if not self.details:
            return self.message
        ctx = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        return f"{self.message} ({ctx})"

    def to_dict(self) -> dict[str, Any]:
        """Sérialise l'erreur pour les charges utiles d'alerte et les logs structurés."""
        return {"code": self.code, "message": self.message, "details": self.details}


# --------------------------------------------------------------------------- config


class ConfigurationError(AlmMonitoringError):
    """La configuration de monitoring est absente, malformée ou incohérente."""

    code = "CONFIGURATION_ERROR"


class UnknownKPIError(ConfigurationError):
    """Un nom de KPI référencé dans la configuration n'est pas enregistré."""

    code = "UNKNOWN_KPI"


# ----------------------------------------------------------------------------- data


class DataError(AlmMonitoringError):
    """Classe de base des problèmes d'acquisition et de validation des données."""

    code = "DATA_ERROR"


class DataSourceError(DataError):
    """Une source de données est injoignable ou illisible (Kaggle, fichiers, Dataiku)."""

    code = "DATA_SOURCE_ERROR"


class SchemaValidationError(DataError):
    """Un dataset ne respecte pas le schéma déclaré."""

    code = "SCHEMA_VALIDATION_ERROR"


class InsufficientDataError(DataError):
    """Pas assez d'observations pour calculer un KPI statistiquement significatif."""

    code = "INSUFFICIENT_DATA"


# ------------------------------------------------------------------------------ kpi


class KPIComputationError(AlmMonitoringError):
    """Un KPI a échoué pendant son calcul (problème numérique, entrées incompatibles)."""

    code = "KPI_COMPUTATION_ERROR"


# -------------------------------------------------------------------------- alerting


class AlertDispatchError(AlmMonitoringError):
    """Une alerte n'a pas pu être remise à sa destination."""

    code = "ALERT_DISPATCH_ERROR"


# -------------------------------------------------------------------------- security


class SecurityError(AlmMonitoringError):
    """Une opération a été refusée parce qu'elle enfreint une politique de sécurité."""

    code = "SECURITY_ERROR"


class UnsafePathError(SecurityError):
    """Un chemin du système de fichiers sort du répertoire de base autorisé."""

    code = "UNSAFE_PATH"


class UnsafeURLError(SecurityError):
    """Une URL ne satisfait pas la politique de transport ou la liste blanche d'hôtes."""

    code = "UNSAFE_URL"


# --------------------------------------------------------------------------- dataiku


class DataikuIntegrationError(AlmMonitoringError):
    """Le SDK Dataiku est indisponible ou a renvoyé un résultat inattendu."""

    code = "DATAIKU_INTEGRATION_ERROR"
