"""Configuration de monitoring typée et validée.

La configuration est la source de vérité unique de *ce qui* est surveillé et de la
*manière* dont les dépassements sont escaladés. Elle est chargée depuis YAML avec
``yaml.safe_load`` uniquement (jamais ``yaml.load``), et chaque modèle interdit les clés
inconnues afin que les fautes de frappe échouent au plus tôt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alm_monitoring.exceptions import ConfigurationError
from alm_monitoring.types import Severity, TaskType


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Thresholds(_StrictModel):
    """Niveaux warning / critical d'un KPI.

    Le sens de comparaison (plus haut = pire ou plus bas = pire) est une propriété du KPI
    lui-même ; la cohérence de l'ordre est donc validée à l'instanciation du KPI.
    """

    warning: float | None = None
    critical: float | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.warning is None and self.critical is None:
            raise ValueError("au moins l'un des niveaux 'warning' ou 'critical' doit être défini")
        return self


class KPISpec(_StrictModel):
    """Déclaration d'un KPI à calculer."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    thresholds: Thresholds | None = None
    columns: list[str] | None = Field(
        default=None,
        description="Restreint le KPI à ces colonnes ; par défaut, toutes les colonnes "
        "applicables.",
    )


class FeatureSpec(_StrictModel):
    """Colonnes de variables explicatives réparties par type statistique."""

    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        overlap = set(self.numeric) & set(self.categorical)
        if overlap:
            raise ValueError(
                f"des colonnes ne peuvent être à la fois numériques et catégorielles : "
                f"{sorted(overlap)}"
            )
        if len(set(self.numeric)) != len(self.numeric):
            raise ValueError("noms de variables numériques en double")
        if len(set(self.categorical)) != len(self.categorical):
            raise ValueError("noms de variables catégorielles en double")
        return self

    @property
    def all(self) -> list[str]:
        """Tous les noms de variables, numériques en premier."""
        return [*self.numeric, *self.categorical]


class ModelSpec(_StrictModel):
    """Identité et contrat de données du modèle surveillé."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_\-]*$")
    version: str = Field(min_length=1)
    task: TaskType
    target: str = Field(min_length=1)
    prediction: str = Field(min_length=1)
    features: FeatureSpec
    timestamp: str | None = Field(
        default=None,
        description="Colonne optionnelle utilisée pour découper les KPI dans le temps.",
    )

    @model_validator(mode="after")
    def _reserved_columns(self) -> Self:
        reserved = {self.target, self.prediction}
        if self.timestamp:
            reserved.add(self.timestamp)
        clash = reserved & set(self.features.all)
        if clash:
            raise ValueError(
                f"target/prediction/timestamp ne peuvent pas être des variables : {sorted(clash)}"
            )
        if self.target == self.prediction:
            raise ValueError("les colonnes target et prediction doivent être distinctes")
        return self


class LoggingDispatcherConfig(_StrictModel):
    """Émet les alertes via le module standard ``logging``."""

    type: Literal["logging"] = "logging"
    logger_name: str = "alm_monitoring.alerts"


class WebhookDispatcherConfig(_StrictModel):
    """Envoie les alertes en JSON (POST) vers un endpoint HTTPS (Teams, Slack, Dataiku...)."""

    type: Literal["webhook"] = "webhook"
    url: str
    allowed_hosts: list[str] | None = None
    allow_private_hosts: bool = False
    timeout_s: float = Field(default=5.0, gt=0, le=60)
    auth_token_env: str | None = Field(
        default=None,
        description="Nom de la variable d'environnement contenant un bearer token. "
        "Les secrets ne sont jamais stockés dans les fichiers de configuration.",
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )


DispatcherConfig = Annotated[
    LoggingDispatcherConfig | WebhookDispatcherConfig, Field(discriminator="type")
]


class AlertConfig(_StrictModel):
    """Politique d'escalade."""

    min_severity: Severity = Severity.WARNING
    dispatchers: list[DispatcherConfig] = Field(
        default_factory=lambda: [cast("DispatcherConfig", LoggingDispatcherConfig())]
    )

    @field_validator("min_severity")
    @classmethod
    def _not_error(cls, value: Severity) -> Severity:
        if value is Severity.ERROR:
            raise ValueError("min_severity ne peut pas valoir 'error' ; utiliser 'critical'")
        return value


class MonitoringConfig(_StrictModel):
    """Objet de configuration racine."""

    model: ModelSpec
    kpis: list[KPISpec] = Field(min_length=1)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    min_sample_size: int = Field(default=100, ge=10)

    @model_validator(mode="after")
    def _unique_kpis(self) -> Self:
        names = [k.name for k in self.kpis]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"déclarations de KPI en double : {sorted(dupes)}")
        return self

    @property
    def enabled_kpis(self) -> list[KPISpec]:
        """KPI activés."""
        return [k for k in self.kpis if k.enabled]

    def fingerprint(self) -> str:
        """Empreinte SHA-256 (12 hex) de la configuration **effective**.

        Calculée sur la représentation JSON canonique du modèle validé (clés triées,
        défauts inclus), donc insensible à la mise en forme du YAML et aux commentaires,
        mais sensible à tout changement de seuil, de feature ou de paramètre. Elle est
        écrite sur chaque ligne de ``monitoring_kpis`` : six mois plus tard, on peut
        prouver quels seuils étaient en vigueur lors d'une exécution donnée.
        """
        canonical = self.model_dump_json(exclude_none=False)
        # model_dump_json ne trie pas les clés : on repasse par json pour canoniser.
        payload = json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    # ------------------------------------------------------------------ chargement

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MonitoringConfig:
        """Construit et valide une configuration à partir d'un dictionnaire simple."""
        try:
            return cls.model_validate(raw)
        except ValueError as exc:  # pydantic.ValidationError est une ValueError
            raise ConfigurationError(
                "Configuration de monitoring invalide", details={"error": str(exc)}
            ) from exc

    @classmethod
    def from_yaml(cls, path: Path | str) -> MonitoringConfig:
        """Charge un fichier de configuration avec ``yaml.safe_load``.

        Raises:
            ConfigurationError: Si le fichier est absent, n'est pas du YAML valide ou ne
                respecte pas le schéma.
        """
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                "Impossible de lire le fichier de configuration", details={"path": str(file)}
            ) from exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                "La configuration n'est pas du YAML valide", details={"path": str(file)}
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(
                "La racine de la configuration doit être un mapping", details={"path": str(file)}
            )
        return cls.from_dict(raw)

    @classmethod
    def from_yaml_string(cls, text: str) -> MonitoringConfig:
        """Charge une configuration à partir d'un document YAML en mémoire."""
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError("La configuration n'est pas du YAML valide") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("La racine de la configuration doit être un mapping")
        return cls.from_dict(raw)
