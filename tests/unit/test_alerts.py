"""Tests de la politique d'alerte et des dispatchers (webhook simulé avec ``responses``)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import requests
import responses

from alm_monitoring.alerts import (
    AlertDispatcher,
    AlertManager,
    LoggingDispatcher,
    WebhookDispatcher,
    build_alert_payload,
    build_dispatcher,
)
from alm_monitoring.config import AlertConfig, LoggingDispatcherConfig, WebhookDispatcherConfig
from alm_monitoring.exceptions import AlertDispatchError, ConfigurationError, UnsafeURLError
from alm_monitoring.kpis import KPIResult
from alm_monitoring.report import MonitoringReport
from alm_monitoring.types import KPICategory, Severity

pytestmark = pytest.mark.unit

URL = "https://hooks.example.com/alm"


def _report(*severities: Severity, metadata: dict[str, Any] | None = None) -> MonitoringReport:
    results = tuple(
        KPIResult("psi", KPICategory.DATA_DRIFT, f"c{i}", 0.3, s) for i, s in enumerate(severities)
    )
    return MonitoringReport("m", "1", results, 10, 10, metadata=metadata or {})


class _Recorder(AlertDispatcher):
    def __init__(self, fail: bool = False) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.fail = fail

    def dispatch(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)
        if self.fail:
            raise AlertDispatchError("nope")


class TestPayload:
    def test_structure_and_redaction(self) -> None:
        report = _report(
            Severity.OK, Severity.WARNING, Severity.CRITICAL, metadata={"token": "x", "k": 1}
        )
        payload = build_alert_payload(report, Severity.WARNING)
        assert payload["event"] == "alm_monitoring.alert"
        assert payload["severity"] == "critical"
        assert payload["n_breaches"] == 2
        assert [b["column"] for b in payload["breaches"]] == ["c1", "c2"]
        assert payload["metadata"] == {"token": "***", "k": 1}
        assert payload["model"] == {"name": "m", "version": "1"}

    def test_breaches_are_capped(self) -> None:
        report = _report(*([Severity.WARNING] * 80))
        payload = build_alert_payload(report, Severity.WARNING)
        assert payload["n_breaches"] == 80
        assert len(payload["breaches"]) == 50


class TestLoggingDispatcher:
    def test_levels(self, caplog: pytest.LogCaptureFixture) -> None:
        dispatcher = LoggingDispatcher("test.alerts")
        with caplog.at_level(logging.WARNING, logger="test.alerts"):
            dispatcher.dispatch(build_alert_payload(_report(Severity.WARNING), Severity.WARNING))
            dispatcher.dispatch(build_alert_payload(_report(Severity.CRITICAL), Severity.WARNING))
        levels = [r.levelno for r in caplog.records]
        assert levels == [logging.WARNING, logging.CRITICAL]
        assert "1 dépassement(s)" in caplog.text


class TestWebhookDispatcher:
    @responses.activate
    def test_posts_json_with_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALM_TOKEN", "abc")
        responses.post(URL, json={"ok": True}, status=200)
        WebhookDispatcher(
            URL, allowed_hosts=["hooks.example.com"], auth_token_env="ALM_TOKEN"
        ).dispatch({"severity": "critical"})
        call = responses.calls[0].request
        assert call.headers["Authorization"] == "Bearer abc"
        assert call.headers["Content-Type"] == "application/json"
        assert b'"severity": "critical"' in (call.body or b"")

    @responses.activate
    def test_http_error_wrapped(self) -> None:
        responses.post(URL, status=500)
        with pytest.raises(AlertDispatchError) as exc:
            WebhookDispatcher(URL).dispatch({"severity": "warning"})
        assert exc.value.details == {"url_host": "hooks.example.com"}

    @responses.activate
    def test_connection_error_wrapped(self) -> None:
        responses.post(URL, body=requests.ConnectionError("boom"))
        with pytest.raises(AlertDispatchError):
            WebhookDispatcher(URL).dispatch({})

    def test_missing_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALM_TOKEN", raising=False)
        with pytest.raises(ConfigurationError, match="n'est pas définie"):
            WebhookDispatcher(URL, auth_token_env="ALM_TOKEN")._headers()

    def test_url_policy_enforced_at_construction(self) -> None:
        with pytest.raises(UnsafeURLError):
            WebhookDispatcher("http://hooks.example.com/alm")
        with pytest.raises(UnsafeURLError):
            WebhookDispatcher("https://127.0.0.1/alm")
        WebhookDispatcher("https://127.0.0.1/alm", allow_private_hosts=True)


class TestBuildDispatcher:
    def test_logging(self) -> None:
        assert isinstance(build_dispatcher(LoggingDispatcherConfig()), LoggingDispatcher)

    def test_webhook(self) -> None:
        cfg = WebhookDispatcherConfig(url=URL, allowed_hosts=["hooks.example.com"], timeout_s=2)
        dispatcher = build_dispatcher(cfg)
        assert isinstance(dispatcher, WebhookDispatcher)
        assert dispatcher._timeout == 2


class TestAlertManager:
    def test_below_threshold_returns_none(self) -> None:
        rec = _Recorder()
        manager = AlertManager(AlertConfig(min_severity=Severity.CRITICAL), [rec])
        assert manager.process(_report(Severity.WARNING)) is None
        assert rec.payloads == []
        assert not manager.should_alert(_report(Severity.OK))

    def test_dispatches_to_all(self) -> None:
        a, b = _Recorder(), _Recorder()
        manager = AlertManager(AlertConfig(), [a, b])
        payload = manager.process(_report(Severity.WARNING, Severity.CRITICAL))
        assert payload is not None
        assert a.payloads == b.payloads == [payload]
        assert len(manager.dispatchers) == 2

    def test_failures_are_aggregated_after_trying_all(self) -> None:
        bad, good = _Recorder(fail=True), _Recorder()
        manager = AlertManager(AlertConfig(), [bad, good])
        with pytest.raises(AlertDispatchError) as exc:
            manager.process(_report(Severity.CRITICAL))
        assert exc.value.details == {"dispatchers": ["_Recorder"]}
        assert good.payloads  # tout de même envoyé

    def test_error_severity_triggers_alert(self) -> None:
        rec = _Recorder()
        AlertManager(AlertConfig(), [rec]).process(_report(Severity.ERROR))
        assert rec.payloads[0]["severity"] == "error"

    def test_builds_dispatchers_from_config(self) -> None:
        manager = AlertManager(AlertConfig())
        assert isinstance(manager.dispatchers[0], LoggingDispatcher)
