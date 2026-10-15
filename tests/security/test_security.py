"""Tests de non-régression sécurité : traversée de chemins, SSRF, YAML, secrets, zip-slip."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import responses
import yaml

from alm_monitoring.config import MonitoringConfig
from alm_monitoring.data.sources import (
    LocalCsvSource,
    UCIBankMarketingSource,
    _find_member,
    _read_capped,
)
from alm_monitoring.exceptions import (
    ConfigurationError,
    DataSourceError,
    SecurityError,
    UnsafePathError,
    UnsafeURLError,
)
from alm_monitoring.security import redact, resolve_within, validate_url

pytestmark = pytest.mark.security


class TestPathContainment:
    def test_relative_inside(self, tmp_path: Path) -> None:
        (tmp_path / "data").mkdir()
        assert resolve_within(tmp_path, "data/x.csv") == (tmp_path / "data/x.csv").resolve()

    def test_absolute_inside(self, tmp_path: Path) -> None:
        assert resolve_within(tmp_path, tmp_path / "x.csv") == (tmp_path / "x.csv").resolve()

    @pytest.mark.parametrize("candidate", ["../x.csv", "a/../../x.csv", "/etc/passwd"])
    def test_escape_rejected(self, tmp_path: Path, candidate: str) -> None:
        with pytest.raises(UnsafePathError) as exc:
            resolve_within(tmp_path, candidate)
        assert isinstance(exc.value, SecurityError)
        assert exc.value.details["candidate"] == candidate

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        base = tmp_path / "base"
        base.mkdir()
        (base / "link").symlink_to(outside)
        with pytest.raises(UnsafePathError):
            resolve_within(base, "link/secret.csv")

    def test_local_csv_source_enforces_containment(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError):
            LocalCsvSource("../../etc/passwd", base_dir=tmp_path)


class TestUrlPolicy:
    def test_https_public_host_ok(self) -> None:
        assert validate_url("https://hooks.example.com/x") == "https://hooks.example.com/x"

    @pytest.mark.parametrize(
        "url",
        [
            "http://hooks.example.com/x",
            "ftp://hooks.example.com/x",
            "file:///etc/passwd",
            "https:///path-only",
            "https://user:pw@hooks.example.com/x",
        ],
    )
    def test_bad_scheme_or_host(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url(url)

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "localhost", "[::1]"],
    )
    def test_private_hosts_blocked_by_default(self, host: str) -> None:
        with pytest.raises(UnsafeURLError, match="privés"):
            validate_url(f"https://{host}/hook")
        validate_url(f"https://{host}/hook", allow_private=True)

    def test_allow_list(self) -> None:
        validate_url("https://Hooks.Example.com/x", allowed_hosts=["hooks.example.com"])
        with pytest.raises(UnsafeURLError, match="liste blanche"):
            validate_url("https://evil.example.com/x", allowed_hosts=["hooks.example.com"])


class TestRedaction:
    def test_secret_keys_masked_recursively(self) -> None:
        payload = {
            "Authorization": "Bearer x",
            "api_key": "k",
            "API-KEY": "k",
            "passwd": "p",
            "nested": {"token": "t", "safe": 1},
            "safe": "value",
        }
        out = redact(payload)
        assert out["Authorization"] == out["api_key"] == out["API-KEY"] == out["passwd"] == "***"
        assert out["nested"] == {"token": "***", "safe": 1}
        assert out["safe"] == "value"
        assert payload["api_key"] == "k"  # original intact


class TestYamlSafety:
    def test_python_tags_are_rejected(self, tmp_path: Path) -> None:
        malicious = "!!python/object/apply:os.system ['echo pwned']\n"
        path = tmp_path / "evil.yaml"
        path.write_text(malicious, encoding="utf-8")
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_yaml(path)
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_yaml_string(malicious)

    def test_safe_loader_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[object] = []
        original = yaml.load

        def spy(stream: object, Loader: object) -> object:  # noqa: N803 - reflète l'API yaml
            seen.append(Loader)
            return original(stream, Loader=Loader)  # type: ignore[arg-type]

        monkeypatch.setattr(yaml, "load", spy)
        with pytest.raises(ConfigurationError):
            MonitoringConfig.from_yaml_string("model: {}\nkpis: []\n")
        assert seen and all(loader is yaml.SafeLoader for loader in seen)

    def test_unknown_keys_fail_fast(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            MonitoringConfig.from_yaml_string(
                "model:\n  name: m\n  version: '1'\n  task: regression\n  target: t\n"
                "  prediction: p\n  features: {numeric: [x]}\nkpis: [{name: rmse}]\nextra: 1\n"
            )
        assert "extra" in exc.value.details["error"]


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestArchiveSafety:
    @pytest.mark.parametrize("name", ["../evil.csv", "/abs/evil.csv", "a/../../evil.csv"])
    def test_zip_slip_rejected(self, name: str) -> None:
        with (
            zipfile.ZipFile(io.BytesIO(_zip({name: b"x"}))) as zf,
            pytest.raises(DataSourceError, match="non sûr"),
        ):
            _find_member(zf, "evil.csv")

    def test_missing_member(self) -> None:
        with (
            zipfile.ZipFile(io.BytesIO(_zip({"a.csv": b"x"}))) as zf,
            pytest.raises(DataSourceError, match="introuvable"),
        ):
            _find_member(zf, "b.csv")

    def test_download_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from alm_monitoring.data import sources

        monkeypatch.setattr(sources, "_MAX_DOWNLOAD_BYTES", 10)
        with pytest.raises(DataSourceError, match="plafond de taille"):
            _read_capped(iter([b"x" * 8, b"y" * 8]))
        assert _read_capped(iter([b"ab"])) == b"ab"

    def test_member_size_cap(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from alm_monitoring.data import sources

        inner = _zip({"bank-additional/bank-additional-full.csv": b"a;b\n1;2\n" * 50})
        outer = _zip({"bank-additional.zip": inner})
        monkeypatch.setattr(sources, "_MAX_DOWNLOAD_BYTES", len(inner) - 1)
        with pytest.raises(DataSourceError, match="plafond de taille"):
            UCIBankMarketingSource(tmp_path)._extract(outer)

    @responses.activate
    def test_uci_download_end_to_end_with_mock(self, tmp_path: Path) -> None:
        csv = b'age;job;y\n30;"admin.";no\n'
        inner = _zip({"bank-additional/bank-additional-full.csv": csv})
        outer = _zip({"bank-additional.zip": inner})
        responses.get(UCIBankMarketingSource.URL, body=outer, status=200)
        source = UCIBankMarketingSource(tmp_path / "cache")
        frame = source.load()
        assert list(frame.columns) == ["age", "job", "y"]
        assert source.cached_file.exists()
        # Le second appel utilise le cache, pas le réseau.
        assert len(responses.calls) == 1
        source.load()
        assert len(responses.calls) == 1

    @responses.activate
    def test_uci_corrupted_archive(self, tmp_path: Path) -> None:
        responses.get(UCIBankMarketingSource.URL, body=b"not a zip", status=200)
        with pytest.raises(DataSourceError, match="corrompue"):
            UCIBankMarketingSource(tmp_path).load()

    @responses.activate
    def test_uci_http_failure(self, tmp_path: Path) -> None:
        responses.get(UCIBankMarketingSource.URL, status=503)
        with pytest.raises(DataSourceError, match="téléchargement UCI"):
            UCIBankMarketingSource(tmp_path).load()
