"""Tests for the public web app's invite/session/tenant layer."""

from __future__ import annotations

import json
import time

import pytest

from web.public.backend import auth, limits, paths, tenants


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Point every durable path at a temp dir and pin the signing secret."""
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(paths, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.setattr(paths, "INVITES_FILE", tmp_path / "auth" / "invites.json")
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(paths, "VOLUME_CACHE", tmp_path / "cache" / "cache.db")
    monkeypatch.setattr(paths, "LOCAL_ROOT", tmp_path / "local")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "local" / "cache.db")
    monkeypatch.setattr(auth, "_secret", "test-secret")
    monkeypatch.setattr(auth, "_join_attempts", {})
    paths.ensure_layout()


class TestInvites:
    def test_mint_and_check(self):
        (code,) = auth.mint_codes(1, note="beta")
        assert code.startswith("CC-")
        h = auth.check_code(code)
        assert h is not None
        # case-insensitive, whitespace-tolerant
        assert auth.check_code("  " + code.lower() + " ") == h
        assert auth.check_code("CC-NOPE-NOPE") is None
        assert auth.list_codes()[0]["uses"] == 2

    def test_disable(self):
        (code,) = auth.mint_codes(1)
        h = auth.check_code(code)
        assert auth.disable_code(h[:8]) == 1
        assert auth.check_code(code) is None


class TestCookies:
    def test_roundtrip(self):
        sid = auth.create_session("h" * 64)
        ck = auth.make_cookie(sid)
        assert auth.parse_cookie(ck) == sid

    def test_tamper_and_expiry(self):
        sid = auth.create_session("h" * 64)
        ck = auth.make_cookie(sid)
        assert auth.parse_cookie(ck[:-2] + "zz") is None
        past = int(time.time()) - 10
        payload = f"{sid}.{past}"
        assert auth.parse_cookie(f"{payload}.{auth._sign(payload)}") is None
        assert auth.parse_cookie("garbage") is None


class TestTenants:
    def _sess(self):
        return auth.load_session(auth.create_session("h" * 64))

    def test_keys_roundtrip_and_at_rest(self):
        sess = self._sess()
        tenants.update_keys(sess, {"gemini_api_key": "AIza-secret-123", "s2_api_key": ""})
        assert tenants.get_key(sess, "gemini_api_key") == "AIza-secret-123"
        assert tenants.key_presence(sess)["gemini_api_key"] is True
        assert tenants.key_presence(sess)["s2_api_key"] is False
        on_disk = paths.session_file(sess["sid"]).read_text()
        if tenants._HAVE_AES:
            assert "AIza-secret-123" not in on_disk
        # reload from disk sees the same value
        again = auth.load_session(sess["sid"])
        assert tenants.get_key(again, "gemini_api_key") == "AIza-secret-123"

    def test_key_overrides_shape(self):
        sess = self._sess()
        tenants.update_keys(sess, {"gemini_api_key": "g", "openai_api_key": "o"})
        assert tenants.key_overrides(sess) == {"gemini_api_key": "g", "openai_api_key": "o"}

    def test_settings_clamped(self):
        sess = self._sess()
        tenants.update_settings(sess, {"max_papers": 10_000, "model": "m"})
        assert tenants.get_settings(sess)["max_papers"] == limits.MAX_PAPERS_CEILING
        assert tenants.get_settings(sess)["model"] == "m"

    def test_daily_quota(self):
        sess = self._sess()
        for _ in range(limits.RUNS_PER_DAY):
            assert tenants.can_start_run(sess) is None
            tenants.note_run_started(sess)
        assert "Daily limit" in tenants.can_start_run(sess)
        # a stale day rolls the counter over
        sess["usage"]["day"] = "2000-01-01"
        assert tenants.can_start_run(sess) is None


class TestJoinRateLimit:
    def test_window(self):
        ip = "1.2.3.4"
        for _ in range(limits.JOIN_ATTEMPTS):
            assert auth.join_allowed(ip)
        assert not auth.join_allowed(ip)
        assert auth.join_allowed("5.6.7.8")


class TestSessionFileShape:
    def test_created_session_json(self):
        sid = auth.create_session("c" * 64)
        data = json.loads(paths.session_file(sid).read_text())
        assert data["sid"] == sid and data["code_hash"] == "c" * 64
        assert paths.session_runs_dir(sid).is_dir()
