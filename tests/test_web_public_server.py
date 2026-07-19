"""HTTP-level tests for the public server: gate, join, isolation, downloads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.public.backend import auth, paths
from web.public.backend import server as public_server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(paths, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.setattr(paths, "INVITES_FILE", tmp_path / "auth" / "invites.json")
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(paths, "VOLUME_CACHE", tmp_path / "volcache" / "cache.db")
    monkeypatch.setattr(paths, "LOCAL_ROOT", tmp_path / "local")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "local" / "cache.db")
    monkeypatch.setattr(auth, "_secret", "test-secret")
    monkeypatch.setattr(auth, "_join_attempts", {})
    # TestClient speaks plain http — a Secure cookie would never echo back.
    monkeypatch.setenv("CITECLAW_PUBLIC_INSECURE_COOKIE", "1")
    paths.ensure_layout()
    with TestClient(public_server.app) as c:
        yield c


def _join(client) -> str:
    (code,) = auth.mint_codes(1)
    r = client.post("/api/auth/join", json={"code": code})
    assert r.status_code == 200
    return code


class TestGate:
    def test_unauthed_index_is_gate(self, client):
        html = client.get("/").text
        assert "invite" in html.lower()
        assert "text/babel" not in html  # no app code leaks pre-auth

    def test_api_requires_session(self, client):
        assert client.get("/api/settings").status_code == 401
        assert client.get("/api/session/runs").status_code == 401
        assert client.get("/api/models").status_code == 401

    def test_join_bad_code(self, client):
        r = client.post("/api/auth/join", json={"code": "CC-0000-0000"})
        assert r.status_code == 403

    def test_join_then_app(self, client):
        _join(client)
        html = client.get("/").text
        assert "text/babel" in html and "__PUBLIC__" in html
        assert client.get("/api/settings").status_code == 200


class TestSessions:
    def test_keys_isolated_between_sessions(self, client):
        _join(client)
        client.post("/api/settings", json={"gemini_api_key": "user-a-key"})
        assert client.get("/api/settings").json()["keys"]["gemini_api_key"] is True
        # a second browser joins with its own code — must not see A's key
        client.cookies.clear()
        _join(client)
        assert client.get("/api/settings").json()["keys"]["gemini_api_key"] is False

    def test_max_papers_clamped(self, client):
        _join(client)
        r = client.post("/api/settings", json={"max_papers": 99999})
        assert r.json()["max_papers"] == 500

    def test_me_reports_quota(self, client):
        _join(client)
        me = client.get("/api/auth/me").json()
        assert me["authed"] is True and me["runs_today"] == 0


class TestRunsAndDownloads:
    def test_run_requires_llm_key(self, client):
        _join(client)
        r = client.post("/api/run", json={"pipeline": [], "seeds": [{"id": "x"}]})
        assert r.status_code == 400
        assert "key" in r.json()["detail"].lower()

    def test_download_unknown_run(self, client):
        _join(client)
        assert client.get("/api/download/abcdef123456/zip").status_code == 404
        assert client.get("/api/download/abcdef123456/collection").status_code == 404
        # a path-shaped run id never reaches the filesystem
        assert client.get("/api/download/../../etc/zip").status_code in (404, 400)

    def test_session_runs_lists_disk_artifacts(self, client):
        _join(client)
        sid = auth.parse_cookie(client.cookies.get(auth.COOKIE_NAME))
        rid = "a" * 12
        d = paths.run_dir(sid, rid)
        d.mkdir(parents=True)
        (d / "literature_collection.json").write_text(
            '{"summary": {"total_accepted": 3}, "papers": []}')
        rows = client.get("/api/session/runs").json()
        assert rows and rows[0]["run_id"] == rid
        assert rows[0]["papers"] == 3 and rows[0]["status"] == "finished"
        z = client.get(f"/api/download/{rid}/zip")
        assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
        c = client.get(f"/api/download/{rid}/collection")
        assert c.status_code == 200 and b"total_accepted" in c.content

    def test_other_sessions_runs_invisible(self, client):
        _join(client)
        sid_a = auth.parse_cookie(client.cookies.get(auth.COOKIE_NAME))
        rid = "b" * 12
        d = paths.run_dir(sid_a, rid)
        d.mkdir(parents=True)
        (d / "literature_collection.json").write_text('{"papers": []}')
        client.cookies.clear()
        _join(client)
        assert client.get("/api/session/runs").json() == []
        assert client.get(f"/api/download/{rid}/zip").status_code == 404


class TestHardening:
    def test_body_cap(self, client):
        _join(client)
        r = client.post("/api/settings", content=b"x" * 10,
                        headers={"Content-Length": str(10 * 1024 * 1024),
                                 "Content-Type": "application/json"})
        assert r.status_code == 413

    def test_security_headers(self, client):
        r = client.get("/")
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
