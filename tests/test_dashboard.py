import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import nanobot.dashboard.app as dashboard_app


@pytest.fixture
def dashboard_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config_dir = fake_home / ".nanobot"
    config_dir.mkdir()
    config = {
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "model": "anthropic/claude-opus-4-5",
            },
            "profiles": {},
        },
        "tools": {
            "mcpServers": {
                "filesystem": {
                    "command": "python",
                    "args": ["-m", "http.server"],
                    "env": {"ROOT_DIR": "/tmp"},
                },
                "google-drive": {
                    "command": "npx",
                    "env": {
                        "GOOGLE_CLIENT_ID": "google-client-id",
                        "GOOGLE_CLIENT_SECRET": "google-client-secret",
                    },
                },
            },
        },
        "channels": {
            "telegram": {
                "enabled": True,
                "token": "telegram-token",
                "allowFrom": ["alice", "bob"],
            },
            "discord": {
                "enabled": False,
                "token": "discord-token",
                "guildId": "guild-123",
            },
        },
        "providers": {
            "openrouter": {"apiKey": "sk-or-test"},
            "anthropic": {"apiKey": ""},
        },
    }
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    dashboard_app._config_cache = None
    dashboard_app._config_mtime = 0
    dashboard_app.app._oauth_states = {}

    with TestClient(dashboard_app.app) as client:
        yield client, fake_home, workspace

    dashboard_app._config_cache = None
    dashboard_app._config_mtime = 0
    dashboard_app.app._oauth_states = {}


def test_get_integrations_returns_mcp_channels_and_providers(dashboard_client) -> None:
    client, _fake_home, _workspace = dashboard_client

    response = client.get("/api/integrations")

    assert response.status_code == 200
    payload = response.json()
    integrations = payload["integrations"]
    by_id = {item["id"]: item for item in integrations}

    assert by_id["filesystem"]["type"] == "mcp"
    assert by_id["filesystem"]["status"] == "configured"
    assert by_id["filesystem"]["command"] == "python"
    assert by_id["filesystem"]["has_env"] is True

    assert by_id["google-drive"]["type"] == "mcp"
    assert by_id["google-drive"]["name"] == "Google Drive"
    assert by_id["google-drive"]["oauth_supported"] is True
    assert by_id["google-drive"]["status"] == "error"

    assert by_id["telegram"]["type"] == "channel"
    assert by_id["telegram"]["status"] == "connected"
    assert by_id["telegram"]["bot_token_set"] is True
    assert by_id["telegram"]["allowed_users"] == 2

    assert by_id["discord"]["type"] == "channel"
    assert by_id["discord"]["status"] == "disabled"
    assert by_id["discord"]["guild_id"] == "guild-123"

    assert by_id["provider-openrouter"]["type"] == "provider"
    assert by_id["provider-openrouter"]["status"] == "configured"
    assert by_id["provider-openrouter"]["has_api_key"] is True

    # Providers without API keys are excluded from the list
    assert "provider-anthropic" not in by_id


def test_check_integration_returns_status_for_known_and_unknown_integrations(dashboard_client) -> None:
    client, fake_home, workspace = dashboard_client

    known = client.post("/api/integrations/filesystem/check")
    unknown = client.post("/api/integrations/not-real/check")
    google_before = client.post("/api/integrations/google-drive/check")

    creds_dir = workspace.parent / "credentials"
    creds_dir.mkdir(parents=True, exist_ok=True)
    (creds_dir / "google-drive.json").write_text(
        json.dumps({"token": "access", "refresh_token": "refresh"}),
        encoding="utf-8",
    )
    google_after = client.post("/api/integrations/google-drive/check")

    claude_dir = fake_home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"subscriptionType": "pro", "expiresAt": 0}}),
        encoding="utf-8",
    )
    anthropic = client.post("/api/integrations/provider-anthropic-oauth/check")

    assert known.status_code == 200
    assert known.json()["status"] == "configured"
    assert "checked_at" in known.json()

    assert unknown.status_code == 200
    assert unknown.json() == {
        "id": "not-real",
        "status": "unknown",
        "error": "Integration not found",
    }

    assert google_before.status_code == 200
    assert google_before.json()["status"] == "error"
    assert google_before.json()["error"] == "Authorization required"

    assert google_after.status_code == 200
    assert google_after.json()["status"] == "connected"
    assert google_after.json()["has_refresh_token"] is True

    assert anthropic.status_code == 200
    assert anthropic.json()["status"] == "connected"
    assert anthropic.json()["subscription"] == "pro"


def test_integration_auth_returns_auth_url_with_expected_parameters(dashboard_client) -> None:
    client, _fake_home, _workspace = dashboard_client

    response = client.get(
        "/api/integrations/google-drive/auth",
        params={"callback_base": "https://dashboard.example/api"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["callback_url"] == "https://dashboard.example/api/integrations/google-drive/callback"
    assert payload["state"]

    parsed = urlparse(payload["auth_url"])
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "accounts.google.com"
    assert parsed.path == "/o/oauth2/auth"
    assert query["client_id"] == ["google-client-id"]
    assert query["redirect_uri"] == [payload["callback_url"]]
    assert query["response_type"] == ["code"]
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert query["state"] == [payload["state"]]
    assert "https://www.googleapis.com/auth/drive" in query["scope"][0]


def test_integration_callback_handles_error_and_missing_code(dashboard_client) -> None:
    client, _fake_home, _workspace = dashboard_client

    error_response = client.get(
        "/api/integrations/google-drive/callback",
        params={"error": "access_denied"},
    )
    missing_code = client.get("/api/integrations/google-drive/callback")

    assert error_response.status_code == 200
    assert "Authorization Failed" in error_response.text
    assert "access_denied" in error_response.text

    assert missing_code.status_code == 400
    assert missing_code.json()["detail"] == "No authorization code received"
