import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import nanobot.dashboard.app as dashboard_app
from nanobot.dashboard.app import _current_context_tokens


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


def test_current_context_tokens_does_not_fallback_to_cumulative_summary() -> None:
    assert _current_context_tokens(
        {},
        {"total_input_tokens": 450000},
        200000,
    ) == 0

    assert _current_context_tokens(
        {
            "usage_snapshot": {
                "total_input_tokens": 32000,
                "source": "provider_usage",
            }
        },
        {"total_input_tokens": 450000},
        200000,
    ) == 32000


def test_dashboard_can_create_agent_via_api(dashboard_client, monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake_home, workspace = dashboard_client

    config = json.loads((fake_home / '.nanobot' / 'config.json').read_text(encoding='utf-8'))
    config['channels']['discord']['enabled'] = True
    (fake_home / '.nanobot' / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    dashboard_app._config_cache = None

    monkeypatch.setattr(dashboard_app, '_discord_channel_category_id', lambda: 'cat-1')

    async def _fake_create_channel(self, guild_id, name, topic=None, category_id=None, channel_type=0):
        assert guild_id == 'guild-123'
        assert category_id == 'cat-1'
        assert topic == 'anthropic/claude-opus-4-5'
        return 'chan-created'

    async def _fake_create_webhook(self, channel_id, name='Agent', avatar_url=None):
        assert channel_id == 'chan-created'
        return 'https://discord.test/webhooks/chan-created/token'

    monkeypatch.setattr('nanobot.channels.discord.DiscordChannel.create_guild_channel', _fake_create_channel)
    monkeypatch.setattr('nanobot.channels.discord.DiscordChannel.create_channel_webhook', _fake_create_webhook)

    response = client.post('/api/agents', json={
        'mode': 'create',
        'agent_id': 'helper',
        'display_name': 'Helper',
        'channel_name': 'helper',
        'system_identity': 'You help.',
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'created'
    assert payload['restart_required'] is True
    assert payload['agent']['agent_id'] == 'helper'
    assert payload['agent']['webhook_enabled'] is True
    assert payload['channel']['id'] == 'chan-created'
    assert payload['channel']['topic'] == 'anthropic/claude-opus-4-5'

    effective = dashboard_app._load_config()
    assert 'helper' in effective['agents']['profiles']
    helper = effective['agents']['profiles']['helper']
    assert helper['displayName'] == 'Helper'
    assert helper['discordChannels'] == ['chan-created']
    assert helper['discordWebhookUrl'] == 'https://discord.test/webhooks/chan-created/token'
    assert (workspace / 'agents' / 'helper' / 'IDENTITY.md').read_text(encoding='utf-8') == 'You help.'


def test_dashboard_can_clone_agent_via_api(dashboard_client, monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake_home, workspace = dashboard_client

    src = workspace / 'agents' / 'source'
    (src / 'memory' / 'history').mkdir(parents=True, exist_ok=True)
    (src / 'skills' / 'local-skill').mkdir(parents=True, exist_ok=True)
    (src / 'sessions').mkdir(parents=True, exist_ok=True)
    (src / 'memory' / 'MEMORY.md').write_text('source memory', encoding='utf-8')
    (src / 'memory' / 'history' / '2026-03-09.md').write_text('history entry', encoding='utf-8')
    (src / 'skills' / 'local-skill' / 'SKILL.md').write_text('skill body', encoding='utf-8')
    (src / 'sessions' / 'source.jsonl').write_text('{"role":"user"}\n', encoding='utf-8')
    (src / 'sessions' / 'source.context.jsonl').write_text('ignored\n', encoding='utf-8')

    config = json.loads((fake_home / '.nanobot' / 'config.json').read_text(encoding='utf-8'))
    config['channels']['discord']['enabled'] = True
    config['agents']['profiles']['source'] = {
        'model': 'openai-codex/gpt-5.4',
        'backgroundModel': 'openai-codex/gpt-5.3-codex-spark',
        'contextWindow': 1000000,
        'backgroundContextWindow': 128000,
        'sessionStore': 'sqlite',
        'maxTokens': 100000,
        'temperature': 0.7,
        'maxToolIterations': 50,
        'reasoningEffort': 'high',
        'skills': ['memory', 'github'],
        'systemIdentity': 'source identity',
        'displayName': 'Source',
        'discordChannels': ['chan-source'],
    }
    (fake_home / '.nanobot' / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    dashboard_app._config_cache = None

    monkeypatch.setattr(dashboard_app, '_discord_channel_category_id', lambda: 'cat-1')

    async def _fake_create_channel(self, guild_id, name, topic=None, category_id=None, channel_type=0):
        assert topic == 'openai-codex/gpt-5.4'
        return 'chan-clone'

    async def _fake_create_webhook(self, channel_id, name='Agent', avatar_url=None):
        return 'https://discord.test/webhooks/chan-clone/token'

    monkeypatch.setattr('nanobot.channels.discord.DiscordChannel.create_guild_channel', _fake_create_channel)
    monkeypatch.setattr('nanobot.channels.discord.DiscordChannel.create_channel_webhook', _fake_create_webhook)

    response = client.post('/api/agents', json={
        'mode': 'clone',
        'source_agent_id': 'source',
        'agent_id': 'clone',
        'display_name': 'Clone Agent',
        'copy_history': True,
        'copy_sessions': True,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'cloned'
    assert payload['agent']['source_agent_id'] == 'source'
    assert payload['copied'] == {'memory': True, 'skills': True, 'history': True, 'sessions': True}
    clone = dashboard_app._load_config()['agents']['profiles']['clone']
    assert clone['model'] == 'openai-codex/gpt-5.4'
    assert clone['discordChannels'] == ['chan-clone']
    clone_ws = workspace / 'agents' / 'clone'
    assert (clone_ws / 'memory' / 'MEMORY.md').read_text(encoding='utf-8') == 'source memory'
    assert (clone_ws / 'memory' / 'history' / '2026-03-09.md').read_text(encoding='utf-8') == 'history entry'
    assert (clone_ws / 'skills' / 'local-skill' / 'SKILL.md').read_text(encoding='utf-8') == 'skill body'
    assert (clone_ws / 'sessions' / 'source.jsonl').exists()
    assert not (clone_ws / 'sessions' / 'source.context.jsonl').exists()
