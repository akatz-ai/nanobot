import json
from pathlib import Path
from types import SimpleNamespace

from nanobot.agent.profile_manager import AgentProfileManager
from nanobot.config.loader import load_config, load_config_data
from nanobot.config.schema import AgentProfile
from nanobot.config.state import StateStore
from nanobot.discord.system_status import SystemStatusDashboard
from nanobot.discord.usage_dashboard import UsageDashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_config_overlays_runtime_state_and_tombstones(tmp_path: Path) -> None:
    config_path = tmp_path / "config.base.json"
    state_path = tmp_path / "state.json"

    _write_json(
        config_path,
        {
            "schemaVersion": 1,
            "agents": {
                "defaults": {"workspace": str(tmp_path / "workspace")},
                "profiles": {
                    "general": {"systemIdentity": "Base general"},
                    "legacy": {"systemIdentity": "Remove me"},
                },
            },
            "channels": {"discord": {"enabled": True, "guildId": "guild-1"}},
        },
    )
    _write_json(
        state_path,
        {
            "schemaVersion": 1,
            "agents": {
                "profiles": {
                    "helper": {
                        "systemIdentity": "Runtime helper",
                        "discordChannels": ["chan-helper"],
                    }
                },
                "deletedProfiles": ["legacy"],
            },
            "channels": {
                "discord": {
                    "usageDashboard": {"messageId": "msg-usage"},
                    "systemStatus": {"messageId": "msg-status"},
                }
            },
            "provisioning": {
                "discord": {
                    "channelIds": {
                        "general": "chan-general",
                        "claude-usage": "chan-usage",
                        "system-status": "chan-status",
                    },
                    "webhookUrls": {"general": "https://discord.test/webhook"},
                }
            },
        },
    )

    merged = load_config_data(config_path=config_path, state_path=state_path)

    profiles = merged["agents"]["profiles"]
    assert "helper" in profiles
    assert "legacy" not in profiles
    assert profiles["general"]["discordChannels"] == ["chan-general"]
    assert profiles["general"]["discordWebhookUrl"] == "https://discord.test/webhook"
    assert merged["channels"]["discord"]["usageDashboard"]["channelId"] == "chan-usage"
    assert merged["channels"]["discord"]["usageDashboard"]["messageId"] == "msg-usage"
    assert merged["channels"]["discord"]["systemStatus"]["channelId"] == "chan-status"
    assert merged["channels"]["discord"]["systemStatus"]["messageId"] == "msg-status"

    config = load_config(config_path=config_path)
    assert "helper" in config.agents.profiles
    assert "legacy" not in config.agents.profiles
    assert config.channels.discord.usage_dashboard.message_id == "msg-usage"


def test_agent_profile_manager_persists_runtime_state_without_mutating_base(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "schemaVersion": 1,
            "agents": {
                "defaults": {"workspace": str(tmp_path / "workspace")},
                "profiles": {
                    "general": {"systemIdentity": "Base general"},
                    "legacy": {"systemIdentity": "Legacy profile"},
                },
            },
        },
    )

    config = load_config(config_path=config_path)
    state_store = StateStore.from_config_path(config_path)
    manager = AgentProfileManager(config, state_store)

    manager.create_profile("helper", system_identity="Runtime helper", display_name="Helper")

    base_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "helper" not in base_data["agents"]["profiles"]

    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["agents"]["profiles"]["helper"]["displayName"] == "Helper"

    manager.delete_profile("legacy")
    reloaded = load_config(config_path=config_path)
    assert "legacy" not in reloaded.agents.profiles


def test_state_store_atomic_runtime_writes_include_schema_version(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.upsert_profile("helper", AgentProfile(system_identity="Atomic helper"))
    store.set_discord_message_id("usage_dashboard", "usage-123")

    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["schemaVersion"] == 1
    assert state_data["agents"]["profiles"]["helper"]["systemIdentity"] == "Atomic helper"
    assert state_data["channels"]["discord"]["usageDashboard"]["messageId"] == "usage-123"


def test_dashboards_persist_message_ids_to_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"schemaVersion": 1})

    usage = UsageDashboard(
        anthropic_token="token",
        discord_token="discord",
        channel_id="chan-1",
        config_path=str(config_path),
    )
    usage.message_id = "usage-msg"
    usage._persist_message_id()

    system = SystemStatusDashboard(
        router=SimpleNamespace(),
        discord_token="discord",
        channel_id="chan-2",
        config_path=str(config_path),
    )
    system.message_id = "status-msg"
    system._persist_message_id()

    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["channels"]["discord"]["usageDashboard"]["messageId"] == "usage-msg"
    assert state_data["channels"]["discord"]["systemStatus"]["messageId"] == "status-msg"
