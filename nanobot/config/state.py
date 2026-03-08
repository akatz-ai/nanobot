"""Runtime state storage for nanobot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.config.loader import _atomic_write_json
from nanobot.config.schema import AgentProfile, SCHEMA_VERSION, State


class StateStore:
    """Manage runtime state stored separately from base config."""

    def __init__(self, state_path: Path):
        self.path = state_path

    @classmethod
    def from_config_path(cls, config_path: Path) -> "StateStore":
        """Build a state store adjacent to a base config file."""
        return cls(config_path.parent / "state.json")

    def load(self) -> State:
        """Load runtime state from disk."""
        if not self.path.exists():
            return State()

        import json

        with open(self.path, encoding="utf-8") as handle:
            data = json.load(handle)
        return State.model_validate(_migrate_state(data))

    def save(self, state: State) -> None:
        """Persist runtime state to disk atomically."""
        data = state.model_dump(by_alias=True, exclude_none=True)
        data["schemaVersion"] = SCHEMA_VERSION
        _atomic_write_json(self.path, data)

    def overlay_config(self, base_data: dict[str, Any]) -> dict[str, Any]:
        """Overlay runtime state onto raw base config data."""
        state = self.load()
        state_data = state.model_dump(by_alias=True, exclude_none=True)
        merged = _deep_merge(dict(base_data), state_data)
        _apply_profile_tombstones(merged, state)
        merged.setdefault("agents", {}).pop("deletedProfiles", None)
        merged.pop("provisioning", None)
        _apply_discord_provisioning(merged, state)
        merged.setdefault("schemaVersion", SCHEMA_VERSION)
        return merged

    def upsert_profile(self, agent_id: str, profile: AgentProfile) -> None:
        """Persist an agent profile to runtime state."""
        state = self.load()
        state.agents.profiles[agent_id] = profile
        state.agents.deleted_profiles = [
            existing for existing in state.agents.deleted_profiles if existing != agent_id
        ]
        self.save(state)

    def delete_profile(self, agent_id: str) -> None:
        """Remove an agent profile from runtime state and add a tombstone."""
        state = self.load()
        state.agents.profiles.pop(agent_id, None)
        if agent_id not in state.agents.deleted_profiles:
            state.agents.deleted_profiles.append(agent_id)
        self.save(state)

    def set_discord_message_id(self, dashboard: str, message_id: str) -> None:
        """Persist a runtime Discord dashboard message ID."""
        state = self.load()
        dashboard_state = getattr(state.channels.discord, dashboard)
        dashboard_state.message_id = message_id
        self.save(state)

    def record_discord_setup(
        self,
        *,
        category_ids: dict[str, str],
        channel_ids: dict[str, str],
        webhook_urls: dict[str, str],
        checkpoints: dict[str, Any] | None = None,
    ) -> None:
        """Persist Discord provisioning results needed at runtime."""
        state = self.load()
        discord_state = state.provisioning.discord
        discord_state.category_ids = dict(category_ids)
        discord_state.channel_ids = dict(channel_ids)
        discord_state.webhook_urls = dict(webhook_urls)
        if checkpoints:
            state.provisioning.checkpoints.update(checkpoints)
        self.save(state)


def _migrate_state(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate old state formats to current."""
    migrated = dict(data)
    migrated.setdefault("schemaVersion", SCHEMA_VERSION)
    return migrated


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries, with overlay values winning."""
    for key, value in overlay.items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            base[key] = _deep_merge(dict(base_value), value)
        else:
            base[key] = value
    return base


def _apply_profile_tombstones(merged: dict[str, Any], state: State) -> None:
    """Remove base-config profiles that were deleted via runtime state."""
    profiles = merged.setdefault("agents", {}).setdefault("profiles", {})
    for agent_id in state.agents.deleted_profiles:
        profiles.pop(agent_id, None)


def _apply_discord_provisioning(merged: dict[str, Any], state: State) -> None:
    """Apply provisioned Discord layout data into the effective config."""
    discord = state.provisioning.discord
    if not (discord.channel_ids or discord.webhook_urls):
        return

    agents = merged.setdefault("agents", {}).setdefault("profiles", {})
    general_profile = agents.setdefault("general", {})

    general_channel_id = discord.channel_ids.get("general")
    if general_channel_id:
        general_profile["discordChannels"] = [general_channel_id]

    general_webhook = discord.webhook_urls.get("general")
    if general_webhook:
        general_profile["discordWebhookUrl"] = general_webhook

    discord_config = merged.setdefault("channels", {}).setdefault("discord", {})

    usage_channel_id = discord.channel_ids.get("claude-usage")
    if usage_channel_id:
        usage_cfg = discord_config.setdefault("usageDashboard", {})
        usage_cfg.setdefault("enabled", True)
        usage_cfg.setdefault("pollIntervalS", 600)
        usage_cfg["channelId"] = usage_channel_id

    status_channel_id = discord.channel_ids.get("system-status")
    if status_channel_id:
        status_cfg = discord_config.setdefault("systemStatus", {})
        status_cfg.setdefault("enabled", True)
        status_cfg.setdefault("pollIntervalS", 60)
        status_cfg["channelId"] = status_channel_id
