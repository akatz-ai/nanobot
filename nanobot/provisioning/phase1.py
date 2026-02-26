"""Phase 1 provisioning config and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nanobot.config.loader import get_config_path
from nanobot.config.schema import AgentProfile, Config
from nanobot.discord.server_setup import (
    BASIC_TEXT_CHANNELS,
    list_guild_channels_for_setup,
    validate_basic_layout,
    validate_discord_bot_access,
)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


@dataclass
class ProvisionCheck:
    """Single validation check result."""

    name: str
    ok: bool
    detail: str


def normalize_anthropic_model(model: str) -> str:
    """Normalize user input into an anthropic-direct model string."""
    value = (model or "").strip()
    if not value:
        value = DEFAULT_ANTHROPIC_MODEL
    if "/" in value:
        return value
    return f"anthropic-direct/{value}"


def build_basic_config(
    *,
    bot_token: str,
    guild_id: str,
    model: str,
    channel_ids: dict[str, str],
    webhook_urls: dict[str, str],
) -> Config:
    """Generate a complete config for Phase 1 basic provisioning."""
    config = Config()

    resolved_model = normalize_anthropic_model(model)

    config.agents.defaults.model = resolved_model
    config.agents.defaults.max_tool_iterations = 40
    config.agents.defaults.memory_window = 50

    config.agents.profiles = {
        "general": AgentProfile(
            model=None,
            max_tokens=None,
            temperature=None,
            max_tool_iterations=None,
            skills=None,
            system_identity=(
                "You are a helpful general-purpose AI assistant for this Discord server. "
                "Be concise, accurate, and practical."
            ),
            discord_channels=[channel_ids["general"]],
            display_name="General",
            avatar_url=None,
            discord_webhook_url=webhook_urls.get("general"),
        )
    }

    config.channels.discord.enabled = True
    config.channels.discord.token = bot_token
    config.channels.discord.guild_id = guild_id

    config.channels.discord.usage_dashboard.enabled = True
    config.channels.discord.usage_dashboard.channel_id = channel_ids["claude-usage"]
    config.channels.discord.usage_dashboard.message_id = ""
    config.channels.discord.usage_dashboard.poll_interval_s = 600

    config.channels.discord.system_status.enabled = True
    config.channels.discord.system_status.channel_id = channel_ids["system-status"]
    config.channels.discord.system_status.message_id = ""
    config.channels.discord.system_status.poll_interval_s = 60

    config.providers.anthropic_direct.enabled = True
    config.providers.anthropic_direct.model = resolved_model.split("/", 1)[1] if "/" in resolved_model else resolved_model

    return config


def validate_config_file(config_path: Path | None = None) -> tuple[Config, dict]:
    """Load and strictly validate config.json from disk."""
    path = config_path or get_config_path()
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Config JSON parse failed: {e}") from e

    try:
        config = Config.model_validate(raw)
    except Exception as e:
        raise RuntimeError(f"Config schema validation failed: {e}") from e

    return config, raw


def check_claude_credentials_file() -> tuple[bool, str]:
    """Check ~/.claude/.credentials.json for a Claude OAuth access token."""
    credentials_path = Path.home() / ".claude" / ".credentials.json"
    if not credentials_path.exists():
        return False, f"Missing credentials file at {credentials_path}"

    try:
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"Credentials file is not valid JSON: {e}"

    token = (
        data.get("claudeAiOauth", {}).get("accessToken", "")
        if isinstance(data, dict)
        else ""
    )
    if not token:
        return False, "Credentials file exists but access token is missing"

    return True, "Claude OAuth credentials present"


async def run_phase1_checks(
    *,
    bot_token: str,
    guild_id: str,
    config_path: Path | None = None,
) -> list[ProvisionCheck]:
    """Run Phase 1 validation checks and return detailed results."""
    results: list[ProvisionCheck] = []

    # Config parse + schema validation.
    try:
        config, _raw = validate_config_file(config_path)
        results.append(ProvisionCheck("config", True, "config.json parsed and validated"))
    except RuntimeError as e:
        results.append(ProvisionCheck("config", False, str(e)))
        return results

    # Validate Discord token and guild access.
    try:
        meta = await validate_discord_bot_access(bot_token, guild_id)
        bot_user = str(meta.get("bot", {}).get("username") or "unknown")
        guild_name = str(meta.get("guild", {}).get("name") or guild_id)
        results.append(
            ProvisionCheck(
                "discord-access",
                True,
                f"Bot authenticated as '{bot_user}', guild '{guild_name}' reachable",
            )
        )
    except RuntimeError as e:
        results.append(ProvisionCheck("discord-access", False, str(e)))
        return results

    channels = await list_guild_channels_for_setup(guild_id, bot_token)
    layout = validate_basic_layout(channels)

    if layout.ok:
        results.append(ProvisionCheck("discord-layout", True, "Required categories/channels exist"))
    else:
        detail_parts: list[str] = []
        if layout.missing_categories:
            detail_parts.append(f"missing categories: {', '.join(layout.missing_categories)}")
        if layout.missing_channels:
            detail_parts.append(f"missing channels: {', '.join(layout.missing_channels)}")
        if layout.wrong_parent_channels:
            detail_parts.append(
                f"wrong parent category: {', '.join(layout.wrong_parent_channels)}"
            )
        results.append(ProvisionCheck("discord-layout", False, "; ".join(detail_parts)))

    # Ensure config maps the general profile to a known channel.
    general = config.agents.profiles.get("general")
    if not general or not general.discord_channels:
        results.append(
            ProvisionCheck("general-profile", False, "general agent missing Discord channel mapping")
        )
    else:
        known_ids = set(layout.channel_ids.values())
        mapped = set(general.discord_channels)
        if known_ids and mapped.intersection(known_ids):
            results.append(
                ProvisionCheck("general-profile", True, "general agent mapped to provisioned channel")
            )
        else:
            results.append(
                ProvisionCheck(
                    "general-profile",
                    False,
                    "general agent channel mapping does not match provisioned layout",
                )
            )

    creds_ok, creds_detail = check_claude_credentials_file()
    results.append(ProvisionCheck("claude-auth", creds_ok, creds_detail))

    required_usage_channel = layout.channel_ids.get("claude-usage")
    configured_usage_channel = config.channels.discord.usage_dashboard.channel_id
    if required_usage_channel and configured_usage_channel == required_usage_channel:
        results.append(
            ProvisionCheck("usage-dashboard", True, "Usage dashboard channel matches layout")
        )
    else:
        results.append(
            ProvisionCheck(
                "usage-dashboard",
                False,
                "Usage dashboard channel is missing or mismatched",
            )
        )

    return results


def summarize_expected_layout() -> str:
    """Return a short text summary of required channels."""
    parts = []
    for name, (category, _topic) in BASIC_TEXT_CHANNELS.items():
        parts.append(f"{category}/#{name}")
    return ", ".join(parts)
