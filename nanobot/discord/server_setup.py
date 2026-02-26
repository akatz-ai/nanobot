"""Phase 1 Discord server provisioning helpers (basic layout only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from nanobot.bus.queue import MessageBus
from nanobot.channels.discord import DISCORD_API_BASE, DiscordChannel
from nanobot.config.schema import DiscordConfig

BASIC_CATEGORIES: tuple[str, ...] = ("AGENTS", "STATUS")
BASIC_TEXT_CHANNELS: dict[str, tuple[str, str]] = {
    "general": ("AGENTS", "General-purpose AI assistant"),
    "system-status": ("STATUS", "System notifications and health"),
    "claude-usage": ("STATUS", "Claude API usage dashboard widget"),
}
GENERAL_WEBHOOK_NAME = "nanobot-general"


@dataclass
class ServerSetupResult:
    """Result payload for Discord setup routines."""

    category_ids: dict[str, str]
    channel_ids: dict[str, str]
    webhook_urls: dict[str, str]


@dataclass
class LayoutValidation:
    """Validation report for the required basic layout."""

    category_ids: dict[str, str] = field(default_factory=dict)
    channel_ids: dict[str, str] = field(default_factory=dict)
    missing_categories: list[str] = field(default_factory=list)
    missing_channels: list[str] = field(default_factory=list)
    wrong_parent_channels: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Layout is OK if required channels exist. Wrong parent is cosmetic, not a failure."""
        return not self.missing_categories and not self.missing_channels


async def validate_discord_bot_access(bot_token: str, guild_id: str) -> dict[str, Any]:
    """Validate bot token and guild access. Returns bot and guild metadata."""
    headers = {"Authorization": f"Bot {bot_token}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        user_resp = await client.get(f"{DISCORD_API_BASE}/users/@me", headers=headers)
        if user_resp.status_code != 200:
            raise RuntimeError("Discord bot token is invalid or unauthorized")

        guild_resp = await client.get(f"{DISCORD_API_BASE}/guilds/{guild_id}", headers=headers)
        if guild_resp.status_code == 403:
            raise RuntimeError("Bot cannot access the guild (missing invite/permissions)")
        if guild_resp.status_code == 404:
            raise RuntimeError("Guild not found or bot not in guild")
        if guild_resp.status_code != 200:
            raise RuntimeError(f"Discord guild check failed ({guild_resp.status_code})")

        return {
            "bot": user_resp.json(),
            "guild": guild_resp.json(),
        }


async def list_guild_channels_for_setup(guild_id: str, bot_token: str) -> list[dict[str, Any]]:
    """List guild channels using existing DiscordChannel primitives."""
    discord = DiscordChannel(DiscordConfig(enabled=True, token=bot_token), MessageBus())
    channels = await discord.list_guild_channels(guild_id)
    return channels if isinstance(channels, list) else []


def validate_basic_layout(channels: list[dict[str, Any]]) -> LayoutValidation:
    """Check whether the guild contains the required Phase 1 layout."""
    report = LayoutValidation()

    categories_by_name: dict[str, dict[str, Any]] = {}
    text_by_name: dict[str, dict[str, Any]] = {}

    for ch in channels:
        name = str(ch.get("name") or "")
        if not name:
            continue
        key = name.casefold()
        if ch.get("type") == 4 and key not in categories_by_name:
            categories_by_name[key] = ch
        if ch.get("type") == 0 and key not in text_by_name:
            text_by_name[key] = ch

    for category_name in BASIC_CATEGORIES:
        cat = categories_by_name.get(category_name.casefold())
        if not cat:
            report.missing_categories.append(category_name)
            continue
        report.category_ids[category_name] = str(cat.get("id"))

    for channel_name, (expected_category, _topic) in BASIC_TEXT_CHANNELS.items():
        ch = text_by_name.get(channel_name.casefold())
        if not ch:
            report.missing_channels.append(channel_name)
            continue

        channel_id = str(ch.get("id"))
        report.channel_ids[channel_name] = channel_id

        # Track parent mismatch as informational but don't treat as failure —
        # the channel ID is what matters for routing, not its category.
        expected_parent = report.category_ids.get(expected_category)
        actual_parent = str(ch.get("parent_id") or "")
        if expected_parent and actual_parent != expected_parent:
            report.wrong_parent_channels.append(channel_name)

    return report


async def setup_basic_server(guild_id: str, bot_token: str) -> ServerSetupResult:
    """Create the Phase 1 basic layout idempotently and return IDs/webhooks."""
    await validate_discord_bot_access(bot_token, guild_id)

    discord = DiscordChannel(DiscordConfig(enabled=True, token=bot_token), MessageBus())
    channels = await discord.list_guild_channels(guild_id)

    categories_by_name: dict[str, dict[str, Any]] = {}
    text_by_name: dict[str, dict[str, Any]] = {}

    for ch in channels:
        name = str(ch.get("name") or "")
        if not name:
            continue
        key = name.casefold()
        if ch.get("type") == 4 and key not in categories_by_name:
            categories_by_name[key] = ch
        if ch.get("type") == 0 and key not in text_by_name:
            text_by_name[key] = ch

    category_ids: dict[str, str] = {}
    for category_name in BASIC_CATEGORIES:
        existing = categories_by_name.get(category_name.casefold())
        if existing:
            category_ids[category_name] = str(existing["id"])
            continue

        category_id = await discord.create_guild_channel(
            guild_id=guild_id,
            name=category_name,
            channel_type=4,
        )
        if not category_id:
            raise RuntimeError(f"Failed creating Discord category '{category_name}'")
        category_ids[category_name] = category_id

    channel_ids: dict[str, str] = {}
    for channel_name, (parent_category, topic) in BASIC_TEXT_CHANNELS.items():
        existing = text_by_name.get(channel_name.casefold())
        if existing:
            channel_id = str(existing["id"])
            channel_ids[channel_name] = channel_id

            # Try to move channel under the correct category if it's misplaced
            expected_parent = category_ids.get(parent_category)
            actual_parent = str(existing.get("parent_id") or "")
            if expected_parent and actual_parent != expected_parent:
                moved = await _try_move_channel(channel_id, expected_parent, bot_token)
                if moved:
                    from loguru import logger
                    logger.info("Moved #{} to category {}", channel_name, parent_category)
            continue

        channel_id = await discord.create_guild_channel(
            guild_id=guild_id,
            name=channel_name,
            topic=topic,
            category_id=category_ids[parent_category],
            channel_type=0,
        )
        if not channel_id:
            raise RuntimeError(f"Failed creating Discord channel '#{channel_name}'")
        channel_ids[channel_name] = channel_id

    general_id = channel_ids["general"]
    general_webhook = await _ensure_channel_webhook(
        channel_id=general_id,
        bot_token=bot_token,
        webhook_name=GENERAL_WEBHOOK_NAME,
        discord=discord,
    )

    return ServerSetupResult(
        category_ids=category_ids,
        channel_ids=channel_ids,
        webhook_urls={"general": general_webhook},
    )


async def _try_move_channel(channel_id: str, new_parent_id: str, bot_token: str) -> bool:
    """Attempt to move a channel to a different category. Returns True on success."""
    headers = {"Authorization": f"Bot {bot_token}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.patch(
                f"{DISCORD_API_BASE}/channels/{channel_id}",
                headers=headers,
                json={"parent_id": new_parent_id},
            )
            return resp.status_code == 200
    except Exception:
        return False


async def _ensure_channel_webhook(
    channel_id: str,
    bot_token: str,
    webhook_name: str,
    discord: DiscordChannel,
) -> str:
    """Return existing matching webhook or create one."""
    headers = {"Authorization": f"Bot {bot_token}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{DISCORD_API_BASE}/channels/{channel_id}/webhooks", headers=headers)
        if resp.status_code == 200:
            for hook in resp.json():
                if str(hook.get("name") or "") != webhook_name:
                    continue
                token = str(hook.get("token") or "")
                if token:
                    hook_id = str(hook["id"])
                    return f"{DISCORD_API_BASE}/webhooks/{hook_id}/{token}"

    webhook_url = await discord.create_channel_webhook(channel_id=channel_id, name=webhook_name)
    if not webhook_url:
        raise RuntimeError("Failed creating webhook for #general")
    return webhook_url
