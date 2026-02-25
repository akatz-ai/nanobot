"""Discord usage dashboard — displays Claude API rate limit utilization.

Polls the Anthropic Messages API with a minimal request (max_tokens=1) to read
rate-limit utilization headers, then renders a Components V2 message in Discord
with progress bars that auto-update on a configurable interval.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

# ── Anthropic API ──────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "oauth-2025-04-20"
PROBE_MODEL = "claude-sonnet-4-20250514"

# ── Discord API ────────────────────────────────────────────────────────────

DISCORD_API_BASE = "https://discord.com/api/v10"
COMPONENTS_V2_FLAG = 1 << 15  # 32768 — IS_COMPONENTS_V2

# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_POLL_INTERVAL_S = 600  # 10 minutes
BAR_WIDTH = 20  # characters for progress bar

# ── Visual ─────────────────────────────────────────────────────────────────

ACCENT_COLOR_NORMAL = 0x585CF2  # blurple-ish
ACCENT_COLOR_WARNING = 0xFFA500  # orange
ACCENT_COLOR_CRITICAL = 0xFF4444  # red


@dataclass
class UsageData:
    """Parsed rate-limit utilization from Anthropic response headers."""

    utilization_5h: float = 0.0
    utilization_7d: float = 0.0
    utilization_7d_sonnet: float = 0.0
    utilization_overage: float = 0.0

    reset_5h: int = 0  # unix timestamp
    reset_7d: int = 0
    reset_7d_sonnet: int = 0
    reset_overage: int = 0

    status_5h: str = "unknown"
    status_7d: str = "unknown"
    status_7d_sonnet: str = "unknown"
    status_overage: str = "unknown"

    representative_claim: str = ""
    fallback_percentage: float = 0.0

    polled_at: float = 0.0  # time.time() when data was fetched

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> "UsageData":
        """Extract usage data from Anthropic response headers."""

        def _float(key: str) -> float:
            try:
                return float(headers.get(key, "0"))
            except (ValueError, TypeError):
                return 0.0

        def _int(key: str) -> int:
            try:
                return int(headers.get(key, "0"))
            except (ValueError, TypeError):
                return 0

        def _str(key: str) -> str:
            return headers.get(key, "unknown")

        return cls(
            utilization_5h=_float("anthropic-ratelimit-unified-5h-utilization"),
            utilization_7d=_float("anthropic-ratelimit-unified-7d-utilization"),
            utilization_7d_sonnet=_float("anthropic-ratelimit-unified-7d_sonnet-utilization"),
            utilization_overage=_float("anthropic-ratelimit-unified-overage-utilization"),
            reset_5h=_int("anthropic-ratelimit-unified-5h-reset"),
            reset_7d=_int("anthropic-ratelimit-unified-7d-reset"),
            reset_7d_sonnet=_int("anthropic-ratelimit-unified-7d_sonnet-reset"),
            reset_overage=_int("anthropic-ratelimit-unified-overage-reset"),
            status_5h=_str("anthropic-ratelimit-unified-5h-status"),
            status_7d=_str("anthropic-ratelimit-unified-7d-status"),
            status_7d_sonnet=_str("anthropic-ratelimit-unified-7d_sonnet-status"),
            status_overage=_str("anthropic-ratelimit-unified-overage-status"),
            representative_claim=_str("anthropic-ratelimit-unified-representative-claim"),
            fallback_percentage=_float("anthropic-ratelimit-unified-fallback-percentage"),
            polled_at=time.time(),
        )


def _progress_bar(pct: float, width: int = BAR_WIDTH) -> str:
    """Render a Unicode progress bar. pct is 0.0–1.0."""
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _status_emoji(status: str, utilization: float) -> str:
    """Pick an emoji based on status and utilization level."""
    if status == "limited":
        return "🔴"
    if utilization >= 0.80:
        return "🟡"
    return "🟢"


def _accent_color(data: UsageData) -> int:
    """Pick container accent color based on overall health."""
    max_util = max(data.utilization_5h, data.utilization_7d)
    if max_util >= 0.90 or data.status_5h == "limited" or data.status_7d == "limited":
        return ACCENT_COLOR_CRITICAL
    if max_util >= 0.70:
        return ACCENT_COLOR_WARNING
    return ACCENT_COLOR_NORMAL


def render_dashboard(data: UsageData) -> list[dict[str, Any]]:
    """Build Discord Components V2 payload from usage data.

    Returns the `components` list for a Components V2 message.
    """
    meters = [
        ("5h Window", data.utilization_5h, data.reset_5h, data.status_5h),
        ("7d Window", data.utilization_7d, data.reset_7d, data.status_7d),
        ("7d Sonnet", data.utilization_7d_sonnet, data.reset_7d_sonnet, data.status_7d_sonnet),
        ("Overage", data.utilization_overage, data.reset_overage, data.status_overage),
    ]

    lines: list[str] = ["## 📊 Claude Usage"]
    for label, util, reset_ts, status in meters:
        emoji = _status_emoji(status, util)
        bar = _progress_bar(util)
        pct_str = f"{util * 100:5.1f}%"
        reset_str = f"<t:{reset_ts}:R>" if reset_ts else ""
        lines.append(f"{emoji} **{label}**  `{bar}`  **{pct_str}**  resets {reset_str}")

    # Footer
    claim = data.representative_claim.replace("_", " ") if data.representative_claim else "—"
    lines.append("")
    lines.append(f"-# Active claim: {claim} · Updated <t:{int(data.polled_at)}:R>")

    content = "\n".join(lines)
    color = _accent_color(data)

    return [
        {
            "type": 17,  # Container
            "accent_color": color,
            "components": [
                {
                    "type": 10,  # Text Display
                    "content": content,
                }
            ],
        }
    ]


class UsageDashboard:
    """Async service that polls Anthropic and updates a Discord message.

    Usage::

        dashboard = UsageDashboard(
            anthropic_token="sk-ant-oat...",
            discord_token="Bot ...",
            channel_id="123456789",
            poll_interval_s=600,
        )
        await dashboard.start()   # runs forever in background
        ...
        dashboard.stop()
    """

    def __init__(
        self,
        anthropic_token: str,
        discord_token: str,
        channel_id: str,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        message_id: str | None = None,
        config_path: str | None = None,
    ):
        self.anthropic_token = anthropic_token
        self.discord_token = discord_token
        self.channel_id = channel_id
        self.poll_interval_s = poll_interval_s
        self.message_id = message_id  # persisted after first create
        self.config_path = config_path  # path to config.json for persistence

        self._running = False
        self._task: asyncio.Task | None = None
        self._http: httpx.AsyncClient | None = None
        self._last_data: UsageData | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the dashboard polling loop."""
        if self._running:
            logger.warning("UsageDashboard already running")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "UsageDashboard started (channel={}, interval={}s)",
            self.channel_id,
            self.poll_interval_s,
        )

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("UsageDashboard stopped")

    async def close(self) -> None:
        """Clean up HTTP client."""
        self.stop()
        if self._http:
            await self._http.aclose()
            self._http = None

    # ── Main loop ──────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Poll → render → update, repeat."""
        # Do an immediate first update
        await self._tick()

        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("UsageDashboard tick error")

    async def _tick(self) -> None:
        """Single poll + update cycle."""
        data = await self._poll_anthropic()
        if data is None:
            return

        self._last_data = data
        components = render_dashboard(data)

        if self.message_id:
            await self._edit_message(components)
        else:
            await self._create_message(components)

    # ── Anthropic polling ──────────────────────────────────────────────

    async def _poll_anthropic(self) -> UsageData | None:
        """Send a minimal API request and extract rate-limit headers."""
        if not self._http:
            return None

        try:
            resp = await self._http.post(
                ANTHROPIC_API_URL,
                headers={
                    "Authorization": f"Bearer {self.anthropic_token}",
                    "anthropic-version": ANTHROPIC_VERSION,
                    "anthropic-beta": ANTHROPIC_BETA,
                    "Content-Type": "application/json",
                },
                json={
                    "model": PROBE_MODEL,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "."}],
                },
            )

            if resp.status_code == 401:
                logger.warning("UsageDashboard: Anthropic auth failed (401), refreshing token...")
                new_token = await self._refresh_anthropic_token()
                if new_token:
                    self.anthropic_token = new_token
                    return await self._poll_anthropic()
                logger.error("UsageDashboard: Token refresh failed")
                return None

            if resp.status_code == 429:
                logger.warning("UsageDashboard: Rate limited (429), will retry next interval")
                # Still parse headers — they're present even on 429
                return UsageData.from_headers(resp.headers)

            if resp.status_code != 200:
                logger.warning(
                    "UsageDashboard: Anthropic returned {}: {}",
                    resp.status_code,
                    resp.text[:200],
                )
                return None

            return UsageData.from_headers(resp.headers)

        except Exception:
            logger.exception("UsageDashboard: Failed to poll Anthropic")
            return None

    async def _refresh_anthropic_token(self) -> str | None:
        """Try to refresh the OAuth token."""
        try:
            from nanobot.providers.anthropic_auth import get_oauth_token

            token = get_oauth_token()
            if token:
                logger.info("UsageDashboard: Got refreshed OAuth token")
                return token
        except Exception:
            logger.exception("UsageDashboard: Token refresh error")
        return None

    # ── Discord messaging ──────────────────────────────────────────────

    async def _create_message(self, components: list[dict[str, Any]]) -> None:
        """Create the initial Components V2 message."""
        if not self._http:
            return

        url = f"{DISCORD_API_BASE}/channels/{self.channel_id}/messages"
        payload = {"flags": COMPONENTS_V2_FLAG, "components": components}

        try:
            resp = await self._http.post(
                url,
                headers={"Authorization": f"Bot {self.discord_token}"},
                json=payload,
            )

            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                logger.warning("Discord rate limited, retrying in {}s", retry_after)
                await asyncio.sleep(retry_after)
                return await self._create_message(components)

            resp.raise_for_status()
            data = resp.json()
            self.message_id = data["id"]
            logger.info("UsageDashboard: Created message {}", self.message_id)
            self._persist_message_id()

        except Exception:
            logger.exception("UsageDashboard: Failed to create Discord message")

    async def _edit_message(self, components: list[dict[str, Any]]) -> None:
        """Edit the existing dashboard message."""
        if not self._http or not self.message_id:
            return

        url = f"{DISCORD_API_BASE}/channels/{self.channel_id}/messages/{self.message_id}"
        payload = {"flags": COMPONENTS_V2_FLAG, "components": components}

        try:
            resp = await self._http.patch(
                url,
                headers={"Authorization": f"Bot {self.discord_token}"},
                json=payload,
            )

            if resp.status_code == 404:
                # Message was deleted — recreate it
                logger.warning("UsageDashboard: Message deleted, recreating...")
                self.message_id = None
                return await self._create_message(components)

            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                logger.warning("Discord rate limited, retrying in {}s", retry_after)
                await asyncio.sleep(retry_after)
                return await self._edit_message(components)

            resp.raise_for_status()
            logger.debug("UsageDashboard: Updated message {}", self.message_id)

        except Exception:
            logger.exception("UsageDashboard: Failed to edit Discord message")

    # ── Config persistence ──────────────────────────────────────────────

    def _persist_message_id(self) -> None:
        """Save the current message_id back to config.json so restarts reuse it."""
        if not self.config_path or not self.message_id:
            return
        try:
            import json
            from pathlib import Path

            path = Path(self.config_path)
            if not path.exists():
                return

            config = json.loads(path.read_text())
            dash = (
                config.get("channels", {})
                .get("discord", {})
                .get("usageDashboard", {})
            )
            if dash.get("messageId") != self.message_id:
                dash["messageId"] = self.message_id
                path.write_text(json.dumps(config, indent=2))
                logger.info(
                    "UsageDashboard: Persisted message_id={} to config",
                    self.message_id,
                )
        except Exception:
            logger.exception("UsageDashboard: Failed to persist message_id")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def last_data(self) -> UsageData | None:
        """Most recently polled usage data."""
        return self._last_data

    async def force_update(self) -> UsageData | None:
        """Manually trigger a poll + update cycle. Returns the data."""
        if not self._http:
            self._http = httpx.AsyncClient(timeout=30.0)
        await self._tick()
        return self._last_data
