"""Discord Codex usage dashboard — displays OpenAI Codex plan utilization.

Polls the ChatGPT Codex Responses backend with a minimal request to read usage
headers, then renders a Components V2 message in Discord with progress bars
that auto-update on a configurable interval.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token

# ── Codex API ──────────────────────────────────────────────────────────────

CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
PROBE_MODEL = "gpt-5.4"

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
class CodexBucketUsage:
    """Usage bucket exposed by Codex response headers."""

    label: str
    key_prefix: str
    used_primary_pct: float = 0.0
    used_secondary_pct: float = 0.0
    primary_window_minutes: int = 300
    secondary_window_minutes: int = 10080
    primary_reset_at: int = 0
    secondary_reset_at: int = 0
    over_secondary_limit_pct: float = 0.0


@dataclass
class CodexUsageData:
    """Parsed plan usage from Codex response headers."""

    plan_type: str = "unknown"
    active_limit: str = "codex"
    has_credits: bool = False
    credits_balance: str = ""
    credits_unlimited: bool = False
    buckets: list[CodexBucketUsage] = None  # type: ignore[assignment]
    polled_at: float = 0.0

    def __post_init__(self) -> None:
        if self.buckets is None:
            self.buckets = []

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> "CodexUsageData":
        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(headers.get(key, str(default)))
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(headers.get(key, str(default)))
            except (TypeError, ValueError):
                return default

        def _str(key: str, default: str = "") -> str:
            return str(headers.get(key, default) or default)

        def _bool(key: str, default: bool = False) -> bool:
            raw = str(headers.get(key, str(default))).strip().lower()
            return raw in {"1", "true", "yes", "on"}

        bucket_specs = [
            ("Codex", "x-codex", "Codex"),
            ("Spark", "x-codex-bengalfox", _str("x-codex-bengalfox-limit-name", "Spark")),
        ]

        buckets: list[CodexBucketUsage] = []
        for _name, prefix, label in bucket_specs:
            primary = _float(f"{prefix}-primary-used-percent")
            secondary = _float(f"{prefix}-secondary-used-percent")
            primary_reset = _int(f"{prefix}-primary-reset-at")
            secondary_reset = _int(f"{prefix}-secondary-reset-at")
            if not any([primary, secondary, primary_reset, secondary_reset]) and prefix != "x-codex":
                continue
            buckets.append(
                CodexBucketUsage(
                    label=label,
                    key_prefix=prefix,
                    used_primary_pct=primary,
                    used_secondary_pct=secondary,
                    primary_window_minutes=_int(f"{prefix}-primary-window-minutes", 300),
                    secondary_window_minutes=_int(f"{prefix}-secondary-window-minutes", 10080),
                    primary_reset_at=primary_reset,
                    secondary_reset_at=secondary_reset,
                    over_secondary_limit_pct=_float(f"{prefix}-primary-over-secondary-limit-percent"),
                )
            )

        return cls(
            plan_type=_str("x-codex-plan-type", "unknown"),
            active_limit=_str("x-codex-active-limit", "codex"),
            has_credits=_bool("x-codex-credits-has-credits"),
            credits_balance=_str("x-codex-credits-balance", ""),
            credits_unlimited=_bool("x-codex-credits-unlimited"),
            buckets=buckets,
            polled_at=time.time(),
        )


def _progress_bar(pct: float, width: int = BAR_WIDTH) -> str:
    """Render a Unicode progress bar. pct is 0.0–1.0."""
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _status_emoji(utilization: float) -> str:
    """Pick an emoji based on utilization level."""
    if utilization >= 0.90:
        return "🔴"
    if utilization >= 0.70:
        return "🟡"
    return "🟢"


def _accent_color(data: CodexUsageData) -> int:
    """Pick container accent color based on overall health."""
    max_util = max(
        [bucket.used_primary_pct / 100.0 for bucket in data.buckets]
        + [bucket.used_secondary_pct / 100.0 for bucket in data.buckets]
        + [0.0]
    )
    if max_util >= 0.90:
        return ACCENT_COLOR_CRITICAL
    if max_util >= 0.70:
        return ACCENT_COLOR_WARNING
    return ACCENT_COLOR_NORMAL


def render_dashboard(data: CodexUsageData) -> list[dict[str, Any]]:
    """Build Discord Components V2 payload from Codex usage data."""
    lines: list[str] = ["## 🤖 Codex Usage"]

    for idx, bucket in enumerate(data.buckets):
        if idx > 0:
            lines.append("")
            lines.append(f"### {bucket.label}")
        elif bucket.label != "Codex":
            lines.append(f"### {bucket.label}")

        primary_util = max(0.0, min(1.0, bucket.used_primary_pct / 100.0))
        secondary_util = max(0.0, min(1.0, bucket.used_secondary_pct / 100.0))

        primary_label = f"{bucket.primary_window_minutes // 60}h Window" if bucket.primary_window_minutes % 60 == 0 else f"{bucket.primary_window_minutes}m Window"
        secondary_label = (
            f"{bucket.secondary_window_minutes // (60 * 24)}d Window"
            if bucket.secondary_window_minutes % (60 * 24) == 0
            else f"{bucket.secondary_window_minutes}m Window"
        )

        for label, util, reset_ts in [
            (primary_label, primary_util, bucket.primary_reset_at),
            (secondary_label, secondary_util, bucket.secondary_reset_at),
        ]:
            emoji = _status_emoji(util)
            bar = _progress_bar(util)
            used_pct = util * 100
            left_pct = max(0.0, 100.0 - used_pct)
            reset_str = f"<t:{reset_ts}:R>" if reset_ts else ""
            lines.append(
                f"{emoji} **{label}**  `{bar}`  **{used_pct:4.1f}% used** · **{left_pct:4.1f}% left**  resets {reset_str}"
            )

        if bucket.over_secondary_limit_pct > 0:
            lines.append(
                f"-# Current window consumed {bucket.over_secondary_limit_pct:.1f}% of the weekly limit"
            )

    lines.append("")
    credits = "unlimited credits" if data.credits_unlimited else (
        f"credits {data.credits_balance}" if data.credits_balance else "no extra credits"
    )
    lines.append(
        f"-# Plan: {data.plan_type.title()} · Active limit: {data.active_limit} · {credits} · Updated <t:{int(data.polled_at)}:R>"
    )

    return [
        {
            "type": 17,  # Container
            "accent_color": _accent_color(data),
            "components": [
                {
                    "type": 10,  # Text Display
                    "content": "\n".join(lines),
                }
            ],
        }
    ]


class CodexUsageDashboard:
    """Async service that polls Codex usage and updates a Discord message."""

    def __init__(
        self,
        discord_token: str,
        channel_id: str,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        message_id: str | None = None,
        config_path: str | None = None,
    ):
        self.discord_token = discord_token
        self.channel_id = channel_id
        self.poll_interval_s = poll_interval_s
        self.message_id = message_id
        self.config_path = config_path

        self._running = False
        self._task: asyncio.Task | None = None
        self._http: httpx.AsyncClient | None = None
        self._last_data: CodexUsageData | None = None

    async def start(self) -> None:
        if self._running:
            logger.warning("CodexUsageDashboard already running")
            return
        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0, verify=False)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "CodexUsageDashboard started (channel={}, interval={}s)",
            self.channel_id,
            self.poll_interval_s,
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("CodexUsageDashboard stopped")

    async def close(self) -> None:
        self.stop()
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _run_loop(self) -> None:
        await self._tick()
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("CodexUsageDashboard tick error")

    async def _tick(self) -> None:
        data = await self._poll_codex()
        if data is None:
            return
        self._last_data = data
        components = render_dashboard(data)
        if self.message_id:
            await self._edit_message(components)
        else:
            await self._create_message(components)

    async def _poll_codex(self) -> CodexUsageData | None:
        if not self._http:
            return None
        try:
            token = await asyncio.to_thread(get_codex_token)
            headers = {
                "Authorization": f"Bearer {token.access}",
                "chatgpt-account-id": token.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": DEFAULT_ORIGINATOR,
                "User-Agent": "nanobot (python)",
                "accept": "text/event-stream",
                "content-type": "application/json",
            }
            body: dict[str, Any] = {
                "model": PROBE_MODEL,
                "store": False,
                "stream": True,
                "instructions": "You are a usage probe.",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "."}]}],
                "text": {"verbosity": "low"},
                "prompt_cache_key": "codex-usage-dashboard-probe",
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
            async with self._http.stream("POST", CODEX_API_URL, headers=headers, json=body) as resp:
                if resp.status_code == 429:
                    logger.warning("CodexUsageDashboard: Rate limited (429), will retry next interval")
                    return CodexUsageData.from_headers(resp.headers)
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "ignore")
                    logger.warning(
                        "CodexUsageDashboard: Codex returned {}: {}",
                        resp.status_code,
                        text[:200],
                    )
                    return None
                data = CodexUsageData.from_headers(resp.headers)
                async for line in resp.aiter_lines():
                    if line:
                        break
                return data
        except Exception:
            logger.exception("CodexUsageDashboard: Failed to poll Codex")
            return None

    async def _create_message(self, components: list[dict[str, Any]]) -> None:
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
            logger.info("CodexUsageDashboard: Created message {}", self.message_id)
            self._persist_message_id()
        except Exception:
            logger.exception("CodexUsageDashboard: Failed to create Discord message")

    async def _edit_message(self, components: list[dict[str, Any]]) -> None:
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
                logger.warning("CodexUsageDashboard: Message deleted, recreating...")
                self.message_id = None
                return await self._create_message(components)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                logger.warning("Discord rate limited, retrying in {}s", retry_after)
                await asyncio.sleep(retry_after)
                return await self._edit_message(components)
            resp.raise_for_status()
            logger.debug("CodexUsageDashboard: Updated message {}", self.message_id)
        except Exception:
            logger.exception("CodexUsageDashboard: Failed to edit Discord message")

    def _persist_message_id(self) -> None:
        if not self.config_path or not self.message_id:
            return
        try:
            from pathlib import Path
            from nanobot.config.state import StateStore

            store = StateStore.from_config_path(Path(self.config_path))
            state = store.load()
            if state.channels.discord.codex_usage.message_id != self.message_id:
                store.set_discord_message_id("codex_usage", self.message_id)
                logger.info("CodexUsageDashboard: Persisted message_id={} to state", self.message_id)
        except Exception:
            logger.exception("CodexUsageDashboard: Failed to persist message_id")

    @property
    def last_data(self) -> CodexUsageData | None:
        return self._last_data

    async def force_update(self) -> CodexUsageData | None:
        if not self._http:
            self._http = httpx.AsyncClient(timeout=30.0, verify=False)
        await self._tick()
        return self._last_data
