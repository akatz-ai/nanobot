"""Discord system status dashboard — displays agent health and context window utilization.

Polls agent state from the AgentRouter and renders a Components V2 message in Discord
with per-agent context window progress bars, token stats, and system health info.
Auto-updates on a configurable interval (default: 60s).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from nanobot.session.usage_log import get_session_summary

if TYPE_CHECKING:
    from nanobot.agent.router import AgentRouter

# ── Discord API ────────────────────────────────────────────────────────────

DISCORD_API_BASE = "https://discord.com/api/v10"
COMPONENTS_V2_FLAG = 1 << 15  # 32768 — IS_COMPONENTS_V2

# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_POLL_INTERVAL_S = 60  # 1 minute
BAR_WIDTH = 20  # characters for progress bar
COMPACTION_THRESHOLD = 0.70  # 70% of context window

# ── Visual ─────────────────────────────────────────────────────────────────

ACCENT_COLOR_NORMAL = 0x585CF2  # blurple-ish
ACCENT_COLOR_WARNING = 0xFFA500  # orange
ACCENT_COLOR_CRITICAL = 0xFF4444  # red


def _progress_bar(pct: float, width: int = BAR_WIDTH) -> str:
    """Render a Unicode progress bar. pct is 0.0–1.0."""
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _status_emoji(utilization: float) -> str:
    """Pick an emoji based on context utilization level."""
    if utilization >= 0.90:
        return "🔴"
    if utilization >= 0.70:
        return "🟡"
    if utilization > 0:
        return "🟢"
    return "⚪"


def _format_tokens(n: int) -> str:
    """Format token count as human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _get_worker_uptime() -> float | None:
    """Get worker process uptime in seconds using /proc."""
    import os

    pid = os.getpid()
    proc_stat = Path(f"/proc/{pid}/stat")
    proc_uptime = Path("/proc/uptime")
    if not proc_stat.exists() or not proc_uptime.exists():
        return None
    try:
        hz = os.sysconf("SC_CLK_TCK")
        start_ticks = int(proc_stat.read_text().split(")")[1].split()[19])
        uptime_total = float(proc_uptime.read_text().split()[0])
        return max(0.0, uptime_total - (start_ticks / hz))
    except Exception:
        return None


@dataclass
class AgentStatus:
    """Status data for a single agent."""

    agent_id: str
    model: str = ""
    context_window: int = 200_000
    compaction_threshold: int = 140_000

    # Per-session data (most recent active session)
    current_input_tokens: int = 0
    utilization_pct: float = 0.0
    total_turns: int = 0
    total_calls: int = 0
    cumulative_input: int = 0
    cumulative_output: int = 0
    avg_cache_hit_pct: float = 0.0
    active_sessions: int = 0
    is_idle: bool = True


@dataclass
class SystemStatus:
    """Aggregated system status."""

    agents: list[AgentStatus] = field(default_factory=list)
    uptime_seconds: float | None = None
    polled_at: float = 0.0


def collect_system_status(router: AgentRouter) -> SystemStatus:
    """Collect current system status from the agent router.

    This reads the in-memory state of each agent — no I/O or API calls needed.
    """
    agent_statuses: list[AgentStatus] = []

    for agent_id, instance in router.agents.items():
        loop = instance.loop

        context_window = loop._get_context_window_size()
        compaction_threshold = loop._compaction_token_threshold

        # Find the most active session (highest token count)
        best_tokens = 0
        best_summary: dict[str, Any] | None = None
        active_count = 0

        for session_info in loop.sessions.list_sessions():
            key = session_info.get("key", "")
            # Skip cron sessions
            if key.startswith("cron:"):
                continue

            active_count += 1
            tokens = loop._last_input_tokens.get(key, 0)
            if tokens > best_tokens:
                best_tokens = tokens

            # Try to get usage summary from sidecar log
            path_str = session_info.get("path")
            if path_str:
                summary = get_session_summary(Path(path_str))
                if summary and summary.get("total_input_tokens", 0) > (
                    best_summary or {}
                ).get("total_input_tokens", 0):
                    best_summary = summary

        utilization = best_tokens / context_window if context_window > 0 else 0
        is_idle = best_tokens == 0

        status = AgentStatus(
            agent_id=agent_id,
            model=loop.model or "",
            context_window=context_window,
            compaction_threshold=compaction_threshold,
            current_input_tokens=best_tokens,
            utilization_pct=utilization,
            active_sessions=active_count,
            is_idle=is_idle,
        )

        if best_summary:
            status.total_turns = best_summary.get("total_turns", 0)
            status.total_calls = best_summary.get("total_calls", 0)
            status.cumulative_input = best_summary.get("total_input_tokens", 0)
            status.cumulative_output = best_summary.get("total_output_tokens", 0)
            status.avg_cache_hit_pct = best_summary.get("avg_cache_hit_pct", 0)

        agent_statuses.append(status)

    # Sort: default/general first, then by agent_id
    agent_statuses.sort(key=lambda a: (0 if a.agent_id == "general" else 1, a.agent_id))

    return SystemStatus(
        agents=agent_statuses,
        uptime_seconds=_get_worker_uptime(),
        polled_at=time.time(),
    )


def render_dashboard(status: SystemStatus) -> list[dict[str, Any]]:
    """Build Discord Components V2 payload from system status.

    Returns the `components` list for a Components V2 message.
    """
    active_count = sum(1 for a in status.agents if not a.is_idle)
    total_count = len(status.agents)
    uptime_str = _format_uptime(status.uptime_seconds) if status.uptime_seconds else "unknown"

    # Overall health color
    max_util = max((a.utilization_pct for a in status.agents), default=0)
    if max_util >= 0.90:
        color = ACCENT_COLOR_CRITICAL
    elif max_util >= 0.70:
        color = ACCENT_COLOR_WARNING
    else:
        color = ACCENT_COLOR_NORMAL

    lines: list[str] = []
    lines.append(f"## 🖥️ System Status")
    lines.append(f"**Uptime:** {uptime_str} · **Agents:** {active_count}/{total_count} active")
    lines.append("")

    for agent in status.agents:
        emoji = _status_emoji(agent.utilization_pct)
        bar = _progress_bar(agent.utilization_pct)
        pct_str = f"{agent.utilization_pct * 100:4.1f}%"
        tokens_str = f"{_format_tokens(agent.current_input_tokens)}/{_format_tokens(agent.context_window)}"

        # Model short name (strip provider prefix)
        model_short = agent.model.split("/")[-1] if "/" in agent.model else agent.model

        if agent.is_idle:
            lines.append(f"{emoji} **{agent.agent_id}** · `{model_short}`")
            lines.append(f"　Context: `{bar}`  idle")
        else:
            lines.append(f"{emoji} **{agent.agent_id}** · `{model_short}`")
            lines.append(
                f"　Context: `{bar}`  **{pct_str}** ({tokens_str})"
            )
            # Stats line
            parts = []
            if agent.total_turns:
                parts.append(f"{agent.total_turns} turns")
            if agent.total_calls:
                parts.append(f"{agent.total_calls} calls")
            if agent.avg_cache_hit_pct:
                parts.append(f"Cache: {agent.avg_cache_hit_pct:.0f}%")
            if parts:
                lines.append(f"　{' · '.join(parts)}")

        lines.append("")

    # Footer
    lines.append(f"-# Compacts at {int(COMPACTION_THRESHOLD * 100)}% · Updated <t:{int(status.polled_at)}:R>")

    content = "\n".join(lines)

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


class SystemStatusDashboard:
    """Async service that polls agent status and updates a Discord message.

    Usage::

        dashboard = SystemStatusDashboard(
            router=router,
            discord_token="Bot ...",
            channel_id="123456789",
            poll_interval_s=60,
        )
        await dashboard.start()   # runs forever in background
        ...
        dashboard.stop()
    """

    def __init__(
        self,
        router: AgentRouter,
        discord_token: str,
        channel_id: str,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
        message_id: str | None = None,
        config_path: str | None = None,
    ):
        self.router = router
        self.discord_token = discord_token
        self.channel_id = channel_id
        self.poll_interval_s = poll_interval_s
        self.message_id = message_id
        self.config_path = config_path

        self._running = False
        self._task: asyncio.Task | None = None
        self._http: httpx.AsyncClient | None = None
        self._last_status: SystemStatus | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the dashboard polling loop."""
        if self._running:
            logger.warning("SystemStatusDashboard already running")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "SystemStatusDashboard started (channel={}, interval={}s)",
            self.channel_id,
            self.poll_interval_s,
        )

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("SystemStatusDashboard stopped")

    async def close(self) -> None:
        """Clean up HTTP client."""
        self.stop()
        if self._http:
            await self._http.aclose()
            self._http = None

    # ── Main loop ──────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Poll → render → update, repeat."""
        # Small delay to let agents initialize
        await asyncio.sleep(5)
        await self._tick()

        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SystemStatusDashboard tick error")

    async def _tick(self) -> None:
        """Single poll + update cycle."""
        try:
            status = collect_system_status(self.router)
        except Exception:
            logger.exception("SystemStatusDashboard: Failed to collect status")
            return

        self._last_status = status
        components = render_dashboard(status)

        if self.message_id:
            await self._edit_message(components)
        else:
            await self._create_message(components)

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
            logger.info("SystemStatusDashboard: Created message {}", self.message_id)
            self._persist_message_id()

        except Exception:
            logger.exception("SystemStatusDashboard: Failed to create Discord message")

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
                logger.warning("SystemStatusDashboard: Message deleted, recreating...")
                self.message_id = None
                return await self._create_message(components)

            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                logger.warning("Discord rate limited, retrying in {}s", retry_after)
                await asyncio.sleep(retry_after)
                return await self._edit_message(components)

            resp.raise_for_status()
            logger.debug("SystemStatusDashboard: Updated message {}", self.message_id)

        except Exception:
            logger.exception("SystemStatusDashboard: Failed to edit Discord message")

    # ── Config persistence ──────────────────────────────────────────────

    def _persist_message_id(self) -> None:
        """Save the current message_id back to config.json so restarts reuse it."""
        if not self.config_path or not self.message_id:
            return
        try:
            import json
            from pathlib import Path as P

            path = P(self.config_path)
            if not path.exists():
                return

            config = json.loads(path.read_text())
            dash = (
                config.get("channels", {})
                .get("discord", {})
                .get("systemStatus", {})
            )
            if not dash:
                config.setdefault("channels", {}).setdefault("discord", {})["systemStatus"] = {}
                dash = config["channels"]["discord"]["systemStatus"]

            if dash.get("messageId") != self.message_id:
                dash["messageId"] = self.message_id
                path.write_text(json.dumps(config, indent=2))
                logger.info(
                    "SystemStatusDashboard: Persisted message_id={} to config",
                    self.message_id,
                )
        except Exception:
            logger.exception("SystemStatusDashboard: Failed to persist message_id")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def last_status(self) -> SystemStatus | None:
        """Most recently collected system status."""
        return self._last_status
