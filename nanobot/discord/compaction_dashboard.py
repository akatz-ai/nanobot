"""Discord compaction dashboard — displays the full compaction flow for each agent.

Shows the last compaction event per agent: trigger conditions, pre/post context layout,
per-batch extraction details (including LLM response preview), and final context shape.
Auto-updates alongside the system status dashboard.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from nanobot.session.compaction_log import load_compaction_log

if TYPE_CHECKING:
    from nanobot.agent.router import AgentRouter

# ── Discord API ────────────────────────────────────────────────────────────

DISCORD_API_BASE = "https://discord.com/api/v10"
COMPONENTS_V2_FLAG = 1 << 15  # 32768 — IS_COMPONENTS_V2

# ── Visual ─────────────────────────────────────────────────────────────────

BAR_WIDTH = 20

ACCENT_COLOR_OK = 0x43B581      # green
ACCENT_COLOR_WARN = 0xFFA500    # orange
ACCENT_COLOR_FAIL = 0xFF4444    # red
ACCENT_COLOR_IDLE = 0x585CF2    # blurple


def _progress_bar(pct: float, width: int = BAR_WIDTH) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _format_ms(ms: int) -> str:
    if ms >= 60_000:
        return f"{ms / 60_000:.1f}m"
    if ms >= 1_000:
        return f"{ms / 1_000:.1f}s"
    return f"{ms}ms"


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _collect_last_compaction(router: "AgentRouter") -> dict[str, dict[str, Any] | None]:
    """Get the last compaction event for each agent."""
    results: dict[str, dict[str, Any] | None] = {}

    for agent_id, instance in router.agents.items():
        loop = instance.loop
        best_event: dict[str, Any] | None = None

        for session_info in loop.sessions.list_sessions():
            key = session_info.get("key", "")
            if key.startswith("cron:"):
                continue
            path_str = session_info.get("path")
            if not path_str:
                continue

            events = load_compaction_log(Path(path_str))
            if events:
                last = events[-1]
                if best_event is None or last.get("timestamp", "") > best_event.get("timestamp", ""):
                    best_event = last

        results[agent_id] = best_event

    return results


def _render_context_bar(
    label: str,
    chars: int,
    total_chars: int,
    emoji: str = "📦",
) -> str:
    """Render a labeled section of the context window."""
    if total_chars <= 0:
        return f"{emoji} {label}: {chars:,} chars"
    pct = chars / total_chars
    bar_chars = max(1, int(pct * BAR_WIDTH))
    bar = "▓" * bar_chars
    return f"{emoji} {label}: `{bar}` {chars:,}c ({pct:.0%})"


def render_compaction_dashboard(
    compaction_data: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Build Discord Components V2 payload for the compaction dashboard."""

    # Filter to agents that have compaction data
    agents_with_data = {k: v for k, v in compaction_data.items() if v is not None}

    if not agents_with_data:
        return [
            {
                "type": 17,
                "accent_color": ACCENT_COLOR_IDLE,
                "components": [
                    {
                        "type": 10,
                        "content": "## 🔄 Compaction Log\nNo compaction events recorded yet.",
                    }
                ],
            }
        ]

    sections: list[str] = ["## 🔄 Compaction Log"]

    # Sort: general first, then alphabetical
    sorted_agents = sorted(
        agents_with_data.items(),
        key=lambda x: (0 if x[0] == "general" else 1, x[0]),
    )

    any_failure = False
    for agent_id, event in sorted_agents:
        if not event:
            continue

        trigger = event.get("trigger", {})
        pre = event.get("pre_context", {})
        batches = event.get("batches", [])
        post = event.get("post_context", {})
        result = event.get("result", {})

        success = result.get("success", False)
        if not success:
            any_failure = True

        status_emoji = "✅" if success else "❌"
        ts = event.get("timestamp", "")
        # Parse ISO timestamp to Discord timestamp
        ts_display = ts[:19].replace("T", " ") if ts else "unknown"

        sections.append(f"### {status_emoji} `{agent_id}` — {ts_display} UTC")

        # ── Trigger ──
        util_pct = trigger.get("utilization_pct", 0)
        input_tok = trigger.get("input_tokens", 0)
        threshold = trigger.get("threshold", 0)
        ctx_window = trigger.get("context_window", 200_000)
        total_msgs = trigger.get("total_messages", 0)
        last_cons = trigger.get("last_consolidated", 0)
        anchor = trigger.get("context_anchor", 0)

        bar = _progress_bar(util_pct / 100)
        sections.append(
            f"**Trigger:** `{bar}` {util_pct:.1f}% "
            f"({_format_tokens(input_tok)}/{_format_tokens(ctx_window)})"
        )
        sections.append(
            f"　Messages: {total_msgs} total · "
            f"consolidated: {last_cons} · anchor: {anchor}"
        )

        # ── Pre-compaction context layout ──
        if pre:
            sys_chars = pre.get("system_prompt_chars", 0)
            mem_chars = pre.get("memory_md_chars", 0)
            conv_msgs = pre.get("conversation_messages", 0)
            conv_start = pre.get("conversation_start_index", 0)
            cont_chars = pre.get("continuity_snapshot_chars", 0)

            sections.append("**Pre-compaction context:**")
            sections.append(f"　📋 System prompt: {sys_chars:,}c · 🧠 MEMORY.md: {mem_chars:,}c")
            sections.append(f"　💬 Conversation: {conv_msgs} msgs (from idx {conv_start})")
            if cont_chars:
                sections.append(f"　📎 Continuity snapshot: {cont_chars:,}c")

        # ── Batches ──
        if batches:
            total_batches = len(batches)
            ok_batches = sum(1 for b in batches if b.get("success"))
            total_items = sum(b.get("items_extracted", 0) for b in batches)
            sections.append(
                f"**Extraction:** {ok_batches}/{total_batches} batches · "
                f"{total_items} items extracted"
            )

            for b in batches:
                b_idx = b.get("batch_index", 0)
                b_start, b_end = b.get("msg_range", [0, 0])
                b_ok = b.get("success", False)
                b_items = b.get("items_extracted", 0)
                b_dur = b.get("duration_ms", 0)
                b_tc = b.get("transcript_chars", 0)
                b_rc = b.get("llm_response_chars", 0)
                b_preview = b.get("llm_response_preview", "")
                b_err = b.get("error")

                b_emoji = "✅" if b_ok else "❌"
                sections.append(
                    f"　{b_emoji} Batch {b_idx}: msgs[{b_start}:{b_end}] "
                    f"→ {b_items} items · {_format_ms(b_dur)} "
                    f"(transcript: {_format_tokens(b_tc)}c, response: {_format_tokens(b_rc)}c)"
                )

                if b_preview and (not b_ok or b_items == 0):
                    # Show the LLM response preview for failed/empty batches
                    preview_text = _truncate(b_preview, 200)
                    sections.append(f"　　`{preview_text}`")
                if b_err and not b_ok:
                    sections.append(f"　　⚠️ {_truncate(b_err, 150)}")

        # ── Post-compaction context layout ──
        if post and success:
            new_anchor = post.get("new_context_anchor", 0)
            new_cons = post.get("new_last_consolidated", 0)
            keep = post.get("keep_count", 25)
            visible = post.get("visible_messages", 0)
            cont_chars = post.get("continuity_chars", 0)
            items = post.get("total_items_extracted", 0)
            hist_file = post.get("history_file")

            sections.append("**Post-compaction context:**")
            sections.append(
                f"　📍 Anchor: {new_anchor} · Consolidated: {new_cons} · "
                f"Visible: {visible} msgs (keep={keep})"
            )
            if cont_chars:
                sections.append(f"　📎 Continuity injected: {cont_chars:,}c")
            if items:
                sections.append(f"　💾 {items} items → history" + (f" ({hist_file})" if hist_file else ""))

        # ── Duration ──
        dur = result.get("total_duration_ms", 0)
        err = result.get("error")
        if dur:
            sections.append(f"**Duration:** {_format_ms(dur)}")
        if err and not success:
            sections.append(f"**Error:** {_truncate(err, 200)}")

        sections.append("")  # blank line between agents

    # Footer
    sections.append(f"-# Updated <t:{int(time.time())}:R>")

    content = "\n".join(sections)

    # Truncate if over Discord's 4000 char limit
    if len(content) > 3900:
        content = content[:3900] + "\n-# _(truncated)_"

    if any_failure:
        color = ACCENT_COLOR_FAIL
    else:
        color = ACCENT_COLOR_OK

    return [
        {
            "type": 17,
            "accent_color": color,
            "components": [
                {
                    "type": 10,
                    "content": content,
                }
            ],
        }
    ]


class CompactionDashboard:
    """Async service that polls compaction logs and updates a Discord message.

    Designed to run alongside SystemStatusDashboard in the same channel.
    """

    def __init__(
        self,
        router: "AgentRouter",
        discord_token: str,
        channel_id: str,
        poll_interval_s: int = 60,
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

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "CompactionDashboard started (channel={}, interval={}s)",
            self.channel_id,
            self.poll_interval_s,
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("CompactionDashboard stopped")

    async def close(self) -> None:
        self.stop()
        if self._http:
            await self._http.aclose()
            self._http = None

    async def _run_loop(self) -> None:
        # Initial delay so system status message is created first
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
                logger.exception("CompactionDashboard tick error")

    async def _tick(self) -> None:
        try:
            data = _collect_last_compaction(self.router)
        except Exception:
            logger.exception("CompactionDashboard: Failed to collect data")
            return

        components = render_compaction_dashboard(data)

        if self.message_id:
            await self._edit_message(components)
        else:
            await self._create_message(components)

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
            resp.raise_for_status()
            data = resp.json()
            self.message_id = data["id"]
            logger.info("CompactionDashboard: Created message {}", self.message_id)
            self._persist_message_id()
        except Exception:
            logger.exception("CompactionDashboard: Failed to create message")

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
                logger.warning("CompactionDashboard: Message deleted, recreating...")
                self.message_id = None
                return await self._create_message(components)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                await asyncio.sleep(retry_after)
                return await self._edit_message(components)
            resp.raise_for_status()
            logger.debug("CompactionDashboard: Updated message {}", self.message_id)
        except Exception:
            logger.exception("CompactionDashboard: Failed to edit message")

    def _persist_message_id(self) -> None:
        if not self.config_path or not self.message_id:
            return
        try:
            import json as _json
            from pathlib import Path as P

            path = P(self.config_path)
            if not path.exists():
                return
            config = _json.loads(path.read_text())
            dash = (
                config.get("channels", {})
                .get("discord", {})
                .get("compactionDashboard", {})
            )
            if not dash:
                config.setdefault("channels", {}).setdefault("discord", {})["compactionDashboard"] = {}
                dash = config["channels"]["discord"]["compactionDashboard"]
            if dash.get("messageId") != self.message_id:
                dash["messageId"] = self.message_id
                path.write_text(_json.dumps(config, indent=2))
                logger.info("CompactionDashboard: Persisted message_id={}", self.message_id)
        except Exception:
            logger.exception("CompactionDashboard: Failed to persist message_id")
