"""Per-compaction event logger.

Records the full compaction flow — trigger conditions, pre/post context layout,
extraction request/response, and final context shape — as a sidecar JSONL file
alongside the session.

Write-only from the agent loop; read-only from dashboards and CLI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class CompactionEvent:
    """Builder for a single compaction event record.

    Usage::

        event = CompactionEvent(session_key="discord:123")
        event.set_trigger(input_tokens=151411, threshold=140000, ...)
        event.set_pre_compaction_context(...)
        event.add_batch(batch_index=0, ...)
        event.set_post_compaction_context(...)
        event.finalize(success=True)
        # then: compaction_logger.write(event)
    """

    def __init__(self, session_key: str) -> None:
        self.data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_key": session_key,
            "trigger": {},
            "pre_context": {},
            "batches": [],
            "post_context": {},
            "result": {},
        }

    # ── Trigger info ──────────────────────────────────────────────────

    def set_trigger(
        self,
        *,
        input_tokens: int,
        threshold: int,
        context_window: int,
        utilization_pct: float,
        total_messages: int,
        last_consolidated: int,
        context_anchor: int,
    ) -> None:
        self.data["trigger"] = {
            "input_tokens": input_tokens,
            "threshold": threshold,
            "context_window": context_window,
            "utilization_pct": round(utilization_pct, 1),
            "total_messages": total_messages,
            "last_consolidated": last_consolidated,
            "context_anchor": context_anchor,
        }

    # ── Pre-compaction context layout ─────────────────────────────────

    def set_pre_compaction_context(
        self,
        *,
        system_prompt_chars: int,
        memory_md_chars: int,
        history_chars: int,
        conversation_messages: int,
        conversation_start_index: int,
        continuity_snapshot_chars: int = 0,
    ) -> None:
        self.data["pre_context"] = {
            "system_prompt_chars": system_prompt_chars,
            "memory_md_chars": memory_md_chars,
            "history_chars": history_chars,
            "conversation_messages": conversation_messages,
            "conversation_start_index": conversation_start_index,
            "continuity_snapshot_chars": continuity_snapshot_chars,
        }

    # ── Per-batch extraction info ─────────────────────────────────────

    def add_batch(
        self,
        *,
        batch_index: int,
        msg_start: int,
        msg_end: int,
        transcript_chars: int,
        llm_response_chars: int,
        llm_response_preview: str,
        items_extracted: int,
        success: bool,
        error: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        self.data["batches"].append({
            "batch_index": batch_index,
            "msg_range": [msg_start, msg_end],
            "transcript_chars": transcript_chars,
            "llm_response_chars": llm_response_chars,
            "llm_response_preview": llm_response_preview[:500],
            "items_extracted": items_extracted,
            "success": success,
            "error": error,
            "duration_ms": duration_ms,
        })

    # ── Post-compaction context layout ────────────────────────────────

    def set_post_compaction_context(
        self,
        *,
        new_context_anchor: int,
        new_last_consolidated: int,
        keep_count: int,
        visible_messages: int,
        continuity_chars: int = 0,
        total_items_extracted: int = 0,
        history_file: str | None = None,
    ) -> None:
        self.data["post_context"] = {
            "new_context_anchor": new_context_anchor,
            "new_last_consolidated": new_last_consolidated,
            "keep_count": keep_count,
            "visible_messages": visible_messages,
            "continuity_chars": continuity_chars,
            "total_items_extracted": total_items_extracted,
            "history_file": history_file,
        }

    # ── Final result ──────────────────────────────────────────────────

    def finalize(
        self,
        *,
        success: bool,
        error: str | None = None,
        total_duration_ms: int = 0,
    ) -> None:
        self.data["result"] = {
            "success": success,
            "error": error,
            "total_duration_ms": total_duration_ms,
            "total_batches": len(self.data["batches"]),
            "total_items": sum(b.get("items_extracted", 0) for b in self.data["batches"]),
        }


class CompactionLogger:
    """Append-only sidecar log of compaction events for a session."""

    def __init__(self, session_path: Path) -> None:
        self._path = session_path.with_suffix(".compaction.jsonl")

    def write(self, event: CompactionEvent) -> None:
        """Append a compaction event to the log."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.data, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to write compaction log entry to {}", self._path
            )

    @property
    def path(self) -> Path:
        return self._path


def load_compaction_log(session_path: Path) -> list[dict[str, Any]]:
    """Read all entries from a compaction log file.

    Args:
        session_path: Path to the session JSONL (the ``.compaction.jsonl``
            sibling will be read).

    Returns:
        List of compaction event entries, ordered by timestamp.
    """
    log_path = session_path.with_suffix(".compaction.jsonl")
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda e: e.get("timestamp", ""))
    return entries
