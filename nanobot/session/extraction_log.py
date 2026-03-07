"""Per-extraction event logger."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class ExtractionEvent:
    """Builder for a single extraction event record."""

    def __init__(self, session_key: str, agent_id: str) -> None:
        self.data: dict[str, Any] = {
            "extraction_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_key": session_key,
            "agent_id": agent_id,
            "trigger": {},
            "message_window": {},
            "existing_memories_recall": {},
            "prompt": {},
            "llm_call": {},
            "extraction_raw": {},
            "extracted_items": [],
            "graph_indexing": {},
            "file_ops": {},
            "result": {},
        }

    def set_trigger(
        self,
        *,
        reason: str,
        compaction_event_id: str | None,
        total_session_messages: int,
        last_consolidated_before: int,
    ) -> None:
        self.data["trigger"] = {
            "reason": reason,
            "compaction_event_id": compaction_event_id,
            "total_session_messages": total_session_messages,
            "last_consolidated_before": last_consolidated_before,
        }

    def set_message_window(
        self,
        *,
        start_index: int,
        end_index: int,
        user_turns: int,
        context_overlap_start: int,
        context_overlap_count: int,
    ) -> None:
        self.data["message_window"] = {
            "start_index": start_index,
            "end_index": end_index,
            "message_count": max(0, end_index - start_index),
            "user_turns": user_turns,
            "context_overlap": {
                "start_index": context_overlap_start,
                "message_count": context_overlap_count,
            },
        }

    def set_existing_memories(
        self,
        *,
        query_text_chars: int,
        results_requested: int,
        results_returned: int,
        recall_duration_ms: int,
        memories: list[dict[str, Any]],
    ) -> None:
        self.data["existing_memories_recall"] = {
            "query_text_chars": query_text_chars,
            "results_requested": results_requested,
            "results_returned": results_returned,
            "recall_duration_ms": recall_duration_ms,
            "memories": [dict(memory) for memory in memories],
        }

    def set_prompt_stats(
        self,
        *,
        system_prompt_chars: int,
        transcript_chars: int,
        context_section_chars: int,
        extraction_section_chars: int,
        existing_memories_section_chars: int,
        max_tokens_budget: int,
        full_prompt_hash: str,
    ) -> None:
        self.data["prompt"] = {
            "system_prompt_chars": system_prompt_chars,
            "transcript_chars": transcript_chars,
            "context_section_chars": context_section_chars,
            "extraction_section_chars": extraction_section_chars,
            "existing_memories_section_chars": existing_memories_section_chars,
            "max_tokens_budget": max_tokens_budget,
            "full_prompt_hash": full_prompt_hash,
        }

    def set_llm_call(
        self,
        *,
        model: str,
        temperature: float,
        duration_ms: int,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_read_tokens: int | None,
        cache_creation_tokens: int | None,
        retry_needed: bool,
        retry_input_tokens: int | None,
        retry_output_tokens: int | None,
        finish_reason: str,
        error: str | None,
    ) -> None:
        self.data["llm_call"] = {
            "model": model,
            "temperature": temperature,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "retry_needed": retry_needed,
            "retry_input_tokens": retry_input_tokens,
            "retry_output_tokens": retry_output_tokens,
            "finish_reason": finish_reason,
            "error": error,
        }

    def set_extraction_raw(
        self,
        *,
        response_chars: int,
        response_preview: str,
        parse_ok: bool,
        parse_error: str | None,
        items_before_cap: int,
        items_after_cap: int,
        items_by_action: dict[str, int],
        items_by_type: dict[str, int],
        items_by_scope: dict[str, int],
    ) -> None:
        self.data["extraction_raw"] = {
            "response_chars": response_chars,
            "response_preview": response_preview[:500],
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "items_before_cap": items_before_cap,
            "items_after_cap": items_after_cap,
            "items_by_action": dict(items_by_action),
            "items_by_type": dict(items_by_type),
            "items_by_scope": dict(items_by_scope),
        }

    def set_extracted_items(self, items: list[dict[str, Any]]) -> None:
        self.data["extracted_items"] = [dict(item) for item in items]

    def set_graph_indexing(
        self,
        *,
        duration_ms: int,
        memories_added: int,
        memories_updated: int,
        memories_superseded: int,
        edges_created: int,
        items_skipped: int,
        indexed_items: list[dict[str, Any]],
    ) -> None:
        self.data["graph_indexing"] = {
            "duration_ms": duration_ms,
            "memories_added": memories_added,
            "memories_updated": memories_updated,
            "memories_superseded": memories_superseded,
            "edges_created": edges_created,
            "items_skipped": items_skipped,
            "indexed_items": [dict(item) for item in indexed_items],
        }

    def set_file_ops(
        self,
        *,
        history_file: str | None,
        history_entries_written: int,
        memory_md_rewrite_triggered: bool,
        memory_md_before_chars: int,
        memory_md_after_chars: int,
        memory_md_duration_ms: int,
    ) -> None:
        self.data["file_ops"] = {
            "history_file": history_file,
            "history_entries_written": history_entries_written,
            "memory_md_rewrite": {
                "triggered": memory_md_rewrite_triggered,
                "before_chars": memory_md_before_chars,
                "after_chars": memory_md_after_chars,
                "duration_ms": memory_md_duration_ms,
            },
        }

    def finalize(
        self,
        *,
        success: bool,
        error: str | None,
        total_duration_ms: int,
        total_items_extracted: int,
        total_items_indexed: int,
        new_last_consolidated: int,
    ) -> None:
        self.data["result"] = {
            "success": success,
            "error": error,
            "total_duration_ms": total_duration_ms,
            "total_items_extracted": total_items_extracted,
            "total_items_indexed": total_items_indexed,
            "new_last_consolidated": new_last_consolidated,
        }


class ExtractionLogger:
    """Append-only sidecar log of extraction events for a session."""

    def __init__(self, session_path: Path) -> None:
        self._path = session_path.with_suffix(".extraction.jsonl")

    def write(self, event: ExtractionEvent) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.data, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to write extraction log entry to {}", self._path
            )

    @property
    def path(self) -> Path:
        return self._path


def load_extraction_log(session_path: Path) -> list[dict[str, Any]]:
    """Read all entries from an extraction log file."""
    log_path = session_path.with_suffix(".extraction.jsonl")
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda entry: entry.get("timestamp", ""))
    return entries
