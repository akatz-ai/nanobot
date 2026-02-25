"""Per-turn context logger.

Records what the LLM actually saw at each turn — system prompt, injected
memory blocks, retrieved memory snippets, and resume notices — as a sidecar
JSONL file alongside the session.

This is write-only from the agent loop and read-only from the dashboard.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class TurnContextLogger:
    """Append-only sidecar log of per-turn LLM context.

    For each turn, we record:
      - turn_index: sequential counter for this session
      - timestamp: ISO timestamp
      - message_index: index of the user message that triggered this turn
      - system_prompt_hash: SHA-256 prefix of the system prompt (to detect changes)
      - system_prompt_length: char count
      - system_messages: list of all injected system messages (role=system)
        with their content (or a truncated version for the base prompt)
      - memory_context: the raw retrieved memory string (or null)
      - resume_notice: the resume notice string (or null)
    """

    # We store the full system prompt only when its hash changes from the
    # previous entry to avoid bloating the log.  Otherwise we store just
    # the hash + length.

    def __init__(self, session_path: Path) -> None:
        self._path = session_path.with_suffix(".context.jsonl")
        self._turn_counter = 0
        self._last_prompt_hash: str | None = None
        # Load existing turn count so we continue numbering
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            idx = entry.get("turn_index", 0)
                            if idx >= self._turn_counter:
                                self._turn_counter = idx + 1
                            self._last_prompt_hash = entry.get("system_prompt_hash")
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def log_turn(
        self,
        *,
        built_messages: list[dict[str, Any]],
        memory_context: str | None,
        resume_notice: str | None,
        user_message_index: int | None = None,
    ) -> None:
        """Record the context for one turn.

        Args:
            built_messages: The full message list returned by
                ``ContextBuilder.build_messages()``.
            memory_context: The raw retrieved memory string (before formatting).
            resume_notice: Resume notice if any.
            user_message_index: Index of the triggering user message in the
                session's message list (for cross-referencing).
        """
        try:
            self._write_entry(
                built_messages=built_messages,
                memory_context=memory_context,
                resume_notice=resume_notice,
                user_message_index=user_message_index,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to write context log entry to {}", self._path
            )

    def _write_entry(
        self,
        *,
        built_messages: list[dict[str, Any]],
        memory_context: str | None,
        resume_notice: str | None,
        user_message_index: int | None,
    ) -> None:
        # Extract system messages and the base system prompt
        system_messages: list[dict[str, str]] = []
        system_prompt: str | None = None

        for i, msg in enumerate(built_messages):
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if i == 0:
                # First system message is the base prompt
                system_prompt = content
                system_messages.append({
                    "index": i,
                    "label": "system_prompt",
                    "length": len(content),
                    "hash": self._hash(content),
                })
            else:
                # Categorize the system message
                label = self._categorize_system_message(content)
                system_messages.append({
                    "index": i,
                    "label": label,
                    "content": content,
                    "length": len(content),
                })

        prompt_hash = self._hash(system_prompt) if system_prompt else None
        prompt_changed = prompt_hash != self._last_prompt_hash

        entry: dict[str, Any] = {
            "turn_index": self._turn_counter,
            "timestamp": datetime.now().isoformat(),
            "user_message_index": user_message_index,
            "system_prompt_hash": prompt_hash,
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "system_prompt_changed": prompt_changed,
            "system_messages": system_messages,
            "memory_context_raw": memory_context,
            "resume_notice": resume_notice,
            "total_messages": len(built_messages),
            "total_chars": sum(
                len(m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", "")))
                for m in built_messages
            ),
        }

        # Include full system prompt only when it changes
        if prompt_changed and system_prompt is not None:
            entry["system_prompt_full"] = system_prompt

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._last_prompt_hash = prompt_hash
        self._turn_counter += 1

    @staticmethod
    def _categorize_system_message(content: str) -> str:
        """Heuristic label for injected system messages."""
        if "## Current Session" in content:
            return "session_info"
        if "Long-term Memory" in content or "<memory_file_data>" in content:
            return "long_term_memory"
        if "Daily History" in content or "<daily_history_data>" in content:
            return "daily_history"
        if "Retrieved Memory" in content:
            return "retrieved_memory"
        if "interrupted mid-turn" in content or "system restart" in content:
            return "resume_notice"
        return "other"


def load_context_log(session_path: Path) -> list[dict[str, Any]]:
    """Read all entries from a context log file.

    Args:
        session_path: Path to the session JSONL (the ``.context.jsonl``
            sibling will be read).

    Returns:
        List of context log entries, ordered by turn_index.
    """
    log_path = session_path.with_suffix(".context.jsonl")
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

    entries.sort(key=lambda e: e.get("turn_index", 0))
    return entries
