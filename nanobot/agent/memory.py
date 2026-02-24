"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""
    _MAX_CONSOLIDATION_INPUT_CHARS = 28000
    _MAX_CONSOLIDATION_MESSAGES = 400
    _MAX_LINE_CONTENT_CHARS = 700
    _MAX_CONTENT_BLOCK_ITEMS = 10

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = re.sub(r"data:[^;\s]+;base64,[A-Za-z0-9+/=\s]+", "[image data omitted]", text)
        # Long opaque tokens (often base64 blobs) degrade consolidation quality.
        text = re.sub(r"[A-Za-z0-9+/=]{180,}", "[blob omitted]", text)
        return " ".join(text.split()).strip()

    @classmethod
    def _clip_text(cls, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if max_chars <= 3:
            return text[:max_chars]
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @classmethod
    def _compact_content(cls, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return cls._clip_text(cls._clean_text(content), cls._MAX_LINE_CONTENT_CHARS)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content[: cls._MAX_CONTENT_BLOCK_ITEMS]:
                if isinstance(item, str):
                    cleaned = cls._clean_text(item)
                    if cleaned:
                        parts.append(cleaned)
                    continue
                if not isinstance(item, dict):
                    rendered = cls._clean_text(json.dumps(item, ensure_ascii=False))
                    if rendered:
                        parts.append(rendered)
                    continue
                item_type = str(item.get("type") or "")
                if item_type in {"image_url", "input_image", "image"}:
                    parts.append("[image]")
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(cls._clean_text(text_value))
                    continue
                if item_type:
                    parts.append(f"[{item_type}]")
            compact = " | ".join(part for part in parts if part)
            return cls._clip_text(compact, cls._MAX_LINE_CONTENT_CHARS)
        rendered = cls._clean_text(json.dumps(content, ensure_ascii=False))
        return cls._clip_text(rendered, cls._MAX_LINE_CONTENT_CHARS)

    def _build_consolidation_lines(self, old_messages: list[dict]) -> list[str]:
        selected = old_messages
        omitted_earlier = 0
        if len(selected) > self._MAX_CONSOLIDATION_MESSAGES:
            omitted_earlier = len(selected) - self._MAX_CONSOLIDATION_MESSAGES
            selected = selected[-self._MAX_CONSOLIDATION_MESSAGES:]

        lines: list[str] = []
        used_chars = 0

        if omitted_earlier > 0:
            prefix = f"[... omitted {omitted_earlier} earlier messages ...]"
            lines.append(prefix)
            used_chars += len(prefix) + 1

        for idx, m in enumerate(selected):
            compact = self._compact_content(m.get("content"))
            if not compact:
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            role = str(m.get("role") or "?").upper()
            stamp = str(m.get("timestamp") or "?")[:16]
            line = f"[{stamp}] {role}{tools}: {compact}"
            projected = used_chars + len(line) + 1
            if projected > self._MAX_CONSOLIDATION_INPUT_CHARS:
                omitted_later = len(selected) - idx
                marker = f"[... omitted {omitted_later} messages due to input size cap ...]"
                remaining = self._MAX_CONSOLIDATION_INPUT_CHARS - used_chars - 1
                if remaining > 0:
                    lines.append(self._clip_text(marker, remaining))
                elif lines:
                    # Replace the previous line so we always expose truncation state.
                    previous = lines.pop()
                    used_chars = max(0, used_chars - len(previous) - 1)
                    remaining = self._MAX_CONSOLIDATION_INPUT_CHARS - used_chars - 1
                    if remaining > 0:
                        lines.append(self._clip_text(marker, remaining))
                break
            lines.append(line)
            used_chars = projected
        return lines

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = list(session.messages)
            keep_count = 0
            target_last_consolidated = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            end_index = len(session.messages) - keep_count
            old_messages = session.messages[session.last_consolidated:end_index]
            if not old_messages:
                return True
            target_last_consolidated = end_index
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = self._build_consolidation_lines(old_messages)

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines) or "(empty)"}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = target_last_consolidated
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
