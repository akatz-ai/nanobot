"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime
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
                        "description": "Full updated long-term memory as markdown using canonical MEMORY.md "
                        "headings (# MEMORY, Identity & Preferences, Active Projects, Decisions, "
                        "Reference Facts, Recent Context). Return unchanged if nothing new.",
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
    _MEMORY_MD_MAX_TOKENS = 4000
    _MEMORY_MD_MAX_CHARS = 16000
    _CANONICAL_SECTIONS = (
        "Identity & Preferences",
        "Active Projects",
        "Decisions",
        "Reference Facts",
        "Recent Context",
    )

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.history_dir = ensure_dir(self.memory_dir / "history")
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

        current_memory = self._coerce_to_canonical(self.read_long_term())
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

Use this exact MEMORY.md structure:
# MEMORY
## Identity & Preferences
## Active Projects
## Decisions
## Reference Facts
## Recent Context

Hard limits for memory_update: <= {self._MEMORY_MD_MAX_TOKENS} tokens and <= {self._MEMORY_MD_MAX_CHARS} chars.
Reject chatter and transient details.

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
                normalized = self._strip_code_fence(update)
                if not normalized.strip():
                    logger.warning("Memory consolidation: rejected empty memory_update")
                elif not self._is_valid_canonical_structure(normalized):
                    logger.warning("Memory consolidation: rejected malformed memory_update")
                else:
                    canonical = self._coerce_to_canonical(normalized)
                    bounded, overflow_sections = self._enforce_memory_budget(canonical)
                    if not self._fits_memory_budget(bounded):
                        logger.warning("Memory consolidation: rejected over-budget memory_update")
                    else:
                        self._archive_memory_overflow(
                            overflow_sections=overflow_sections,
                            session_key=session.key,
                            timestamp=datetime.now(),
                        )
                        if bounded != current_memory:
                            self.write_long_term(bounded)

            session.last_consolidated = target_last_consolidated
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        candidate = (text or "").strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return candidate

    def _coerce_to_canonical(self, content: str) -> str:
        text = self._strip_code_fence(content or "").strip()
        if not text:
            return self._render_canonical({name: [] for name in self._CANONICAL_SECTIONS})

        if not self._is_valid_canonical_structure(text):
            facts = [line.strip() for line in text.splitlines() if line.strip()]
            sections = {name: [] for name in self._CANONICAL_SECTIONS}
            sections["Reference Facts"] = [f"- {line}" for line in facts]
            return self._render_canonical(sections)

        return self._render_canonical(self._parse_canonical_sections(text))

    def _is_valid_canonical_structure(self, content: str) -> bool:
        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if not lines or lines[0] != "# MEMORY":
            return False
        headers = [line[3:].strip() for line in lines if line.startswith("## ")]
        if len(headers) != len(set(headers)):
            return False
        return all(section in headers for section in self._CANONICAL_SECTIONS)

    def _parse_canonical_sections(self, content: str) -> dict[str, list[str]]:
        sections = {name: [] for name in self._CANONICAL_SECTIONS}
        current: str | None = None
        for raw_line in content.strip().splitlines():
            line = raw_line.rstrip()
            if line.startswith("# "):
                continue
            if line.startswith("## "):
                header = line[3:].strip()
                current = header if header in sections else None
                continue
            if current is not None:
                sections[current].append(line)
        return sections

    def _render_canonical(self, sections: dict[str, list[str]]) -> str:
        lines = ["# MEMORY", ""]
        for header in self._CANONICAL_SECTIONS:
            body = sections.get(header, [])
            if not body or not any(line.strip() for line in body):
                body = ["- (none)"]
            lines.append(f"## {header}")
            lines.extend(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _fits_memory_budget(self, content: str) -> bool:
        return (
            len(content) <= self._MEMORY_MD_MAX_CHARS
            and self._estimate_tokens(content) <= self._MEMORY_MD_MAX_TOKENS
        )

    def _enforce_memory_budget(self, content: str) -> tuple[str, list[tuple[str, list[str]]]]:
        if self._fits_memory_budget(content):
            return content, []

        sections = self._parse_canonical_sections(content)
        overflow: dict[str, list[str]] = {}

        for header in reversed(self._CANONICAL_SECTIONS):
            if self._fits_memory_budget(self._render_canonical(sections)):
                break
            body = [line for line in sections.get(header, []) if line.strip()]
            if not body:
                continue
            overflow.setdefault(header, []).extend(body)
            sections[header] = ["- (archived due to size budget)"]

        for header in self._CANONICAL_SECTIONS:
            body = sections.get(header, [])
            while len(body) > 1 and not self._fits_memory_budget(self._render_canonical(sections)):
                removed = body.pop()
                if removed.strip():
                    overflow.setdefault(header, []).insert(0, removed)
            if not any(line.strip() for line in body):
                sections[header] = ["- (archived due to size budget)"]

        bounded = self._render_canonical(sections)
        overflow_sections = [(header, lines) for header, lines in overflow.items() if lines]
        return bounded, overflow_sections

    def _archive_memory_overflow(
        self,
        overflow_sections: list[tuple[str, list[str]]],
        *,
        session_key: str,
        timestamp: datetime,
    ) -> None:
        if not overflow_sections:
            return

        history_file = self.history_dir / f"{timestamp.date().isoformat()}.md"
        heading = f"# {timestamp.date().isoformat()}\n\n"
        if not history_file.exists():
            history_file.write_text(heading, encoding="utf-8")

        payload = json.dumps(overflow_sections, ensure_ascii=False, sort_keys=True)
        marker_id = hashlib.sha256(
            f"{session_key}|{timestamp.isoformat()}|{payload}".encode("utf-8")
        ).hexdigest()[:16]
        marker = f"<!-- memory_overflow_id: {marker_id} -->"

        existing = history_file.read_text(encoding="utf-8")
        if marker in existing:
            return

        lines = [
            marker,
            f"## {timestamp.strftime('%H:%M')} — MEMORY.md overflow archive ({session_key})",
            "",
            "Moved sections from MEMORY.md to enforce compaction budget:",
            "",
        ]
        for header, body in overflow_sections:
            lines.append(f"### {header}")
            lines.extend(body)
            lines.append("")

        section = "\n".join(lines).rstrip()
        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        history_file.write_text(f"{existing}{separator}{section}\n", encoding="utf-8")
