"""Session management for conversation history."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

_UNSET = object()


@dataclass
class CompactionEntry:
    """Structured compaction checkpoint persisted in session JSONL."""

    summary: str
    first_kept_index: int
    tokens_before: int
    file_ops: dict[str, list[str]]
    previous_summary: str | None
    timestamp: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionEntry:
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("CompactionEntry.summary must be a non-empty string")

        raw_first_kept = data.get("first_kept_index", 0)
        try:
            first_kept_index = int(raw_first_kept)
        except (TypeError, ValueError) as e:
            raise ValueError("CompactionEntry.first_kept_index must be an int") from e

        raw_tokens_before = data.get("tokens_before", 0)
        try:
            tokens_before = int(raw_tokens_before)
        except (TypeError, ValueError) as e:
            raise ValueError("CompactionEntry.tokens_before must be an int") from e

        file_ops_raw = data.get("file_ops", {})
        if not isinstance(file_ops_raw, dict):
            file_ops_raw = {}
        file_ops: dict[str, list[str]] = {}
        for key in ("read_files", "modified_files"):
            values = file_ops_raw.get(key, [])
            if isinstance(values, list):
                file_ops[key] = [str(item) for item in values if item is not None]
            else:
                file_ops[key] = []

        previous_summary_raw = data.get("previous_summary")
        previous_summary = (
            previous_summary_raw if isinstance(previous_summary_raw, str) else None
        )

        timestamp_raw = data.get("timestamp")
        timestamp = (
            timestamp_raw if isinstance(timestamp_raw, str) and timestamp_raw.strip()
            else datetime.now().isoformat()
        )

        return cls(
            summary=summary,
            first_kept_index=first_kept_index,
            tokens_before=max(0, tokens_before),
            file_ops=file_ops,
            previous_summary=previous_summary,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "first_kept_index": self.first_kept_index,
            "tokens_before": self.tokens_before,
            "file_ops": {
                "read_files": list(self.file_ops.get("read_files", [])),
                "modified_files": list(self.file_ops.get("modified_files", [])),
            },
            "previous_summary": self.previous_summary,
            "timestamp": self.timestamp,
        }


def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphaned tool results and their dangling assistant tool_calls.

    Ensures every ``tool`` message has a matching ``tool_calls`` entry in a
    preceding ``assistant`` message, and every assistant ``tool_calls`` entry
    has corresponding ``tool`` result messages following it.  Orphaned messages
    on either side are dropped so the conversation stays valid for the API.
    """
    # 1. Collect all tool_call IDs present in assistant messages.
    available_tool_call_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tc_id = tc.get("id") or (tc.get("function") or {}).get("id")
                if tc_id:
                    available_tool_call_ids.add(tc_id)

    # 2. Collect all tool_call IDs that have a tool result.
    answered_tool_call_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id and tc_id in available_tool_call_ids:
                answered_tool_call_ids.add(tc_id)

    # 3. Drop tool messages whose tool_call_id is not in any assistant message.
    cleaned: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id not in available_tool_call_ids:
                logger.warning(
                    "Dropping orphaned tool result (tool_call_id={})", tc_id
                )
                continue
        cleaned.append(m)

    # 4. Strip unanswered tool_calls from assistant messages so the API
    #    doesn't expect tool results that will never arrive.
    result: list[dict[str, Any]] = []
    for m in cleaned:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            kept_calls = [
                tc for tc in m["tool_calls"]
                if (tc.get("id") or (tc.get("function") or {}).get("id"))
                in answered_tool_call_ids
            ]
            if kept_calls != m["tool_calls"]:
                m = dict(m)
                if kept_calls:
                    m["tool_calls"] = kept_calls
                else:
                    m.pop("tool_calls", None)
                # If assistant message has no content and no tool_calls, skip it
                if not m.get("content") and not m.get("tool_calls"):
                    logger.warning("Dropping empty assistant message after tool_call cleanup")
                    continue
        result.append(m)

    return result


@dataclass
class PruneResult:
    messages_pruned: int
    tokens_saved: int
    messages_protected: int
    total_tool_messages: int


_DEFAULT_PROTECTED_TOOLS = {
    "memory_recall",
    "memory_save",
    "memory_graph",
    "memory_ingest",
}


def _tool_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if content is None:
        return 0
    try:
        return len(json.dumps(content, ensure_ascii=False))
    except TypeError:
        return len(str(content))


def _prune_tool_results(
    messages: list[dict[str, Any]],
    *,
    prune_protect_tokens: int,
    prune_minimum_tokens: int,
    protected_tools: set[str] | None = None,
) -> tuple[list[dict[str, Any]], PruneResult]:
    """Prune older tool result payloads from a view of prompt messages."""
    # Lazy import to avoid circular import: compaction imports Session.
    from nanobot.session.compaction import estimate_message_tokens

    protected = {
        str(name)
        for name in (
            protected_tools if protected_tools is not None else _DEFAULT_PROTECTED_TOOLS
        )
    }
    turns_seen = 0
    total_tool_tokens = 0
    total_tool_messages = 0
    messages_protected = 0
    candidates: list[tuple[int, str, int]] = []

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        role = msg.get("role")
        if role == "user":
            turns_seen += 1
            continue
        if role != "tool":
            continue

        total_tool_messages += 1

        # Keep all tool results for the most recent two user turns.
        if turns_seen < 2:
            continue

        tool_tokens = estimate_message_tokens(msg)
        total_tool_tokens += tool_tokens

        tool_name = msg.get("name")
        if isinstance(tool_name, str) and tool_name in protected:
            messages_protected += 1
            continue

        if total_tool_tokens <= prune_protect_tokens:
            continue

        original_chars = _tool_content_chars(msg.get("content"))
        placeholder = (
            f"[Tool output cleared to save context — {original_chars} chars]"
        )
        placeholder_msg = dict(msg)
        placeholder_msg["content"] = placeholder
        placeholder_tokens = estimate_message_tokens(placeholder_msg)
        saved = max(0, tool_tokens - placeholder_tokens)
        if saved <= 0:
            continue
        candidates.append((idx, placeholder, saved))

    total_saved = sum(item[2] for item in candidates)
    if total_saved < prune_minimum_tokens or not candidates:
        return messages, PruneResult(
            messages_pruned=0,
            tokens_saved=0,
            messages_protected=messages_protected,
            total_tool_messages=total_tool_messages,
        )

    out = list(messages)
    for idx, placeholder, _saved in candidates:
        pruned = dict(out[idx])
        pruned["content"] = placeholder
        out[idx] = pruned

    return out, PruneResult(
        messages_pruned=len(candidates),
        tokens_saved=total_saved,
        messages_protected=messages_protected,
        total_tool_messages=total_tool_messages,
    )


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files
    _path: Path | None = field(default=None, repr=False, compare=False)
    _persisted_count: int = field(default=0, repr=False, compare=False)
    _persisted_compaction_count: int = field(default=0, repr=False, compare=False)
    _persisted_last_consolidated: int = field(default=0, repr=False, compare=False)
    _persisted_metadata_sig: str = field(default="", repr=False, compare=False)
    _persisted_compactions_sig: str = field(default="", repr=False, compare=False)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def bind_path(self, path: Path) -> None:
        """Attach a concrete JSONL path used for append-only checkpoints."""
        self._path = path
        if not self._persisted_metadata_sig:
            self._persisted_metadata_sig = self._metadata_signature()
        if not self._persisted_compactions_sig:
            self._persisted_compactions_sig = self._compactions_signature()

    def _metadata_signature(self) -> str:
        """Stable signature of metadata for no-op save detection."""
        try:
            return json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return repr(self.metadata)

    def mark_persisted(self) -> None:
        """Record the current in-memory state as fully persisted."""
        self._persisted_count = len(self.messages)
        self._persisted_compaction_count = len(self.compactions)
        self._persisted_last_consolidated = self.last_consolidated
        self._persisted_metadata_sig = self._metadata_signature()
        self._persisted_compactions_sig = self._compactions_signature()

    def _compactions_signature(self) -> str:
        """Stable signature of compaction entries for no-op save detection."""
        try:
            return json.dumps(self.compactions, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return repr(self.compactions)

    def _ensure_file_initialized(self) -> None:
        """Create session file with metadata header if missing/empty."""
        if self._path is None:
            return
        ensure_dir(self._path.parent)
        if self._path.exists() and self._path.stat().st_size > 0:
            return
        metadata_line = {
            "_type": "metadata",
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "last_consolidated": self.last_consolidated,
        }
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _normalize_checkpoint_entry(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        entry = dict(msg)
        role = entry.get("role")
        if role not in {"user", "assistant", "tool"}:
            return None
        if role == "assistant" and not entry.get("content") and not entry.get("tool_calls"):
            return None

        if role == "user":
            content = entry.get("content")
            if isinstance(content, str):
                from nanobot.agent.context import ContextBuilder

                if content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    return None
            elif isinstance(content, list):
                entry["content"] = [
                    {"type": "text", "text": "[image]"}
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image_url"
                        and isinstance(item.get("image_url"), dict)
                        and str(item.get("image_url", {}).get("url", "")).startswith("data:image/")
                    )
                    else item
                    for item in content
                ]

        entry.setdefault("timestamp", datetime.now().isoformat())
        return entry

    def append(self, msg: dict[str, Any]) -> None:
        """Append one message to in-memory history and session JSONL."""
        self.checkpoint([msg])

    def checkpoint(self, msgs: list[dict[str, Any]]) -> None:
        """Append multiple messages to in-memory history and session JSONL."""
        if not msgs:
            return

        entries: list[dict[str, Any]] = []
        for msg in msgs:
            entry = self._normalize_checkpoint_entry(msg)
            if entry is not None:
                entries.append(entry)

        if not entries:
            return

        self.messages.extend(entries)
        self.updated_at = datetime.now()

        if self._path is None:
            return

        self._ensure_file_initialized()
        with open(self._path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._persisted_count = len(self.messages)

    def append_compaction(
        self,
        *,
        summary: str,
        first_kept_index: int,
        tokens_before: int,
        file_ops: dict[str, list[str]] | None = None,
        previous_summary: str | None = None,
        timestamp: str | None = None,
    ) -> CompactionEntry:
        """Append a compaction entry and persist it directly to JSONL when bound."""
        entry = CompactionEntry(
            summary=summary,
            first_kept_index=int(first_kept_index),
            tokens_before=max(0, int(tokens_before)),
            file_ops={
                "read_files": list((file_ops or {}).get("read_files", [])),
                "modified_files": list((file_ops or {}).get("modified_files", [])),
            },
            previous_summary=previous_summary,
            timestamp=timestamp or datetime.now().isoformat(),
        )
        payload = entry.to_dict()
        self.compactions.append(payload)
        self.updated_at = datetime.now()

        if self._path is not None:
            self._ensure_file_initialized()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_type": "compaction", **payload}, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._persisted_compaction_count = len(self.compactions)
            self._persisted_compactions_sig = self._compactions_signature()

        return entry

    def get_last_compaction(self) -> CompactionEntry | None:
        """Return the latest compaction entry with bounded first_kept_index."""
        if not self.compactions:
            return None
        raw = self.compactions[-1]
        if not isinstance(raw, dict):
            logger.warning("Invalid compaction entry type in session {}: {}", self.key, type(raw).__name__)
            return None
        try:
            entry = CompactionEntry.from_dict(raw)
        except ValueError as e:
            logger.warning("Skipping invalid compaction entry in session {}: {}", self.key, e)
            return None

        clamped_index = max(0, min(entry.first_kept_index, len(self.messages)))
        if clamped_index != entry.first_kept_index:
            logger.warning(
                "Clamped compaction first_kept_index for {}: {} -> {} (messages={})",
                self.key,
                entry.first_kept_index,
                clamped_index,
                len(self.messages),
            )
            entry.first_kept_index = clamped_index
            self.compactions[-1] = entry.to_dict()

        # Also clamp last_consolidated to stay within bounds — compaction may
        # have trimmed the session while last_consolidated still pointed at
        # the pre-trim message count.
        if self.last_consolidated > len(self.messages):
            logger.warning(
                "Clamped last_consolidated for {}: {} -> {} (messages={})",
                self.key,
                self.last_consolidated,
                len(self.messages),
                len(self.messages),
            )
            self.last_consolidated = len(self.messages)

        return entry

    def validate_compaction_invariants(self) -> None:
        """Validate and repair cursor invariants before persistence."""
        if self.last_consolidated < 0:
            logger.warning(
                "Clamped negative last_consolidated for {}: {} -> 0",
                self.key,
                self.last_consolidated,
            )
            self.last_consolidated = 0
        if self.last_consolidated > len(self.messages):
            logger.warning(
                "Clamped last_consolidated for {} at save: {} -> {} (messages={})",
                self.key,
                self.last_consolidated,
                len(self.messages),
                len(self.messages),
            )
            self.last_consolidated = len(self.messages)

    def get_message_count(self) -> int:
        """Return total persisted/in-memory message count for this session."""
        return len(self.messages)

    def get_visible_message_count(self) -> int:
        """Return the prompt-visible message count after compaction boundary."""
        last_compaction = self.get_last_compaction()
        boundary = last_compaction.first_kept_index if last_compaction else 0
        return max(0, self.get_message_count() - boundary)
    
    def get_history(
        self,
        max_messages: int = 500,
        prune_tool_results: bool = True,
        prune_protect_tokens: int | None = None,
        prune_minimum_tokens: int | None = None,
        context_window: int = 200_000,
        protected_tools: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], PruneResult | None]:
        """Return prompt-window messages for LLM input, aligned to a user turn.

        Returns:
            Tuple of (messages, prune_result).  ``prune_result`` is ``None``
            when pruning is disabled or not applicable.
        """
        last_compaction = self.get_last_compaction()
        if last_compaction:
            prompt_window = self.messages[last_compaction.first_kept_index:]
        else:
            prompt_window = self.messages
        sliced = prompt_window[-max_messages:]

        # Drop leading assistant/tool messages to avoid orphaned tool_result blocks,
        # but preserve any legacy leading system messages in older sessions.
        if last_compaction is None:
            leading_system: list[dict[str, Any]] = []
            start_index = 0
            for i, m in enumerate(sliced):
                if m.get("role") == "system":
                    leading_system.append(m)
                    start_index = i + 1
                    continue
                start_index = i
                break
            for i, m in enumerate(sliced[start_index:], start=start_index):
                if m.get("role") == "user":
                    sliced = leading_system + sliced[i:]
                    break

        out: list[dict[str, Any]] = []
        for m in sliced:
            entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
            for k in ("tool_calls", "tool_call_id", "name"):
                if k in m:
                    entry[k] = m[k]
            out.append(entry)

        # Sanitize: remove orphaned tool results whose tool_use_id has no
        # matching tool_call in a preceding assistant message.
        out = _sanitize_tool_pairs(out)

        prune_result: PruneResult | None = None
        if prune_tool_results and out:
            effective_protect = (
                prune_protect_tokens
                if prune_protect_tokens is not None
                else max(10_000, int(context_window) // 5)
            )
            effective_minimum = (
                prune_minimum_tokens
                if prune_minimum_tokens is not None
                else max(5_000, int(context_window) // 20)
            )
            out, prune_result = _prune_tool_results(
                out,
                prune_protect_tokens=max(0, int(effective_protect)),
                prune_minimum_tokens=max(0, int(effective_minimum)),
                protected_tools=protected_tools,
            )
            if prune_result.messages_pruned:
                logger.info(
                    "Tool result pruning for {}: pruned={} saved_tokens={} protected={} total_tool_messages={}",
                    self.key,
                    prune_result.messages_pruned,
                    prune_result.tokens_saved,
                    prune_result.messages_protected,
                    prune_result.total_tool_messages,
                )
        return out, prune_result

    def detect_resume_state(self) -> str:
        """Inspect raw messages and classify restart state."""
        if not self.messages:
            return "clean"

        tail = self.messages[-1]
        role = tail.get("role")

        if role == "assistant":
            tool_calls = tail.get("tool_calls") or []
            return "mid_tool" if tool_calls else "clean"

        if role == "tool":
            return "mid_loop"

        return "clean"
    
    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.compactions = []
        self.last_consolidated = 0
        self.metadata.pop("usage_snapshot", None)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = Path.home() / ".nanobot" / "sessions"
        self._cache: dict[str, Session] = {}
    
    def get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_session_path(self, key: str) -> Path:
        """Backward-compatible alias for get_session_path()."""
        return self.get_session_path(key)

    def get_inbox(self, key: str):
        """Get durable sidecar inbox for the given session key."""
        from nanobot.session.inbox import SessionInbox

        return SessionInbox(self.get_session_path(key))

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]
        
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        session.bind_path(self._get_session_path(key))
        if session._persisted_count == 0 and session._path and session._path.exists():
            session.mark_persisted()

        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            compactions: list[dict[str, Any]] = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed session line {} in {}: {}", lineno, path, e)
                        continue

                    if not isinstance(data, dict):
                        logger.warning("Skipping non-object session line {} in {}", lineno, path)
                        continue

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            try:
                                created_at = datetime.fromisoformat(data["created_at"])
                            except ValueError:
                                logger.warning(
                                    "Invalid created_at in metadata at line {} in {}",
                                    lineno,
                                    path,
                                )
                        if data.get("updated_at"):
                            try:
                                updated_at = datetime.fromisoformat(data["updated_at"])
                            except ValueError:
                                logger.warning(
                                    "Invalid updated_at in metadata at line {} in {}",
                                    lineno,
                                    path,
                                )
                        raw_last_consolidated = data.get("last_consolidated", 0)
                        try:
                            parsed_last_consolidated = int(raw_last_consolidated)
                        except (TypeError, ValueError):
                            parsed_last_consolidated = 0
                        last_consolidated = max(0, parsed_last_consolidated)
                    elif data.get("_type") == "compaction":
                        try:
                            compaction = CompactionEntry.from_dict(data)
                        except ValueError as e:
                            logger.warning(
                                "Skipping malformed compaction line {} in {}: {}",
                                lineno,
                                path,
                                e,
                            )
                            continue
                        compactions.append(compaction.to_dict())
                    elif "_type" in data:
                        logger.warning(
                            "Skipping unknown session _type '{}' at line {} in {}",
                            data.get("_type"),
                            lineno,
                            path,
                        )
                    else:
                        messages.append(data)

            if last_consolidated > len(messages):
                last_consolidated = len(messages)

            # --- V3 extraction migration ---
            # Sessions created before the V3 extraction pipeline have
            # last_consolidated=0 even though they may contain thousands of
            # messages.  Without this migration the first compaction after
            # upgrade would try to extract the *entire* history through
            # Haiku, which is both expensive and low-quality on huge windows.
            #
            # Fix: when a session has messages but last_consolidated was
            # never advanced (still 0), skip the backlog by setting the
            # pointer to the current message count.  New extractions will
            # start from here going forward.
            if last_consolidated == 0 and len(messages) > 0:
                logger.info(
                    "V3 extraction migration for {}: advancing last_consolidated "
                    "from 0 → {} (skipping historical backlog)",
                    key,
                    len(messages),
                )
                last_consolidated = len(messages)

            for idx, compaction in enumerate(compactions):
                raw_first_kept = compaction.get("first_kept_index", 0)
                try:
                    parsed_first_kept = int(raw_first_kept)
                except (TypeError, ValueError):
                    parsed_first_kept = 0
                clamped_first_kept = max(0, min(parsed_first_kept, len(messages)))
                if clamped_first_kept != parsed_first_kept:
                    logger.warning(
                        "Clamped compaction first_kept_index for {} at entry {}: {} -> {}",
                        key,
                        idx,
                        parsed_first_kept,
                        clamped_first_kept,
                    )
                compaction["first_kept_index"] = clamped_first_kept
            session = Session(
                key=key,
                messages=messages,
                compactions=compactions,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
            session.bind_path(path)
            session.mark_persisted()
            return session
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        session.validate_compaction_invariants()
        path = self._get_session_path(session.key)
        session.bind_path(path)
        if (
            path.exists()
            and session._persisted_count == len(session.messages)
            and session._persisted_compaction_count == len(session.compactions)
            and session._persisted_last_consolidated == session.last_consolidated
            and session._persisted_metadata_sig == session._metadata_signature()
            and session._persisted_compactions_sig == session._compactions_signature()
        ):
            self._cache[session.key] = session
            return

        session.updated_at = datetime.now()
        temp_path = path.with_suffix(f"{path.suffix}.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                for compaction in session.compactions:
                    f.write(
                        json.dumps({"_type": "compaction", **compaction}, ensure_ascii=False) + "\n"
                    )
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

        session.mark_persisted()
        self._cache[session.key] = session

    def save_state(self, session: Session) -> None:
        """Persist session row/state changes without requiring a history rewrite."""
        self.save(session)

    def apply_state(
        self,
        session: Session,
        *,
        metadata_updates: dict[str, Any] | None = None,
        metadata_remove: list[str] | tuple[str, ...] | None = None,
        last_consolidated: int | object = _UNSET,
    ) -> dict[str, Any]:
        """Apply persisted session-row changes without treating Session as the write API."""
        removed: dict[str, Any] = {}

        if metadata_updates:
            for key, value in metadata_updates.items():
                session.metadata[str(key)] = value

        if metadata_remove:
            for key in metadata_remove:
                text_key = str(key)
                if text_key in session.metadata:
                    removed[text_key] = session.metadata.pop(text_key)

        if last_consolidated is not _UNSET:
            session.last_consolidated = int(last_consolidated)

        self.save_state(session)
        return removed

    def set_usage_snapshot(
        self,
        session: Session,
        *,
        total_input_tokens: int,
        message_index: int | None = None,
        source: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "total_input_tokens": int(total_input_tokens),
            "message_index": session.get_message_count() if message_index is None else int(message_index),
        }
        if source:
            payload["source"] = source
        self.apply_state(session, metadata_updates={"usage_snapshot": payload})

    def clear_usage_snapshot(self, session: Session) -> None:
        self.apply_state(session, metadata_remove=["usage_snapshot"])

    def set_compaction_plan(self, session: Session, plan: dict[str, Any]) -> None:
        self.apply_state(
            session,
            metadata_updates={"_structured_compaction_plan": dict(plan)},
        )

    def pop_compaction_plan(self, session: Session) -> dict[str, Any]:
        removed = self.apply_state(
            session,
            metadata_remove=["_structured_compaction_plan"],
        )
        value = removed.get("_structured_compaction_plan", {})
        return value if isinstance(value, dict) else {}

    def advance_last_consolidated(self, session: Session, value: int) -> None:
        self.apply_state(session, last_consolidated=int(value))
    
    def save_all(self) -> int:
        """Flush all cached sessions to disk.

        Returns:
            Number of sessions saved.
        """
        saved = 0
        for key, session in list(self._cache.items()):
            try:
                self.save(session)
                saved += 1
            except Exception:
                logger.exception("Failed to flush session {} on shutdown", key)
        return saved

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
