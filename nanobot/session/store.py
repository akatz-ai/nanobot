"""SQLite-backed session storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

from loguru import logger

from nanobot.providers.base import effective_total_input_tokens
from nanobot.session.manager import (
    CompactionEntry,
    PruneResult,
    Session,
    SessionManager,
    _PROMPT_PRUNE_MINIMUM_TOKENS,
    _PROMPT_PRUNE_PROTECT_TOKENS,
    _DEFAULT_PROTECTED_TOOLS,
    _tool_prune_placeholder,
    _sanitize_tool_pairs,
    _tool_content_chars,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session (
    key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    message_count INTEGER NOT NULL DEFAULT 0,
    last_consolidated_seq INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    pending_summary_start INTEGER,
    pending_summary_end INTEGER,
    pending_extract_start INTEGER,
    pending_extract_end INTEGER,
    pending_cut_point_type TEXT,
    storage_version INTEGER NOT NULL DEFAULT 1,
    jsonl_path TEXT,
    jsonl_migrated_at TEXT
);

CREATE TABLE IF NOT EXISTS message (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    raw_json TEXT NOT NULL,
    timestamp TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    has_tool_calls INTEGER NOT NULL DEFAULT 0 CHECK (has_tool_calls IN (0, 1)),
    pruned_at TEXT,
    original_content_chars INTEGER,
    pruned_tokens_saved INTEGER,
    approx_tokens INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (session_key, seq)
);

CREATE INDEX IF NOT EXISTS idx_message_session_seq
    ON message(session_key, seq);
CREATE INDEX IF NOT EXISTS idx_message_session_role_seq
    ON message(session_key, role, seq DESC);
CREATE INDEX IF NOT EXISTS idx_message_session_tool_call_id
    ON message(session_key, tool_call_id);
CREATE INDEX IF NOT EXISTS idx_message_session_unpruned_tools
    ON message(session_key, seq DESC)
    WHERE role = 'tool' AND pruned_at IS NULL;

CREATE TABLE IF NOT EXISTS compaction (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    boundary_seq INTEGER NOT NULL,
    summary TEXT NOT NULL,
    tokens_before INTEGER NOT NULL DEFAULT 0,
    file_ops_json TEXT NOT NULL DEFAULT '{}',
    previous_summary TEXT,
    created_at TEXT NOT NULL,
    CHECK (boundary_seq >= 0),
    CHECK (tokens_before >= 0)
);

CREATE INDEX IF NOT EXISTS idx_compaction_session_latest
    ON compaction(session_key, id DESC);
CREATE INDEX IF NOT EXISTS idx_compaction_session_boundary
    ON compaction(session_key, boundary_seq DESC);

CREATE TABLE IF NOT EXISTS provider_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    call_index INTEGER NOT NULL,
    turn INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    provider_name TEXT,
    model TEXT,
    finish_reason TEXT,
    input_tokens_raw INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    total_input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    context_window INTEGER,
    utilization_pct REAL,
    assembly_snapshot_id INTEGER REFERENCES prompt_assembly_snapshot(id) ON DELETE SET NULL,
    produced_message_seq_start INTEGER,
    produced_message_seq_end INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (session_key, call_index)
);

CREATE INDEX IF NOT EXISTS idx_provider_call_session_call_index
    ON provider_call(session_key, call_index);

CREATE TABLE IF NOT EXISTS retrieved_memory_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    turn INTEGER NOT NULL,
    query_message_seq INTEGER,
    content_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    item_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieved_memory_snapshot_session_turn
    ON retrieved_memory_snapshot(session_key, turn, id DESC);

CREATE TABLE IF NOT EXISTS turn_context_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    turn INTEGER NOT NULL,
    channel TEXT,
    chat_id TEXT,
    rendered_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turn_context_snapshot_session_turn
    ON turn_context_snapshot(session_key, turn, id DESC);

CREATE TABLE IF NOT EXISTS prompt_assembly_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    turn INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    provider_name TEXT,
    model TEXT,
    system_prompt_hash TEXT,
    memory_hash TEXT,
    memory_chars INTEGER,
    compaction_id INTEGER REFERENCES compaction(id) ON DELETE SET NULL,
    compaction_summary_hash TEXT,
    message_seq_start INTEGER,
    message_seq_end INTEGER,
    prune_watermark_event_id INTEGER REFERENCES tool_prune_event(id) ON DELETE SET NULL,
    retrieved_memory_snapshot_id INTEGER REFERENCES retrieved_memory_snapshot(id) ON DELETE SET NULL,
    turn_context_snapshot_id INTEGER REFERENCES turn_context_snapshot(id) ON DELETE SET NULL,
    assembled_prompt_hash TEXT NOT NULL,
    assembled_prompt_tokens_est INTEGER,
    stable_prefix_tokens_est INTEGER,
    dynamic_turn_tokens_est INTEGER,
    visible_conversation_tokens_est INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_assembly_snapshot_session_turn
    ON prompt_assembly_snapshot(session_key, turn, iteration);

CREATE TABLE IF NOT EXISTS tool_prune_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    turn INTEGER NOT NULL,
    iteration INTEGER,
    trigger_call_index INTEGER,
    reason TEXT,
    estimated_tokens_before INTEGER,
    estimated_tokens_after INTEGER,
    estimated_tokens_saved INTEGER,
    messages_pruned INTEGER NOT NULL DEFAULT 0,
    tool_results_pruned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_prune_event_session_created
    ON tool_prune_event(session_key, id DESC);

CREATE TABLE IF NOT EXISTS tool_prune_item (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prune_event_id INTEGER NOT NULL REFERENCES tool_prune_event(id) ON DELETE CASCADE,
    session_key TEXT NOT NULL REFERENCES session(key) ON DELETE CASCADE,
    message_seq INTEGER NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT,
    replacement_kind TEXT,
    original_content_chars INTEGER,
    estimated_tokens_removed INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_prune_item_event
    ON tool_prune_item(prune_event_id, message_seq);

CREATE INDEX IF NOT EXISTS idx_tool_prune_item_session_seq
    ON tool_prune_item(session_key, message_seq);
"""
class LazyMessageList(list[dict[str, Any]]):
    """List-like session history that defers hydration until content access."""

    def __init__(
        self,
        manager: SQLiteSessionManager,
        session_key: str,
        *,
        total_count: int = 0,
        loaded_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(loaded_messages or [])
        self._manager = manager
        self._session_key = session_key
        self._total_count = max(0, int(total_count))
        self._loaded = loaded_messages is not None
        if self._loaded:
            self._total_count = len(self)
        self._version = 0
        self._dirty = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_count(self) -> int:
        return self._total_count if not self._loaded else super().__len__()

    def signature(self) -> str:
        return f"lazy:{self.total_count}:{self._version}:{int(self._dirty)}"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        messages = self._manager._load_messages(self._session_key)
        super().clear()
        super().extend(messages)
        self._total_count = super().__len__()
        self._loaded = True
        self._dirty = False

    def note_db_append(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        self._version += 1
        if self._loaded:
            super().extend(entries)
            self._total_count = super().__len__()
        else:
            self._total_count += len(entries)
        self._dirty = False

    def note_db_replace(self, seq: int, message: dict[str, Any]) -> None:
        self._version += 1
        if self._loaded and 0 <= seq < super().__len__():
            super().__setitem__(seq, message)
        self._dirty = False

    def __len__(self) -> int:
        return self.total_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._ensure_loaded()
        return super().__iter__()

    def __getitem__(self, item):  # type: ignore[override]
        self._ensure_loaded()
        return super().__getitem__(item)

    def __eq__(self, other: object) -> bool:
        self._ensure_loaded()
        return super().__eq__(other)

    def __setitem__(self, item, value) -> None:  # type: ignore[override]
        self._ensure_loaded()
        super().__setitem__(item, value)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def __delitem__(self, item) -> None:  # type: ignore[override]
        self._ensure_loaded()
        super().__delitem__(item)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def append(self, item: dict[str, Any]) -> None:
        self._ensure_loaded()
        super().append(item)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def extend(self, iterable) -> None:  # type: ignore[override]
        self._ensure_loaded()
        super().extend(iterable)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def insert(self, index: int, item: dict[str, Any]) -> None:
        self._ensure_loaded()
        super().insert(index, item)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def pop(self, index: int = -1):  # type: ignore[override]
        self._ensure_loaded()
        value = super().pop(index)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True
        return value

    def remove(self, value: dict[str, Any]) -> None:
        self._ensure_loaded()
        super().remove(value)
        self._version += 1
        self._total_count = super().__len__()
        self._dirty = True

    def clear(self) -> None:
        super().clear()
        self._version += 1
        self._total_count = 0
        self._loaded = True
        self._dirty = True

    def reverse(self) -> None:
        self._ensure_loaded()
        super().reverse()
        self._version += 1
        self._dirty = True

    def sort(self, *args: Any, **kwargs: Any) -> None:
        self._ensure_loaded()
        super().sort(*args, **kwargs)
        self._version += 1
        self._dirty = True


class SQLiteSessionManager(SessionManager):
    """SQLite-backed drop-in replacement for SessionManager."""

    def __init__(self, workspace: Path):
        super().__init__(workspace)
        self.db_path = self.sessions_dir / "sessions.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._ensure_compat_schema_locked()
            self._conn.commit()

    def _ensure_compat_schema_locked(self) -> None:
        session_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(session)").fetchall()
        }
        if "revision" not in session_columns:
            self._conn.execute("ALTER TABLE session ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
        for column_name, column_sql in (
            ("pending_summary_start", "INTEGER"),
            ("pending_summary_end", "INTEGER"),
            ("pending_extract_start", "INTEGER"),
            ("pending_extract_end", "INTEGER"),
            ("pending_cut_point_type", "TEXT"),
        ):
            if column_name not in session_columns:
                self._conn.execute(
                    f"ALTER TABLE session ADD COLUMN {column_name} {column_sql}"
                )
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(message)").fetchall()
        }
        if "pruned_tokens_saved" not in columns:
            self._conn.execute("ALTER TABLE message ADD COLUMN pruned_tokens_saved INTEGER")

        provider_call_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(provider_call)").fetchall()
        }
        if "assembly_snapshot_id" not in provider_call_columns:
            self._conn.execute(
                "ALTER TABLE provider_call ADD COLUMN assembly_snapshot_id INTEGER REFERENCES prompt_assembly_snapshot(id) ON DELETE SET NULL"
            )


    @staticmethod
    def _sha256_text(value: str | None) -> str | None:
        if value is None:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _hash_message_payload(cls, messages: list[dict[str, Any]]) -> str:
        return hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _assembly_message_range(assembly: dict[str, Any]) -> tuple[int | None, int | None]:
        sections = assembly.get("sections") if isinstance(assembly, dict) else None
        if not isinstance(sections, list):
            return None, None
        history_indexes: list[int] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            if section.get("cache_scope") != "history":
                continue
            source = str(section.get("source") or "")
            if not source.startswith("history:"):
                continue
            metadata = section.get("metadata")
            if not isinstance(metadata, dict):
                continue
            message_index = metadata.get("message_index")
            try:
                history_indexes.append(int(message_index))
            except (TypeError, ValueError):
                continue
        if not history_indexes:
            return None, None
        return min(history_indexes), max(history_indexes)

    def _latest_compaction_row_locked(self, session_key: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT id, summary
            FROM compaction
            WHERE session_key = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_key,),
        ).fetchone()

    def _latest_prune_event_id_locked(self, session_key: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM tool_prune_event WHERE session_key = ? ORDER BY id DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        if row is None or row["id"] is None:
            return None
        return int(row["id"])

    def _persist_prompt_artifact_snapshots_locked(
        self,
        session: Session,
        *,
        turn: int,
        channel: str | None,
        chat_id: str | None,
        query_message_seq: int | None,
        memory_context: str | None,
        turn_context_text: str | None,
        created_at: str,
    ) -> tuple[int | None, int | None, str | None, int]:
        retrieved_memory_snapshot_id: int | None = None
        turn_context_snapshot_id: int | None = None
        memory_hash: str | None = None
        memory_chars = 0

        if isinstance(memory_context, str):
            memory_text = memory_context
            memory_hash = self._sha256_text(memory_text)
            memory_chars = len(memory_text)
            item_count = sum(1 for line in memory_text.splitlines() if line.strip())
            cursor = self._conn.execute(
                """
                INSERT INTO retrieved_memory_snapshot (
                    session_key,
                    turn,
                    query_message_seq,
                    content_text,
                    content_hash,
                    item_count,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.key,
                    int(turn),
                    int(query_message_seq) if query_message_seq is not None else None,
                    memory_text,
                    memory_hash,
                    item_count,
                    created_at,
                ),
            )
            retrieved_memory_snapshot_id = int(cursor.lastrowid)

        if isinstance(turn_context_text, str):
            rendered_text = turn_context_text
            cursor = self._conn.execute(
                """
                INSERT INTO turn_context_snapshot (
                    session_key,
                    turn,
                    channel,
                    chat_id,
                    rendered_text,
                    content_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.key,
                    int(turn),
                    channel,
                    chat_id,
                    rendered_text,
                    self._sha256_text(rendered_text),
                    created_at,
                ),
            )
            turn_context_snapshot_id = int(cursor.lastrowid)

        return (
            retrieved_memory_snapshot_id,
            turn_context_snapshot_id,
            memory_hash,
            memory_chars,
        )

    def record_provider_call(
        self,
        session: Session,
        *,
        turn: int,
        iteration: int,
        provider_name: str | None = None,
        model: str | None = None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
        context_window: int | None = None,
        produced_message_seq_start: int | None = None,
        produced_message_seq_end: int | None = None,
        prompt_messages: list[dict[str, Any]] | None = None,
        prompt_assembly: dict[str, Any] | None = None,
        memory_context: str | None = None,
        turn_context_text: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        query_message_seq: int | None = None,
        created_at: str | None = None,
    ) -> int:
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens_raw = int(usage_dict.get("prompt_tokens", 0) or 0)
        cache_read_tokens = int(usage_dict.get("cache_read_input_tokens", 0) or 0)
        cache_creation_tokens = int(usage_dict.get("cache_creation_input_tokens", 0) or 0)
        total_input_tokens = effective_total_input_tokens(usage_dict)
        output_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
        total_tokens = total_input_tokens + output_tokens
        window = int(context_window or 0)
        utilization_pct = round(total_input_tokens / window * 100, 1) if window > 0 else 0.0
        now_iso = created_at or datetime.now().isoformat()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    jsonl_path=str(session._path) if session._path else None,
                    preserve_existing_message_count=True,
                    bump_revision=False,
                )
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(call_index), -1) AS max_call_index FROM provider_call WHERE session_key = ?",
                    (session.key,),
                ).fetchone()
                next_call_index = (int(row["max_call_index"]) + 1) if row is not None else 0

                assembly_snapshot_id: int | None = None
                assembly = prompt_assembly if isinstance(prompt_assembly, dict) else {}
                prompt_payload = prompt_messages if isinstance(prompt_messages, list) else []
                if assembly and prompt_payload:
                    (
                        retrieved_memory_snapshot_id,
                        turn_context_snapshot_id,
                        memory_hash,
                        memory_chars,
                    ) = self._persist_prompt_artifact_snapshots_locked(
                        session,
                        turn=int(turn),
                        channel=channel,
                        chat_id=chat_id,
                        query_message_seq=query_message_seq,
                        memory_context=memory_context,
                        turn_context_text=turn_context_text,
                        created_at=now_iso,
                    )
                    system_prompt_hash = self._sha256_text(
                        str(prompt_payload[0].get("content"))
                        if prompt_payload and isinstance(prompt_payload[0], dict)
                        else None
                    )
                    latest_compaction = self._latest_compaction_row_locked(session.key)
                    compaction_id = int(latest_compaction["id"]) if latest_compaction is not None and latest_compaction["id"] is not None else None
                    compaction_summary_hash = self._sha256_text(
                        str(latest_compaction["summary"]) if latest_compaction is not None and latest_compaction["summary"] is not None else None
                    )
                    message_seq_start, message_seq_end = self._assembly_message_range(assembly)
                    assembly_cursor = self._conn.execute(
                        """
                        INSERT INTO prompt_assembly_snapshot (
                            session_key,
                            turn,
                            iteration,
                            provider_name,
                            model,
                            system_prompt_hash,
                            memory_hash,
                            memory_chars,
                            compaction_id,
                            compaction_summary_hash,
                            message_seq_start,
                            message_seq_end,
                            prune_watermark_event_id,
                            retrieved_memory_snapshot_id,
                            turn_context_snapshot_id,
                            assembled_prompt_hash,
                            assembled_prompt_tokens_est,
                            stable_prefix_tokens_est,
                            dynamic_turn_tokens_est,
                            visible_conversation_tokens_est,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.key,
                            int(turn),
                            int(iteration),
                            provider_name,
                            model,
                            system_prompt_hash,
                            memory_hash,
                            memory_chars,
                            compaction_id,
                            compaction_summary_hash,
                            message_seq_start,
                            message_seq_end,
                            self._latest_prune_event_id_locked(session.key),
                            retrieved_memory_snapshot_id,
                            turn_context_snapshot_id,
                            self._hash_message_payload(prompt_payload),
                            int(assembly.get("estimated_total_tokens", 0) or 0),
                            int(((assembly.get("pre_compaction_snapshot") or {}).get("stable_cached_prefix_tokens", 0)) or 0),
                            int(((assembly.get("pre_compaction_snapshot") or {}).get("dynamic_turn_tokens", 0)) or 0),
                            int(((assembly.get("pre_compaction_snapshot") or {}).get("visible_conversation_slice_tokens", 0)) or 0),
                            now_iso,
                        ),
                    )
                    assembly_snapshot_id = int(assembly_cursor.lastrowid)

                cursor = self._conn.execute(
                    """
                    INSERT INTO provider_call (
                        session_key,
                        call_index,
                        turn,
                        iteration,
                        provider_name,
                        model,
                        finish_reason,
                        input_tokens_raw,
                        cache_read_tokens,
                        cache_creation_tokens,
                        total_input_tokens,
                        output_tokens,
                        total_tokens,
                        context_window,
                        utilization_pct,
                        assembly_snapshot_id,
                        produced_message_seq_start,
                        produced_message_seq_end,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.key,
                        next_call_index,
                        int(turn),
                        int(iteration),
                        provider_name,
                        model,
                        finish_reason,
                        input_tokens_raw,
                        cache_read_tokens,
                        cache_creation_tokens,
                        total_input_tokens,
                        output_tokens,
                        total_tokens,
                        window if window > 0 else None,
                        utilization_pct,
                        assembly_snapshot_id,
                        produced_message_seq_start,
                        produced_message_seq_end,
                        now_iso,
                    ),
                )
                self._conn.commit()
                return int(cursor.lastrowid)
            except Exception:
                self._conn.rollback()
                raise

    def list_provider_calls(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM provider_call
                WHERE session_key = ?
                ORDER BY call_index ASC
                """,
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_provider_call(self, session_key: str, call_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *
                FROM provider_call
                WHERE session_key = ? AND id = ?
                LIMIT 1
                """,
                (session_key, int(call_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_prompt_assembly_snapshot(self, session_key: str, snapshot_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *
                FROM prompt_assembly_snapshot
                WHERE session_key = ? AND id = ?
                LIMIT 1
                """,
                (session_key, int(snapshot_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_retrieved_memory_snapshot(self, session_key: str, snapshot_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *
                FROM retrieved_memory_snapshot
                WHERE session_key = ? AND id = ?
                LIMIT 1
                """,
                (session_key, int(snapshot_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_turn_context_snapshot(self, session_key: str, snapshot_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *
                FROM turn_context_snapshot
                WHERE session_key = ? AND id = ?
                LIMIT 1
                """,
                (session_key, int(snapshot_id)),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_compaction(self, session_key: str, compaction_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, boundary_seq, summary, tokens_before, file_ops_json, previous_summary, created_at
                FROM compaction
                WHERE session_key = ? AND id = ?
                LIMIT 1
                """,
                (session_key, int(compaction_id)),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        try:
            payload["file_ops"] = json.loads(payload.pop("file_ops_json") or "{}")
        except json.JSONDecodeError:
            payload["file_ops"] = {}
        return payload

    def list_tool_prune_events(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM tool_prune_event
                WHERE session_key = ?
                ORDER BY id ASC
                """,
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_related_tool_prune_event(
        self,
        session_key: str,
        *,
        turn: int | None = None,
        call_index: int | None = None,
        assembly_snapshot_id: int | None = None,
    ) -> dict[str, Any] | None:
        if assembly_snapshot_id is not None:
            assembly = self.get_prompt_assembly_snapshot(session_key, assembly_snapshot_id)
            if assembly is not None and assembly.get("prune_watermark_event_id") is not None:
                prune_id = int(assembly["prune_watermark_event_id"])
                with self._lock:
                    row = self._conn.execute(
                        "SELECT * FROM tool_prune_event WHERE session_key = ? AND id = ? LIMIT 1",
                        (session_key, prune_id),
                    ).fetchone()
                if row is not None:
                    return dict(row)

        clauses = ["session_key = ?"]
        params: list[Any] = [session_key]
        if turn is not None:
            clauses.append("turn = ?")
            params.append(int(turn))
        if call_index is not None:
            clauses.append("(trigger_call_index = ? OR trigger_call_index = ? - 1)")
            params.extend([int(call_index), int(call_index)])
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM tool_prune_event WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 1",
                tuple(params),
            ).fetchone()
        return dict(row) if row is not None else None

    def build_prompt_pressure_history(self, session_key: str) -> dict[str, Any]:
        provider_calls = self.list_provider_calls(session_key)
        prune_events = self.list_tool_prune_events(session_key)
        with self._lock:
            compaction_rows = self._conn.execute(
                """
                SELECT id, session_key, boundary_seq, summary, tokens_before, created_at
                FROM compaction
                WHERE session_key = ?
                ORDER BY id ASC
                """,
                (session_key,),
            ).fetchall()
        compactions = [dict(row) for row in compaction_rows]

        timeline: list[dict[str, Any]] = []
        for call in provider_calls:
            call_index = int(call.get("call_index") or 0)
            turn = int(call.get("turn") or 0)
            related_prunes = [
                event for event in prune_events
                if int(event.get("turn") or 0) == turn
                and (
                    event.get("trigger_call_index") is None
                    or int(event.get("trigger_call_index") or 0) in {call_index - 1, call_index}
                )
            ]
            related_compactions = [
                compaction for compaction in compactions
                if int(compaction.get("boundary_seq") or 0) <= int(call.get("produced_message_seq_start") or 0)
            ]
            timeline.append(
                {
                    "turn": turn,
                    "call_id": call.get("id"),
                    "call_index": call_index,
                    "iteration": call.get("iteration"),
                    "provider": {
                        "provider_name": call.get("provider_name"),
                        "model": call.get("model"),
                        "finish_reason": call.get("finish_reason"),
                        "total_input_tokens": call.get("total_input_tokens"),
                        "output_tokens": call.get("output_tokens"),
                        "utilization_pct": call.get("utilization_pct"),
                        "assembly_snapshot_id": call.get("assembly_snapshot_id"),
                        "created_at": call.get("created_at"),
                    },
                    "prune_events": related_prunes,
                    "compaction_events": related_compactions,
                }
            )

        return {
            "session_key": session_key,
            "approximate": True,
            "attribution": "Provider totals are correlated with nearby prune and compaction events by turn/call proximity; they are not exact per-message token attribution.",
            "provider_calls": provider_calls,
            "prune_events": prune_events,
            "compaction_events": compactions,
            "timeline": timeline,
        }

    def record_tool_prune_event(
        self,
        session: Session,
        *,
        turn: int,
        iteration: int | None = None,
        trigger_call_index: int | None = None,
        reason: str | None = None,
        estimated_tokens_before: int | None = None,
        estimated_tokens_after: int | None = None,
        estimated_tokens_saved: int | None = None,
        items: list[dict[str, Any]] | None = None,
        created_at: str | None = None,
    ) -> int | None:
        prune_items = list(items or [])
        if not prune_items:
            return None
        now_iso = created_at or datetime.now().isoformat()
        tool_results_pruned = sum(1 for item in prune_items if item.get("tool_name"))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    jsonl_path=str(session._path) if session._path else None,
                    preserve_existing_message_count=True,
                    bump_revision=False,
                )
                cursor = self._conn.execute(
                    """
                    INSERT INTO tool_prune_event (
                        session_key,
                        turn,
                        iteration,
                        trigger_call_index,
                        reason,
                        estimated_tokens_before,
                        estimated_tokens_after,
                        estimated_tokens_saved,
                        messages_pruned,
                        tool_results_pruned,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.key,
                        int(turn),
                        int(iteration) if iteration is not None else None,
                        int(trigger_call_index) if trigger_call_index is not None else None,
                        reason,
                        int(estimated_tokens_before) if estimated_tokens_before is not None else None,
                        int(estimated_tokens_after) if estimated_tokens_after is not None else None,
                        int(estimated_tokens_saved) if estimated_tokens_saved is not None else None,
                        len(prune_items),
                        tool_results_pruned,
                        now_iso,
                    ),
                )
                prune_event_id = int(cursor.lastrowid)
                for item in prune_items:
                    self._conn.execute(
                        """
                        INSERT INTO tool_prune_item (
                            prune_event_id,
                            session_key,
                            message_seq,
                            tool_call_id,
                            tool_name,
                            replacement_kind,
                            original_content_chars,
                            estimated_tokens_removed,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prune_event_id,
                            session.key,
                            int(item["message_seq"]),
                            item.get("tool_call_id"),
                            item.get("tool_name"),
                            item.get("replacement_kind"),
                            int(item.get("original_content_chars", 0) or 0),
                            int(item.get("estimated_tokens_removed", 0) or 0),
                            now_iso,
                        ),
                    )
                self._conn.commit()
                return prune_event_id
            except Exception:
                self._conn.rollback()
                raise

    def list_prompt_assembly_snapshots(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM prompt_assembly_snapshot WHERE session_key = ? ORDER BY id ASC",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_retrieved_memory_snapshots(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM retrieved_memory_snapshot WHERE session_key = ? ORDER BY id ASC",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_turn_context_snapshots(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM turn_context_snapshot WHERE session_key = ? ORDER BY id ASC",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tool_prune_events(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_prune_event WHERE session_key = ? ORDER BY id ASC",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_tool_prune_items(self, session_key: str, *, prune_event_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if prune_event_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM tool_prune_item WHERE session_key = ? ORDER BY id ASC",
                    (session_key,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tool_prune_item WHERE session_key = ? AND prune_event_id = ? ORDER BY id ASC",
                    (session_key, int(prune_event_id)),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_or_create(self, key: str) -> Session:
        """Get an existing session or create a new SQLite-backed session."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                if self._cached_session_needs_reload_locked(cached):
                    reloaded = self._load_session_metadata_locked(key)
                    if reloaded is not None:
                        self._bind_sqlite_session(reloaded)
                        self._cache[key] = reloaded
                        return reloaded
                return cached

            session = self._load_session_metadata_locked(key)
            if session is None:
                session = self._migrate_jsonl_locked(key)

            if session is None:
                session = Session(key=key)
                session.bind_path(self._get_session_path(key))
                session.messages = self._make_lazy_messages(key, total_count=0)
                self._rewrite_session_locked(session)

            self._bind_sqlite_session(session)
            self._cache[key] = session
            return session

    def save(self, session: Session) -> None:
        """Persist session metadata and reconcile any in-memory drift."""
        session.validate_compaction_invariants()
        session.bind_path(self._get_session_path(session.key))

        with self._lock:
            self._bind_sqlite_session(session)
            rewrite_messages = self._messages_changed(session)
            rewrite_compactions = self._compactions_changed(session)
            metadata_changed = session._persisted_metadata_sig != session._metadata_signature()
            last_consolidated_changed = (
                session._persisted_last_consolidated != session.last_consolidated
            )
            if not (
                rewrite_messages
                or rewrite_compactions
                or metadata_changed
                or last_consolidated_changed
            ):
                self._cache[session.key] = session
                return
            self._rewrite_session_locked(
                session,
                rewrite_messages=rewrite_messages,
                rewrite_compactions=rewrite_compactions,
            )
            self._cache[session.key] = session

    def save_state(self, session: Session) -> None:
        """Persist row-level session state without rewriting messages/compactions."""
        session.validate_compaction_invariants()
        session.bind_path(self._get_session_path(session.key))

        with self._lock:
            self._bind_sqlite_session(session)
            if self._messages_changed(session) or self._compactions_changed(session):
                self.save(session)
                return

            metadata_changed = session._persisted_metadata_sig != session._metadata_signature()
            last_consolidated_changed = (
                session._persisted_last_consolidated != session.last_consolidated
            )
            if not (metadata_changed or last_consolidated_changed):
                self._cache[session.key] = session
                return

            now = datetime.now()
            now_iso = now.isoformat()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    preserve_existing_message_count=True,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            session.updated_at = now
            session._persisted_last_consolidated = session.last_consolidated
            session._persisted_metadata_sig = session._metadata_signature()
            self._sync_session_tracking(session)
            self._cache[session.key] = session

    def set_compaction_plan(self, session: Session, plan: dict[str, Any]) -> None:
        """Persist pending compaction plan state as explicit session-row columns."""
        session.bind_path(self._get_session_path(session.key))
        now = datetime.now()
        now_iso = now.isoformat()
        summary_start = (
            int(plan["summary_start"]) if plan.get("summary_start") is not None else None
        )
        summary_end = (
            int(plan["summary_end"]) if plan.get("summary_end") is not None else None
        )
        extract_start = (
            int(plan["extract_start"]) if plan.get("extract_start") is not None else None
        )
        extract_end = (
            int(plan["extract_end"]) if plan.get("extract_end") is not None else None
        )
        cut_point_type = (
            str(plan["cut_point_type"]) if plan.get("cut_point_type") is not None else None
        )

        with self._lock:
            self._bind_sqlite_session(session)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    preserve_existing_message_count=True,
                )
                self._conn.execute(
                    """
                    UPDATE session
                    SET
                        updated_at = ?,
                        revision = revision + 1,
                        pending_summary_start = ?,
                        pending_summary_end = ?,
                        pending_extract_start = ?,
                        pending_extract_end = ?,
                        pending_cut_point_type = ?
                    WHERE key = ?
                    """,
                    (
                        now_iso,
                        summary_start,
                        summary_end,
                        extract_start,
                        extract_end,
                        cut_point_type,
                        session.key,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            session.updated_at = now
            self._sync_session_tracking(session)
            self._cache[session.key] = session

    def pop_compaction_plan(self, session: Session) -> dict[str, Any]:
        """Load and clear pending compaction plan state from explicit session-row columns."""
        session.bind_path(self._get_session_path(session.key))
        now = datetime.now()
        now_iso = now.isoformat()

        with self._lock:
            self._bind_sqlite_session(session)
            row = self._conn.execute(
                """
                SELECT
                    pending_summary_start,
                    pending_summary_end,
                    pending_extract_start,
                    pending_extract_end,
                    pending_cut_point_type
                FROM session
                WHERE key = ?
                """,
                (session.key,),
            ).fetchone()
            if row is None:
                return {}

            plan: dict[str, Any] = {}
            if row["pending_summary_start"] is not None:
                plan["summary_start"] = int(row["pending_summary_start"])
            if row["pending_summary_end"] is not None:
                plan["summary_end"] = int(row["pending_summary_end"])
            if row["pending_extract_start"] is not None:
                plan["extract_start"] = int(row["pending_extract_start"])
            if row["pending_extract_end"] is not None:
                plan["extract_end"] = int(row["pending_extract_end"])
            if row["pending_cut_point_type"] is not None:
                plan["cut_point_type"] = str(row["pending_cut_point_type"])
            has_pending = any(
                row[name] is not None
                for name in (
                    "pending_summary_start",
                    "pending_summary_end",
                    "pending_extract_start",
                    "pending_extract_end",
                    "pending_cut_point_type",
                )
            )
            if not has_pending:
                return {}

            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    UPDATE session
                    SET
                        updated_at = ?,
                        revision = revision + 1,
                        pending_summary_start = NULL,
                        pending_summary_end = NULL,
                        pending_extract_start = NULL,
                        pending_extract_end = NULL,
                        pending_cut_point_type = NULL
                    WHERE key = ?
                    """,
                    (now_iso, session.key),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            session.updated_at = now
            self._sync_session_tracking(session)
            self._cache[session.key] = session
            return plan

    def list_sessions(self) -> list[dict[str, Any]]:
        """List sessions from SQLite plus any JSONL files not yet migrated."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, created_at, updated_at
                FROM session
                ORDER BY updated_at DESC
                """
            ).fetchall()

        sessions: dict[str, dict[str, Any]] = {
            str(row["key"]): {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "path": str(self.get_session_path(str(row["key"]))),
            }
            for row in rows
        }

        for directory in (self.sessions_dir, self.legacy_sessions_dir):
            if not directory.exists():
                continue
            for path in directory.glob("*.jsonl"):
                try:
                    with open(path, encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    if not first_line:
                        continue
                    data = json.loads(first_line)
                    if not isinstance(data, dict) or data.get("_type") != "metadata":
                        continue
                    key = data.get("key") or path.stem.replace("_", ":", 1)
                    if not isinstance(key, str) or key in sessions:
                        continue
                    sessions[key] = {
                        "key": key,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "path": str(path),
                    }
                except Exception:
                    continue

        return sorted(
            sessions.values(),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )

    def _load(self, key: str) -> Session | None:
        """Load a session from SQLite without consulting the in-memory cache."""
        with self._lock:
            return self._load_session_metadata_locked(key)

    def _make_lazy_messages(
        self,
        key: str,
        *,
        total_count: int,
        loaded_messages: list[dict[str, Any]] | None = None,
    ) -> LazyMessageList:
        return LazyMessageList(
            self,
            key,
            total_count=total_count,
            loaded_messages=loaded_messages,
        )

    def _load_session_metadata_locked(self, key: str) -> Session | None:
        row = self._conn.execute(
            """
            SELECT key, created_at, updated_at, metadata_json, message_count, last_consolidated_seq, revision
            FROM session
            WHERE key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        row = self._repair_session_counts_locked(row)

        compactions: list[dict[str, Any]] = []
        for compaction_row in self._conn.execute(
            """
            SELECT boundary_seq, summary, tokens_before, file_ops_json, previous_summary, created_at
            FROM compaction
            WHERE session_key = ?
            ORDER BY id ASC
            """,
            (key,),
        ).fetchall():
            try:
                file_ops = json.loads(compaction_row["file_ops_json"] or "{}")
            except json.JSONDecodeError:
                file_ops = {}
            compactions.append(
                {
                    "summary": compaction_row["summary"],
                    "first_kept_index": max(0, int(compaction_row["boundary_seq"])),
                    "tokens_before": max(0, int(compaction_row["tokens_before"])),
                    "file_ops": file_ops if isinstance(file_ops, dict) else {},
                    "previous_summary": compaction_row["previous_summary"],
                    "timestamp": compaction_row["created_at"],
                }
            )

        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}

        created_at = self._parse_dt(row["created_at"])
        updated_at = self._parse_dt(row["updated_at"])
        try:
            message_count = int(row["message_count"])
        except (TypeError, ValueError):
            message_count = 0
        message_count = max(0, message_count)
        try:
            last_consolidated = int(row["last_consolidated_seq"])
        except (TypeError, ValueError):
            last_consolidated = 0
        last_consolidated = max(0, min(last_consolidated, message_count))

        for compaction in compactions:
            compaction["first_kept_index"] = max(
                0,
                min(int(compaction.get("first_kept_index", 0)), message_count),
            )

        session = Session(
            key=key,
            compactions=compactions,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata if isinstance(metadata, dict) else {},
            last_consolidated=last_consolidated,
        )
        session.bind_path(self._get_session_path(key))
        session.messages = self._make_lazy_messages(key, total_count=message_count)
        session.mark_persisted()
        self._sync_session_tracking(session)
        return session

    def _migrate_jsonl_locked(self, key: str) -> Session | None:
        session = SessionManager._load(self, key)
        if session is None:
            return None

        original_jsonl_path = session._path
        migrated_at = datetime.now().isoformat()
        session.bind_path(self._get_session_path(key))
        self._rewrite_session_locked(
            session,
            jsonl_path=str(original_jsonl_path) if original_jsonl_path else None,
            jsonl_migrated_at=migrated_at,
            rewrite_messages=True,
            rewrite_compactions=True,
        )
        if original_jsonl_path and original_jsonl_path.exists():
            backup_path = original_jsonl_path.with_name(f"{original_jsonl_path.name}.bak")
            try:
                backup_path.unlink(missing_ok=True)
                original_jsonl_path.replace(backup_path)
            except Exception:
                logger.exception("Failed to rename migrated JSONL session {} to backup", key)
        session.messages = self._make_lazy_messages(key, total_count=len(session.messages))
        session.mark_persisted()
        self._sync_session_tracking(session)
        logger.info("Migrated session {} from JSONL to SQLite", key)
        return session

    def _bind_sqlite_session(self, session: Session) -> None:
        if getattr(session, "_sqlite_bound", False):
            return
        session.add_message = MethodType(self._session_add_message, session)
        session.append = MethodType(self._session_append, session)
        session.checkpoint = MethodType(self._session_checkpoint, session)
        session.append_compaction = MethodType(self._session_append_compaction, session)
        session.clear = MethodType(self._session_clear, session)
        session.get_message_count = MethodType(self._session_get_message_count, session)
        session.get_visible_message_count = MethodType(
            self._session_get_visible_message_count, session
        )
        session.get_visible_bounds = MethodType(self._session_get_visible_bounds, session)
        session.get_messages_slice = MethodType(self._session_get_messages_slice, session)
        session.get_message_at = MethodType(self._session_get_message_at, session)
        session.get_history = MethodType(self._session_get_history, session)
        session.detect_resume_state = MethodType(self._session_detect_resume_state, session)
        setattr(session, "_sqlite_bound", True)

    def _session_add_message(
        self, session: Session, role: str, content: str, **kwargs: Any
    ) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        session.checkpoint([msg])

    def _session_append(self, session: Session, msg: dict[str, Any]) -> None:
        session.checkpoint([msg])

    def _session_checkpoint(self, session: Session, msgs: list[dict[str, Any]]) -> None:
        if not msgs:
            return

        entries: list[dict[str, Any]] = []
        for msg in msgs:
            entry = session._normalize_checkpoint_entry(msg)
            if entry is not None:
                entries.append(entry)
        if not entries:
            return

        with self._lock:
            start_seq = len(session.messages)
            now = datetime.now()
            now_iso = now.isoformat()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    jsonl_path=str(session._path) if session._path else None,
                    bump_revision=False,
                )
                for seq, entry in enumerate(entries, start=start_seq):
                    columns = self._message_columns(entry, seq=seq, now_iso=now_iso)
                    self._conn.execute(
                        """
                        INSERT INTO message (
                            session_key,
                            seq,
                            role,
                            raw_json,
                            timestamp,
                            tool_call_id,
                            tool_name,
                            has_tool_calls,
                            pruned_at,
                            original_content_chars,
                            pruned_tokens_saved,
                            approx_tokens,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.key,
                            columns["seq"],
                            columns["role"],
                            columns["raw_json"],
                            columns["timestamp"],
                            columns["tool_call_id"],
                            columns["tool_name"],
                            columns["has_tool_calls"],
                            columns["pruned_at"],
                            columns["original_content_chars"],
                            columns["pruned_tokens_saved"],
                            columns["approx_tokens"],
                            columns["created_at"],
                            columns["updated_at"],
                        ),
                    )
                self._conn.execute(
                    """
                    UPDATE session
                    SET updated_at = ?, message_count = ?, revision = revision + 1
                    WHERE key = ?
                    """,
                    (now_iso, start_seq + len(entries), session.key),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        if isinstance(session.messages, LazyMessageList):
            session.messages.note_db_append(entries)
        else:
            session.messages.extend(entries)
        session.updated_at = now
        session._persisted_count = len(session.messages)
        self._sync_session_tracking(session)

    def _session_append_compaction(
        self,
        session: Session,
        *,
        summary: str,
        first_kept_index: int,
        tokens_before: int,
        file_ops: dict[str, list[str]] | None = None,
        previous_summary: str | None = None,
        timestamp: str | None = None,
    ) -> CompactionEntry:
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
        now = datetime.now()
        now_iso = now.isoformat()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    jsonl_path=str(session._path) if session._path else None,
                    preserve_existing_message_count=True,
                    bump_revision=False,
                )
                self._conn.execute(
                    """
                    INSERT INTO compaction (
                        session_key,
                        boundary_seq,
                        summary,
                        tokens_before,
                        file_ops_json,
                        previous_summary,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.key,
                        payload["first_kept_index"],
                        payload["summary"],
                        payload["tokens_before"],
                        json.dumps(payload["file_ops"], ensure_ascii=False, sort_keys=True),
                        payload["previous_summary"],
                        payload["timestamp"],
                    ),
                )
                self._conn.execute(
                    "UPDATE session SET updated_at = ?, revision = revision + 1 WHERE key = ?",
                    (now_iso, session.key),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        session.compactions.append(payload)
        session.updated_at = now
        session._persisted_compaction_count = len(session.compactions)
        session._persisted_compactions_sig = session._compactions_signature()
        self._sync_session_tracking(session)
        return entry

    def _session_clear(self, session: Session) -> None:
        now = datetime.now()
        now_iso = now.isoformat()
        metadata = dict(session.metadata)
        metadata.pop("usage_snapshot", None)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert_session_row_locked(
                    session,
                    updated_at=now_iso,
                    metadata=metadata,
                    message_count=0,
                    last_consolidated=0,
                    jsonl_path=str(session._path) if session._path else None,
                )
                self._conn.execute("DELETE FROM compaction WHERE session_key = ?", (session.key,))
                self._conn.execute("DELETE FROM message WHERE session_key = ?", (session.key,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        session.messages = []
        session.messages = self._make_lazy_messages(session.key, total_count=0, loaded_messages=[])
        session.compactions = []
        session.last_consolidated = 0
        session.metadata = metadata
        session.updated_at = now
        session._persisted_count = 0
        session._persisted_compaction_count = 0
        session._persisted_last_consolidated = 0
        session._persisted_metadata_sig = session._metadata_signature()
        session._persisted_compactions_sig = session._compactions_signature()
        self._sync_session_tracking(session)

    def _session_get_message_count(self, session: Session) -> int:
        with self._lock:
            if self._session_has_local_changes_locked(session):
                return len(session.messages)
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM message WHERE session_key = ?",
                (session.key,),
            ).fetchone()
            return int(row["count"]) if row is not None else 0

    def _session_get_visible_message_count(self, session: Session) -> int:
        with self._lock:
            if self._session_has_local_changes_locked(session):
                return Session.get_visible_message_count(session)
            total_count = session.get_message_count()
            boundary = self._get_compaction_boundary(session.key)
            return max(0, total_count - boundary)

    def _session_get_visible_bounds(self, session: Session) -> tuple[int, int]:
        with self._lock:
            if self._session_has_local_changes_locked(session):
                return Session.get_visible_bounds(session)
            total_count = session.get_message_count()
            boundary = self._get_compaction_boundary(session.key)
            start = max(0, min(boundary, total_count))
            return start, total_count

    def _session_get_messages_slice(
        self,
        session: Session,
        start: int = 0,
        end: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._session_has_local_changes_locked(session):
                return Session.get_messages_slice(session, start, end)
            total_count = session.get_message_count()
            safe_start = max(0, min(int(start), total_count))
            safe_end = total_count if end is None else max(safe_start, min(int(end), total_count))
            if safe_end <= safe_start:
                return []
            rows = self._conn.execute(
                """
                SELECT raw_json
                FROM message
                WHERE session_key = ? AND seq >= ? AND seq < ?
                ORDER BY seq ASC
                """,
                (session.key, safe_start, safe_end),
            ).fetchall()
            messages: list[dict[str, Any]] = []
            for row in rows:
                raw = row["raw_json"]
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed SQLite message for {} slice lookup: {}",
                        session.key,
                        exc,
                    )
                    continue
                if isinstance(payload, dict):
                    messages.append(payload)
            return messages

    def _session_get_message_at(self, session: Session, index: int) -> dict[str, Any] | None:
        if index < 0:
            return None
        with self._lock:
            if self._session_has_local_changes_locked(session):
                return Session.get_message_at(session, index)
            row = self._conn.execute(
                """
                SELECT raw_json
                FROM message
                WHERE session_key = ? AND seq = ?
                """,
                (session.key, int(index)),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = json.loads(row["raw_json"])
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed SQLite message for {} exact lookup: {}",
                    session.key,
                    exc,
                )
                return None
            return payload if isinstance(payload, dict) else None

    def _rewrite_session_locked(
        self,
        session: Session,
        *,
        jsonl_path: str | None = None,
        jsonl_migrated_at: str | None = None,
        rewrite_messages: bool = True,
        rewrite_compactions: bool = True,
    ) -> None:
        now = datetime.now()
        session.updated_at = now
        now_iso = now.isoformat()

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._upsert_session_row_locked(
                session,
                updated_at=now_iso,
                jsonl_path=jsonl_path,
                jsonl_migrated_at=jsonl_migrated_at,
                preserve_existing_message_count=not rewrite_messages,
            )

            if rewrite_messages:
                # CRITICAL: Hydrate lazy messages BEFORE deleting rows.
                # LazyMessageList._ensure_loaded() queries the same table;
                # if we DELETE first, the lazy load returns 0 rows and we
                # commit an empty message table.
                if isinstance(session.messages, LazyMessageList) and not session.messages.is_loaded:
                    session.messages._ensure_loaded()
                self._conn.execute("DELETE FROM message WHERE session_key = ?", (session.key,))
                for seq, message in enumerate(session.messages):
                    columns = self._message_columns(message, seq=seq, now_iso=now_iso)
                    self._conn.execute(
                        """
                        INSERT INTO message (
                            session_key,
                            seq,
                            role,
                            raw_json,
                            timestamp,
                            tool_call_id,
                            tool_name,
                            has_tool_calls,
                            pruned_at,
                            original_content_chars,
                            pruned_tokens_saved,
                            approx_tokens,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.key,
                            columns["seq"],
                            columns["role"],
                            columns["raw_json"],
                            columns["timestamp"],
                            columns["tool_call_id"],
                            columns["tool_name"],
                            columns["has_tool_calls"],
                            columns["pruned_at"],
                            columns["original_content_chars"],
                            columns["pruned_tokens_saved"],
                            columns["approx_tokens"],
                            columns["created_at"],
                            columns["updated_at"],
                        ),
                    )

            if rewrite_compactions:
                self._conn.execute("DELETE FROM compaction WHERE session_key = ?", (session.key,))
                for compaction in session.compactions:
                    entry = CompactionEntry.from_dict(compaction)
                    payload = entry.to_dict()
                    self._conn.execute(
                        """
                        INSERT INTO compaction (
                            session_key,
                            boundary_seq,
                            summary,
                            tokens_before,
                            file_ops_json,
                            previous_summary,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.key,
                            payload["first_kept_index"],
                            payload["summary"],
                            payload["tokens_before"],
                            json.dumps(payload["file_ops"], ensure_ascii=False, sort_keys=True),
                            payload["previous_summary"],
                            payload["timestamp"],
                        ),
                    )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        session.mark_persisted()
        self._sync_session_tracking(session)

    def _upsert_session_row_locked(
        self,
        session: Session,
        *,
        updated_at: str,
        metadata: dict[str, Any] | None = None,
        message_count: int | None = None,
        last_consolidated: int | None = None,
        jsonl_path: str | None = None,
        jsonl_migrated_at: str | None = None,
        preserve_existing_message_count: bool = False,
        bump_revision: bool = True,
    ) -> None:
        existing = self._conn.execute(
            "SELECT message_count, revision FROM session WHERE key = ?",
            (session.key,),
        ).fetchone()
        created_at = session.created_at.isoformat()
        metadata_json = json.dumps(
            metadata if metadata is not None else session.metadata,
            ensure_ascii=False,
            sort_keys=True,
        )
        resolved_message_count: int
        if message_count is not None:
            resolved_message_count = int(message_count)
        elif preserve_existing_message_count:
            resolved_message_count = (
                int(existing["message_count"])
                if existing is not None and existing["message_count"] is not None
                else len(session.messages)
            )
        else:
            resolved_message_count = len(session.messages)
        existing_revision = int(existing["revision"]) if existing is not None else 0
        resolved_revision = existing_revision + 1 if bump_revision else existing_revision
        resolved_last_consolidated = (
            session.last_consolidated
            if last_consolidated is None
            else int(last_consolidated)
        )
        self._conn.execute(
            """
            INSERT INTO session (
                key,
                created_at,
                updated_at,
                metadata_json,
                message_count,
                last_consolidated_seq,
                revision,
                storage_version,
                jsonl_path,
                jsonl_migrated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json,
                message_count = excluded.message_count,
                last_consolidated_seq = excluded.last_consolidated_seq,
                revision = excluded.revision,
                storage_version = excluded.storage_version,
                jsonl_path = COALESCE(excluded.jsonl_path, session.jsonl_path),
                jsonl_migrated_at = COALESCE(excluded.jsonl_migrated_at, session.jsonl_migrated_at)
            """,
            (
                session.key,
                created_at,
                updated_at,
                metadata_json,
                max(0, resolved_message_count),
                max(0, resolved_last_consolidated),
                max(0, resolved_revision),
                jsonl_path,
                jsonl_migrated_at,
            ),
        )

    def _repair_session_counts_locked(self, row: sqlite3.Row) -> sqlite3.Row:
        """Repair derived session counters against canonical message rows."""
        key = str(row["key"])
        actual_count_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM message WHERE session_key = ?",
            (key,),
        ).fetchone()
        actual_count = int(actual_count_row["count"]) if actual_count_row is not None else 0
        stored_count = int(row["message_count"]) if row["message_count"] is not None else 0
        stored_last_consolidated = (
            int(row["last_consolidated_seq"]) if row["last_consolidated_seq"] is not None else 0
        )
        repaired_last_consolidated = max(0, min(stored_last_consolidated, actual_count))

        if stored_count != actual_count or stored_last_consolidated != repaired_last_consolidated:
            self._conn.execute(
                """
                UPDATE session
                SET message_count = ?, last_consolidated_seq = ?, revision = revision + 1
                WHERE key = ?
                """,
                (actual_count, repaired_last_consolidated, key),
            )
            self._conn.execute(
                """
                UPDATE compaction
                SET boundary_seq = ?
                WHERE session_key = ? AND boundary_seq > ?
                """,
                (actual_count, key, actual_count),
            )
            self._conn.commit()
            logger.warning(
                "Repaired SQLite session counters for {}: message_count {} -> {}, "
                "last_consolidated {} -> {}",
                key,
                stored_count,
                actual_count,
                stored_last_consolidated,
                repaired_last_consolidated,
            )
            refreshed = self._conn.execute(
                """
                SELECT key, created_at, updated_at, metadata_json, message_count, last_consolidated_seq, revision
                FROM session
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
            if refreshed is not None:
                return refreshed

        return row

    def _messages_changed(self, session: Session) -> bool:
        current_sig = self._messages_signature(session.messages)
        if getattr(session, "_sqlite_message_sig", "") != current_sig:
            return True
        if isinstance(session.messages, LazyMessageList) and not session.messages.is_loaded:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM message WHERE session_key = ?",
            (session.key,),
        ).fetchone()
        count = int(row["count"]) if row is not None else 0
        return count != len(session.messages)

    def _compactions_changed(self, session: Session) -> bool:
        current_sig = session._compactions_signature()
        if getattr(session, "_sqlite_compaction_sig", "") != current_sig:
            return True
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM compaction WHERE session_key = ?",
            (session.key,),
        ).fetchone()
        count = int(row["count"]) if row is not None else 0
        return count != len(session.compactions)

    def _sync_session_tracking(self, session: Session) -> None:
        setattr(session, "_sqlite_message_sig", self._messages_signature(session.messages))
        setattr(session, "_sqlite_compaction_sig", session._compactions_signature())
        setattr(session, "_sqlite_revision", self._session_revision_locked(session.key))
        setattr(session, "_sqlite_row_fingerprint", self._session_row_fingerprint_locked(session.key))

    def _session_has_local_changes_locked(self, session: Session) -> bool:
        return (
            getattr(session, "_sqlite_message_sig", "") != self._messages_signature(session.messages)
            or getattr(session, "_sqlite_compaction_sig", "") != session._compactions_signature()
            or session._persisted_metadata_sig != session._metadata_signature()
            or session._persisted_last_consolidated != session.last_consolidated
        )

    def _cached_session_needs_reload_locked(self, session: Session) -> bool:
        persisted_revision = self._session_revision_locked(session.key)
        cached_revision = getattr(session, "_sqlite_revision", None)
        if (
            persisted_revision is not None
            and cached_revision is not None
            and persisted_revision != cached_revision
        ):
            if self._session_has_local_changes_locked(session):
                logger.warning(
                    "Detected external SQLite session revision update for {} but kept cached state "
                    "because the in-memory session has local changes",
                    session.key,
                )
                return False
            return True
        persisted = self._session_row_fingerprint_locked(session.key)
        cached = getattr(session, "_sqlite_row_fingerprint", None)
        if persisted is None or cached is None or persisted == cached:
            return False
        if self._session_has_local_changes_locked(session):
            logger.warning(
                "Detected external SQLite session update for {} but kept cached state "
                "because the in-memory session has local changes",
                session.key,
            )
            return False
        return True

    def _session_revision_locked(self, key: str) -> int | None:
        row = self._conn.execute(
            "SELECT revision FROM session WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return int(row["revision"]) if row["revision"] is not None else 0

    def _session_row_fingerprint_locked(self, key: str) -> tuple[str, int, int] | None:
        row = self._conn.execute(
            """
            SELECT updated_at, message_count, last_consolidated_seq
            FROM session
            WHERE key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        return (
            str(row["updated_at"] or ""),
            int(row["message_count"] or 0),
            int(row["last_consolidated_seq"] or 0),
        )

    @staticmethod
    def _messages_signature(messages: list[dict[str, Any]]) -> str:
        if isinstance(messages, LazyMessageList):
            if messages.is_loaded:
                try:
                    return json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
                except TypeError:
                    return repr(list(messages))
            return messages.signature()
        try:
            return json.dumps(messages, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return repr(messages)

    def _load_messages(
        self,
        session_key: str,
        from_seq: int = 0,
        *,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        order = "DESC" if descending else "ASC"
        sql = [
            """
            SELECT raw_json
            FROM message
            WHERE session_key = ? AND seq >= ?
            ORDER BY seq
            """.replace("ORDER BY seq", f"ORDER BY seq {order}")
        ]
        params: list[Any] = [session_key, max(0, int(from_seq))]
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(max(0, int(limit)))
        rows = self._conn.execute(" ".join(sql), tuple(params)).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            raw = row["raw_json"]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed SQLite message for {}: {}", session_key, exc)
                continue
            if isinstance(payload, dict):
                messages.append(payload)
        return messages

    def _load_history_messages(
        self,
        session_key: str,
        from_seq: int = 0,
        *,
        descending: bool = False,
        render_pruned_placeholders: bool = True,
    ) -> list[dict[str, Any]]:
        order = "DESC" if descending else "ASC"
        rows = self._conn.execute(
            """
            SELECT seq, raw_json, timestamp, tool_name, pruned_at, original_content_chars
            FROM message
            WHERE session_key = ? AND seq >= ?
            ORDER BY seq
            """.replace("ORDER BY seq", f"ORDER BY seq {order}"),
            (session_key, max(0, int(from_seq))),
        ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["raw_json"])
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed SQLite message for {}: {}", session_key, exc)
                continue
            if not isinstance(payload, dict):
                continue
            if render_pruned_placeholders and row["pruned_at"] and payload.get("role") == "tool":
                payload = self._render_pruned_tool_placeholder(row, payload)
            messages.append(payload)
        return messages

    @staticmethod
    def _render_pruned_tool_placeholder(
        row: sqlite3.Row,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        rendered = dict(payload)
        tool_name = str(payload.get("name") or row["tool_name"] or "tool")
        try:
            seq = int(row["seq"])
        except (TypeError, ValueError):
            seq = -1
        original_chars = int(
            row["original_content_chars"] or _tool_content_chars(payload.get("content"))
        )
        rendered["content"] = _tool_prune_placeholder(
            tool_name=tool_name,
            message_index=seq if seq >= 0 else None,
            timestamp=payload.get("timestamp") if isinstance(payload.get("timestamp"), str) else row["timestamp"],
            original_chars=original_chars,
        )
        return rendered

    def _get_compaction_boundary(self, session_key: str) -> int:
        row = self._conn.execute(
            """
            SELECT boundary_seq
            FROM compaction
            WHERE session_key = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_key,),
        ).fetchone()
        if row is None:
            return 0
        try:
            return max(0, int(row["boundary_seq"]))
        except (TypeError, ValueError):
            return 0

    def _get_prune_stats(
        self,
        session_key: str,
        *,
        protected_tools: set[str] | None = None,
    ) -> PruneResult:
        protected = {
            str(name)
            for name in (
                protected_tools if protected_tools is not None else _DEFAULT_PROTECTED_TOOLS
            )
        }
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS pruned,
                COALESCE(SUM(pruned_tokens_saved), SUM(original_content_chars), 0) AS tokens_saved
            FROM message
            WHERE session_key = ? AND role = 'tool' AND pruned_at IS NOT NULL
            """,
            (session_key,),
        ).fetchone()
        total_tools_row = self._conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM message
            WHERE session_key = ? AND role = 'tool'
            """,
            (session_key,),
        ).fetchone()
        protected_count = 0
        if protected:
            protected_row = self._conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM message
                WHERE session_key = ?
                  AND role = 'tool'
                  AND pruned_at IS NULL
                  AND tool_name IN ({})
                """.format(",".join("?" for _ in protected)),
                (session_key, *protected),
            ).fetchone()
            protected_count = int(protected_row["count"]) if protected_row is not None else 0
        return PruneResult(
            messages_pruned=int(row["pruned"]) if row is not None else 0,
            tokens_saved=int(row["tokens_saved"]) if row is not None else 0,
            messages_protected=protected_count,
            total_tool_messages=int(total_tools_row["count"]) if total_tools_row is not None else 0,
        )

    def _apply_pruned_rows_to_cache(
        self,
        session_key: str,
        replacements: list[tuple[int, dict[str, Any]]],
    ) -> None:
        session = self._cache.get(session_key)
        if session is None or not replacements:
            return
        messages = session.messages
        if isinstance(messages, LazyMessageList):
            for seq, payload in replacements:
                messages.note_db_replace(seq, payload)
            return
        for seq, payload in replacements:
            if 0 <= seq < len(messages):
                messages[seq] = payload

    def _session_get_history(
        self,
        session: Session,
        max_messages: int = 500,
        prune_tool_results: bool = True,
        prune_protect_tokens: int | None = None,
        prune_minimum_tokens: int | None = None,
        context_window: int = 200_000,
        protected_tools: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], PruneResult | None]:
        _ = (prune_protect_tokens, prune_minimum_tokens, context_window, protected_tools)
        with self._lock:
            boundary_seq = self._get_compaction_boundary(session.key)
            prompt_window = self._load_history_messages(
                session.key,
                from_seq=boundary_seq,
                render_pruned_placeholders=prune_tool_results,
            )

        sliced = prompt_window[-max_messages:]

        if boundary_seq == 0:
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

        out = _sanitize_tool_pairs(out)
        prune_result = (
            self._get_prune_stats(session.key, protected_tools=protected_tools)
            if prune_tool_results
            else None
        )
        return out, prune_result

    def _session_detect_resume_state(self, session: Session) -> str:
        """Inspect raw tail rows in SQLite and classify restart state."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, raw_json
                FROM message
                WHERE session_key = ?
                ORDER BY seq DESC
                LIMIT 10
                """,
                (session.key,),
            ).fetchall()

        for row in rows:
            try:
                payload = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            role = payload.get("role") or row["role"]
            if role == "assistant":
                tool_calls = payload.get("tool_calls") or []
                return "mid_tool" if tool_calls else "clean"
            if role == "tool":
                return "mid_loop"
            return "clean"
        return "clean"

    def prune_old_tool_results(
        self,
        session_key: str,
        *,
        context_window: int = 200_000,
        prune_protect_tokens: int | None = None,
        prune_minimum_tokens: int | None = None,
        protected_tools: set[str] | None = None,
    ) -> PruneResult:
        from nanobot.session.compaction import estimate_message_tokens

        effective_protect = (
            prune_protect_tokens
            if prune_protect_tokens is not None
            else _PROMPT_PRUNE_PROTECT_TOKENS
        )
        effective_minimum = (
            prune_minimum_tokens
            if prune_minimum_tokens is not None
            else _PROMPT_PRUNE_MINIMUM_TOKENS
        )
        protected = {
            str(name)
            for name in (
                protected_tools if protected_tools is not None else _DEFAULT_PROTECTED_TOOLS
            )
        }

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, seq, role, raw_json, tool_name, timestamp, pruned_at, original_content_chars
                FROM message
                WHERE session_key = ? AND role IN ('user', 'tool')
                ORDER BY seq DESC
                """,
                (session_key,),
            ).fetchall()

            turns_seen = 0
            total_tool_tokens = 0
            total_tool_messages = 0
            messages_protected = 0
            candidates: list[tuple[int, int, dict[str, Any], int, int]] = []

            for row in rows:
                role = row["role"]
                if role == "user":
                    turns_seen += 1
                    continue
                total_tool_messages += 1
                if turns_seen < 2:
                    continue
                if row["pruned_at"]:
                    break
                try:
                    payload = json.loads(row["raw_json"])
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed tool message for {}: {}", session_key, exc)
                    continue
                if not isinstance(payload, dict):
                    continue
                tool_name = payload.get("name")
                if isinstance(tool_name, str) and tool_name in protected:
                    messages_protected += 1
                    continue
                tool_tokens = estimate_message_tokens(payload)
                total_tool_tokens += tool_tokens
                if total_tool_tokens <= max(0, int(effective_protect)):
                    continue
                original_chars = _tool_content_chars(payload.get("content"))
                placeholder_payload = self._render_pruned_tool_placeholder(row, payload)
                saved = max(0, tool_tokens - estimate_message_tokens(placeholder_payload))
                if saved <= 0:
                    continue
                candidates.append(
                    (int(row["id"]), int(row["seq"]), placeholder_payload, original_chars, saved)
                )

            total_saved = sum(saved for *_rest, saved in candidates)

            if total_saved < max(0, int(effective_minimum)) or not candidates:
                return PruneResult(
                    messages_pruned=0,
                    tokens_saved=0,
                    messages_protected=messages_protected,
                    total_tool_messages=total_tool_messages,
                )

            tool_rows = self._conn.execute(
                """
                SELECT seq, raw_json
                FROM message
                WHERE session_key = ? AND role = 'tool'
                ORDER BY seq ASC
                """,
                (session_key,),
            ).fetchall()
            visible_messages = []
            for tool_row in tool_rows:
                try:
                    payload = json.loads(tool_row["raw_json"])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    visible_messages.append(payload)
            estimated_before = sum(estimate_message_tokens(msg) for msg in visible_messages)
            estimated_after = max(0, estimated_before - total_saved)

            now_iso = datetime.now().isoformat()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                prune_items: list[dict[str, Any]] = []
                for message_id, seq, payload, original_chars, _saved in candidates:
                    self._conn.execute(
                        """
                        UPDATE message
                        SET pruned_at = ?,
                            original_content_chars = ?,
                            pruned_tokens_saved = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (now_iso, original_chars, _saved, now_iso, message_id),
                    )
                    prune_items.append({
                        "message_seq": seq,
                        "tool_call_id": payload.get("tool_call_id") if isinstance(payload.get("tool_call_id"), str) else None,
                        "tool_name": payload.get("name") if isinstance(payload.get("name"), str) else None,
                        "replacement_kind": "tool_output_placeholder",
                        "original_content_chars": original_chars,
                        "estimated_tokens_removed": _saved,
                    })
                session = self.get_or_create(session_key)
                turn = max(0, int(getattr(self, '_usage_turn_for_session', lambda _k: 0)(session_key)))
                iteration = getattr(self, '_usage_iteration_for_session', lambda _k: None)(session_key)
                trigger_call_index = getattr(self, '_last_provider_call_index_for_session', lambda _k: None)(session_key)
                tool_results_pruned = sum(1 for item in prune_items if item.get("tool_name"))
                cursor = self._conn.execute(
                    """
                    INSERT INTO tool_prune_event (
                        session_key,
                        turn,
                        iteration,
                        trigger_call_index,
                        reason,
                        estimated_tokens_before,
                        estimated_tokens_after,
                        estimated_tokens_saved,
                        messages_pruned,
                        tool_results_pruned,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.key,
                        turn,
                        int(iteration) if iteration is not None else None,
                        int(trigger_call_index) if trigger_call_index is not None else None,
                        "post_response_tool_prune",
                        estimated_before,
                        estimated_after,
                        total_saved,
                        len(prune_items),
                        tool_results_pruned,
                        now_iso,
                    ),
                )
                prune_event_id = int(cursor.lastrowid)
                for item in prune_items:
                    self._conn.execute(
                        """
                        INSERT INTO tool_prune_item (
                            prune_event_id,
                            session_key,
                            message_seq,
                            tool_call_id,
                            tool_name,
                            replacement_kind,
                            original_content_chars,
                            estimated_tokens_removed,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prune_event_id,
                            session.key,
                            int(item["message_seq"]),
                            item.get("tool_call_id"),
                            item.get("tool_name"),
                            item.get("replacement_kind"),
                            int(item.get("original_content_chars", 0) or 0),
                            int(item.get("estimated_tokens_removed", 0) or 0),
                            now_iso,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return PruneResult(
            messages_pruned=len(candidates),
            tokens_saved=total_saved,
            messages_protected=messages_protected,
            total_tool_messages=total_tool_messages,
        )

    @staticmethod
    def _message_columns(msg: dict[str, Any], *, seq: int, now_iso: str) -> dict[str, Any]:
        role = str(msg.get("role") or "user")
        tool_calls = msg.get("tool_calls")
        has_tool_calls = int(isinstance(tool_calls, list) and bool(tool_calls))

        tool_call_id: str | None = None
        tool_name: str | None = None
        if role == "tool":
            raw_tool_call_id = msg.get("tool_call_id")
            tool_call_id = raw_tool_call_id if isinstance(raw_tool_call_id, str) else None
            raw_tool_name = msg.get("name")
            tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None
        elif has_tool_calls and isinstance(tool_calls, list):
            first = tool_calls[0] if tool_calls else None
            if isinstance(first, dict):
                raw_tool_call_id = first.get("id")
                tool_call_id = raw_tool_call_id if isinstance(raw_tool_call_id, str) else None
                fn = first.get("function")
                if isinstance(fn, dict):
                    raw_tool_name = fn.get("name")
                    tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None

        timestamp = msg.get("timestamp")
        return {
            "seq": seq,
            "role": role,
            "raw_json": json.dumps(msg, ensure_ascii=False, sort_keys=True),
            "timestamp": timestamp if isinstance(timestamp, str) else None,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "has_tool_calls": has_tool_calls,
            "pruned_at": None,
            "original_content_chars": (
                _tool_content_chars(msg.get("content")) if role == "tool" else None
            ),
            "pruned_tokens_saved": None,
            "approx_tokens": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

    def _usage_turn_for_session(self, session_key: str) -> int:
        row = self._conn.execute(
            "SELECT turn FROM provider_call WHERE session_key = ? ORDER BY call_index DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        return int(row["turn"]) if row is not None and row["turn"] is not None else 0

    def _usage_iteration_for_session(self, session_key: str) -> int | None:
        row = self._conn.execute(
            "SELECT iteration FROM provider_call WHERE session_key = ? ORDER BY call_index DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        if row is None or row["iteration"] is None:
            return None
        return int(row["iteration"])

    def _last_provider_call_index_for_session(self, session_key: str) -> int | None:
        row = self._conn.execute(
            "SELECT call_index FROM provider_call WHERE session_key = ? ORDER BY call_index DESC LIMIT 1",
            (session_key,),
        ).fetchone()
        if row is None or row["call_index"] is None:
            return None
        return int(row["call_index"])

    @staticmethod
    def _parse_dt(value: Any) -> datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now()
