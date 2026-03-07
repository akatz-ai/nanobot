"""Session Evidence Index — per-session SQLite FTS5 index for tool results."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
MIN_CONTENT_CHARS = 512

INDEXABLE_TOOLS = frozenset(
    {
        "exec",
        "read_file",
        "write_file",
        "edit_file",
        "web_fetch",
        "web_search",
        "list_dir",
        "batch",
    }
)

SKIP_TOOLS = frozenset(
    {
        "memory_recall",
        "memory_save",
        "memory_graph",
        "memory_forget",
        "memory_ingest",
        "memory_stats",
        "cron",
        "message",
        "spawn",
        "manage_agents",
        "session_search",
    }
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_index INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    timestamp TEXT,
    char_count INTEGER NOT NULL,
    metadata TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    content,
    tool_name,
    content='evidence',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
    INSERT INTO evidence_fts(rowid, content, tool_name)
    VALUES (new.id, new.content, new.tool_name);
END;

CREATE TRIGGER IF NOT EXISTS evidence_ad AFTER DELETE ON evidence BEGIN
    INSERT INTO evidence_fts(evidence_fts, rowid, content, tool_name)
    VALUES ('delete', old.id, old.content, old.tool_name);
END;
"""


def _chunk_content(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring newline boundaries."""
    if not text:
        return [""]
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    safe_overlap = min(max(0, overlap), max(chunk_size - 1, 0))
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            search_start = start + int(chunk_size * 0.8)
            newline_at = text.rfind("\n", search_start, end)
            if newline_at >= search_start:
                end = newline_at + 1

        if end <= start:
            end = min(len(text), start + chunk_size)

        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - safe_overlap)

    return chunks


def should_index_tool(tool_name: str, content: str) -> bool:
    """Decide whether a tool payload should be indexed."""
    if tool_name in SKIP_TOOLS:
        return False
    if tool_name not in INDEXABLE_TOOLS:
        return False
    return len(content) >= MIN_CONTENT_CHARS


def build_evidence_content(
    tool_name: str,
    result: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Build searchable content and metadata for a tool execution."""
    args = arguments or {}
    metadata: dict[str, Any] = {}
    blocks: list[str] = []

    def add_label(label: str, value: Any) -> None:
        if isinstance(value, str) and value:
            blocks.append(f"{label}: {value}")
            metadata[label.lower().replace(" ", "_")] = value

    def add_section(label: str, value: Any) -> None:
        if isinstance(value, str) and value:
            blocks.append(f"{label}:\n{value}")

    add_result = True

    if tool_name == "exec":
        add_label("Command", args.get("command"))
        add_label("Working directory", args.get("working_dir"))
        add_section("Output", result)
        add_result = False
    elif tool_name == "read_file":
        add_label("Path", args.get("path"))
        add_section("Content", result)
        add_result = False
    elif tool_name == "write_file":
        add_label("Path", args.get("path"))
        add_section("Written content", args.get("content"))
        add_result = False
    elif tool_name == "edit_file":
        add_label("Path", args.get("path"))
        add_section("Old text", args.get("old_text"))
        add_section("New text", args.get("new_text"))
        if result and result.startswith("Error"):
            add_section("Result", result)
        add_result = False
    elif tool_name == "web_fetch":
        add_label("URL", args.get("url"))
        add_section("Fetched content", result)
        add_result = False
    elif tool_name == "web_search":
        add_label("Query", args.get("query"))
        add_section("Results", result)
        add_result = False
    elif tool_name == "list_dir":
        add_label("Path", args.get("path"))
        add_section("Listing", result)
        add_result = False
    elif tool_name == "batch":
        add_section("Code", args.get("code"))
        add_section("Output", result)
        add_result = False

    if add_result and result:
        blocks.append(result)

    content = "\n\n".join(block for block in blocks if block).strip()
    return content or result, metadata or None


def _evidence_threshold_content(
    tool_name: str,
    result: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Return the substantive payload used for the hook's size threshold."""
    args = arguments or {}

    if tool_name == "write_file":
        content = args.get("content")
        return content if isinstance(content, str) else result

    if tool_name == "edit_file":
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        blocks = [
            value
            for value in (old_text, new_text)
            if isinstance(value, str) and value
        ]
        if result.startswith("Error"):
            blocks.append(result)
        return "\n\n".join(blocks) or result

    if tool_name == "batch":
        code = args.get("code")
        if isinstance(code, str) and code:
            return f"{code}\n\n{result}"

    return result


def _fts_query_candidates(query: str) -> list[str]:
    raw = query.strip()
    if not raw:
        return []

    sanitized_terms = re.findall(r"[A-Za-z0-9_]+", raw)
    if not sanitized_terms:
        return [raw]

    sanitized = " ".join(sanitized_terms)
    if sanitized == raw:
        return [raw]
    return [raw, sanitized]


class EvidenceIndex:
    """SQLite FTS5 wrapper for indexing tool results in a session."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        except Exception:
            logger.warning("Failed to initialize evidence index at {}", self.db_path)
            self._conn = None

    def add(
        self,
        message_index: int,
        tool_name: str,
        content: str,
        timestamp: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Index a tool result, chunking large content for better FTS matches."""
        if self._conn is None:
            return

        try:
            ts = timestamp or datetime.now().isoformat()
            meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
            char_count = len(content)

            for chunk_index, chunk in enumerate(_chunk_content(content)):
                self._conn.execute(
                    """
                    INSERT INTO evidence (
                        message_index, tool_name, chunk_index, content,
                        timestamp, char_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_index,
                        tool_name,
                        chunk_index,
                        chunk,
                        ts,
                        char_count,
                        meta_json,
                    ),
                )
            self._conn.commit()
        except Exception:
            logger.warning(
                "Failed to add evidence for tool={} msg_idx={}",
                tool_name,
                message_index,
            )

    def search(
        self,
        query: str,
        tool_filter: str | None = None,
        max_results: int = 5,
    ) -> list[dict]:
        """Run an FTS5 search and return ranked result dicts."""
        if self._conn is None or not query.strip():
            return []

        try:
            limit = max(1, int(max_results))
            sql = """
                SELECT
                    e.message_index,
                    e.tool_name,
                    e.timestamp,
                    e.char_count,
                    snippet(evidence_fts, 0, '>>>', '<<<', '...', 32) AS excerpt,
                    bm25(evidence_fts) AS rank
                FROM evidence_fts
                JOIN evidence e ON evidence_fts.rowid = e.id
                WHERE evidence_fts MATCH ?
            """
            for candidate in _fts_query_candidates(query):
                params: list[Any] = [candidate]
                if tool_filter:
                    sql_with_filter = sql + " AND e.tool_name = ?"
                    params.append(tool_filter)
                else:
                    sql_with_filter = sql
                sql_with_filter += " ORDER BY rank, e.message_index DESC, e.chunk_index ASC LIMIT ?"
                params.append(limit)

                try:
                    cursor = self._conn.execute(sql_with_filter, params)
                except sqlite3.OperationalError:
                    continue

                return [
                    {
                        "message_index": row[0],
                        "tool_name": row[1],
                        "timestamp": row[2],
                        "char_count": row[3],
                        "excerpt": row[4],
                        "rank": row[5],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception:
            logger.warning("Evidence search failed for query={!r}", query)
            return []

        logger.warning("Evidence search failed for query={!r}", query)
        return []

    def count(self) -> int:
        """Return the total number of indexed rows."""
        if self._conn is None:
            return 0
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM evidence")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        finally:
            self._conn = None


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def seed_evidence_index(session_jsonl_path: Path) -> EvidenceIndex | None:
    """Walk a session JSONL and build the sidecar evidence index once."""
    sqlite_path = session_jsonl_path.with_suffix(".evidence.sqlite")
    if sqlite_path.exists() or not session_jsonl_path.exists():
        return None

    index = EvidenceIndex(sqlite_path)
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    message_index = 0

    try:
        with open(session_jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue

                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                if not isinstance(message, dict):
                    continue
                if message.get("_type") in {"metadata", "compaction"}:
                    continue

                current_index = message_index
                message_index += 1

                if message.get("role") == "assistant":
                    for tool_call in message.get("tool_calls") or []:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = tool_call.get("id")
                        function_data = tool_call.get("function") or {}
                        if not isinstance(function_data, dict) or not isinstance(tool_call_id, str):
                            continue
                        tool_calls_by_id[tool_call_id] = _parse_tool_arguments(
                            function_data.get("arguments")
                        )
                    continue

                if message.get("role") != "tool":
                    continue

                tool_name = message.get("name")
                result = message.get("content")
                if not isinstance(tool_name, str) or not isinstance(result, str):
                    continue

                tool_call_id = message.get("tool_call_id")
                arguments = (
                    tool_calls_by_id.get(tool_call_id, {})
                    if isinstance(tool_call_id, str)
                    else {}
                )
                content, metadata = build_evidence_content(tool_name, result, arguments)
                threshold_content = _evidence_threshold_content(tool_name, result, arguments)
                if not should_index_tool(tool_name, threshold_content):
                    continue

                index.add(
                    message_index=current_index,
                    tool_name=tool_name,
                    content=content,
                    timestamp=message.get("timestamp"),
                    metadata=metadata,
                )
    except Exception:
        logger.warning("Failed to seed evidence index for {}", session_jsonl_path)

    return index
