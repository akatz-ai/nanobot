from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.agent.router import AgentRouter
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.session.manager import SessionManager
from nanobot.session.store import LazyMessageList, SQLiteSessionManager


def _fetch_all(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query, params).fetchall()


def _insert_message_rows(
    db_path: Path,
    *,
    session_key: str,
    start_seq: int,
    count: int,
) -> None:
    now = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        for seq in range(start_seq, start_seq + count):
            payload = {"role": "user", "content": f"ext-{seq}", "timestamp": now}
            conn.execute(
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
                    session_key,
                    seq,
                    "user",
                    json.dumps(payload),
                    now,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
        conn.commit()


def _populate_basic(session) -> None:
    session.checkpoint(
        [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
    )


def _tool_turn(turn_index: int, *, tool_name: str = "exec", output_chars: int = 18_000) -> list[dict[str, Any]]:
    tool_call_id = f"tc-{turn_index}"
    return [
        {"role": "user", "content": f"user-{turn_index}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": f"{tool_name}-{turn_index}-" + ("x" * output_chars),
        },
        {"role": "assistant", "content": f"assistant-{turn_index}"},
    ]


def _build_tool_session(manager, key: str) -> None:
    session = manager.get_or_create(key)
    for idx, tool_name in enumerate(["exec", "memory_recall", "exec", "exec", "exec"], start=1):
        session.checkpoint(_tool_turn(idx, tool_name=tool_name))


@pytest.fixture
def sqlite_manager(tmp_path: Path) -> SQLiteSessionManager:
    return SQLiteSessionManager(tmp_path)


def test_sqlite_store_initializes_schema(sqlite_manager: SQLiteSessionManager) -> None:
    tables = {
        row["name"]
        for row in _fetch_all(
            sqlite_manager.db_path,
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        )
    }
    indexes = {
        row["name"]
        for row in _fetch_all(
            sqlite_manager.db_path,
            "SELECT name FROM sqlite_master WHERE type = 'index'",
        )
    }
    journal_mode = _fetch_all(sqlite_manager.db_path, "PRAGMA journal_mode")[0][0]

    assert {"session", "message", "compaction"} <= tables
    assert {
        "idx_message_session_seq",
        "idx_message_session_role_seq",
        "idx_message_session_tool_call_id",
        "idx_message_session_unpruned_tools",
        "idx_compaction_session_latest",
        "idx_compaction_session_boundary",
    } <= indexes
    assert str(journal_mode).lower() == "wal"


def test_get_or_create_creates_session_row(sqlite_manager: SQLiteSessionManager) -> None:
    session = sqlite_manager.get_or_create("cli:new")
    rows = _fetch_all(
        sqlite_manager.db_path,
        "SELECT key, message_count, last_consolidated_seq FROM session WHERE key = ?",
        (session.key,),
    )

    assert len(rows) == 1
    assert rows[0]["key"] == "cli:new"
    assert rows[0]["message_count"] == 0
    assert rows[0]["last_consolidated_seq"] == 0


def test_checkpoint_inserts_messages_with_monotonic_seq(sqlite_manager: SQLiteSessionManager) -> None:
    session = sqlite_manager.get_or_create("cli:seq")
    session.checkpoint(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    session.append({"role": "user", "content": "again"})

    rows = _fetch_all(
        sqlite_manager.db_path,
        "SELECT seq, role, raw_json FROM message WHERE session_key = ? ORDER BY seq ASC",
        (session.key,),
    )

    assert [row["seq"] for row in rows] == [0, 1, 2]
    assert [row["role"] for row in rows] == ["user", "assistant", "user"]
    assert json.loads(rows[2]["raw_json"])["content"] == "again"


def test_get_or_create_reloads_messages_from_sqlite(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:reload"
    session = sqlite_manager.get_or_create(key)
    _populate_basic(session)

    sqlite_manager.invalidate(key)
    loaded = sqlite_manager.get_or_create(key)

    assert isinstance(loaded.messages, LazyMessageList)
    assert not loaded.messages.is_loaded
    assert [msg["content"] for msg in loaded.messages] == ["u0", "a0", "u1", "a1"]


def test_compaction_persists_and_get_history_respects_boundary(
    sqlite_manager: SQLiteSessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "cli:compact"
    session = sqlite_manager.get_or_create(key)
    batch = []
    for index in range(1_000):
        role = "user" if index % 2 == 0 else "assistant"
        batch.append({"role": role, "content": f"msg-{index}"})
    session.checkpoint(batch)
    session.append_compaction(
        summary="summary",
        first_kept_index=900,
        tokens_before=200,
        file_ops={"read_files": ["a"], "modified_files": ["b"]},
    )

    sqlite_manager.invalidate(key)
    loaded = sqlite_manager.get_or_create(key)
    seen: dict[str, int] = {}
    original_load_messages = sqlite_manager._load_messages

    def _recorded_load_messages(
        session_key: str,
        from_seq: int = 0,
        *,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        seen["from_seq"] = from_seq
        return original_load_messages(
            session_key,
            from_seq=from_seq,
            limit=limit,
            descending=descending,
        )

    monkeypatch.setattr(sqlite_manager, "_load_messages", _recorded_load_messages)
    assert isinstance(loaded.messages, LazyMessageList)
    assert not loaded.messages.is_loaded
    history, _ = loaded.get_history(max_messages=50)
    compaction_rows = _fetch_all(
        sqlite_manager.db_path,
        "SELECT boundary_seq, tokens_before FROM compaction WHERE session_key = ?",
        (key,),
    )

    assert seen["from_seq"] == 900
    assert not loaded.messages.is_loaded
    assert len(history) == 50
    assert history[0]["content"] == "msg-950"
    assert history[-1]["content"] == "msg-999"
    assert len(compaction_rows) == 1
    assert compaction_rows[0]["boundary_seq"] == 900
    assert compaction_rows[0]["tokens_before"] == 200


def test_tool_pruning_matches_jsonl_behavior(tmp_path: Path) -> None:
    sqlite_manager = SQLiteSessionManager(tmp_path / "sqlite")
    jsonl_manager = SessionManager(tmp_path / "jsonl")
    key = "cli:prune"

    _build_tool_session(sqlite_manager, key)
    _build_tool_session(jsonl_manager, key)

    sqlite_manager.prune_old_tool_results(
        key,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    sqlite_history, sqlite_prune = sqlite_manager.get_or_create(key).get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    jsonl_history, jsonl_prune = jsonl_manager.get_or_create(key).get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    assert sqlite_history == jsonl_history
    assert sqlite_prune is not None
    assert jsonl_prune is not None
    assert sqlite_prune.messages_pruned == jsonl_prune.messages_pruned


def test_detect_resume_state_variants(sqlite_manager: SQLiteSessionManager) -> None:
    clean = sqlite_manager.get_or_create("cli:clean")
    assert clean.detect_resume_state() == "clean"

    clean.checkpoint(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert clean.detect_resume_state() == "clean"

    mid_tool = sqlite_manager.get_or_create("cli:mid-tool")
    mid_tool.checkpoint(
        [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert mid_tool.detect_resume_state() == "mid_tool"

    mid_loop = sqlite_manager.get_or_create("cli:mid-loop")
    mid_loop.checkpoint(
        [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "tc-2",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc-2", "name": "list_dir", "content": "[]"},
        ]
    )
    sqlite_manager.invalidate("cli:mid-loop")
    mid_loop = sqlite_manager.get_or_create("cli:mid-loop")
    assert isinstance(mid_loop.messages, LazyMessageList)
    assert not mid_loop.messages.is_loaded
    assert mid_loop.detect_resume_state() == "mid_loop"
    assert not mid_loop.messages.is_loaded


def test_jsonl_session_is_auto_migrated_on_first_access(tmp_path: Path) -> None:
    jsonl_manager = SessionManager(tmp_path)
    key = "cli:migrate"
    session = jsonl_manager.get_or_create(key)
    _populate_basic(session)
    session.metadata["origin_channel"] = "discord"
    session.last_consolidated = 2
    jsonl_manager.save(session)

    sqlite_manager = SQLiteSessionManager(tmp_path)
    migrated = sqlite_manager.get_or_create(key)
    rows = _fetch_all(
        sqlite_manager.db_path,
        "SELECT jsonl_path, jsonl_migrated_at, message_count FROM session WHERE key = ?",
        (key,),
    )

    assert [msg["content"] for msg in migrated.messages] == ["u0", "a0", "u1", "a1"]
    assert rows[0]["jsonl_path"].endswith("cli_migrate.jsonl")
    assert rows[0]["jsonl_migrated_at"] is not None
    assert rows[0]["message_count"] == 4


def test_jsonl_migration_preserves_session_content_exactly(tmp_path: Path) -> None:
    key = "cli:parity"
    jsonl_manager = SessionManager(tmp_path)
    source = jsonl_manager.get_or_create(key)
    source.checkpoint(
        [
            {"role": "system", "content": "bootstrap"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc-1", "name": "exec", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
    )
    source.metadata["origin_channel"] = "discord"
    source.metadata["origin_chat_id"] = "thread-42"
    source.last_consolidated = 3
    source.append_compaction(
        summary="summary",
        first_kept_index=1,
        tokens_before=50,
        file_ops={"read_files": ["a"], "modified_files": ["b"]},
    )
    jsonl_manager.save(source)
    jsonl_manager.invalidate(key)
    baseline = jsonl_manager.get_or_create(key)

    sqlite_manager = SQLiteSessionManager(tmp_path)
    sqlite_manager.invalidate(key)
    migrated = sqlite_manager.get_or_create(key)
    sqlite_manager.invalidate(key)
    reloaded = sqlite_manager.get_or_create(key)

    assert reloaded.messages == baseline.messages
    assert reloaded.compactions == baseline.compactions
    assert reloaded.metadata == baseline.metadata
    assert reloaded.last_consolidated == baseline.last_consolidated
    assert reloaded.created_at == baseline.created_at
    assert reloaded.updated_at.isoformat() == migrated.updated_at.isoformat()


def test_save_persists_metadata_and_last_consolidated(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:metadata"
    session = sqlite_manager.get_or_create(key)
    _populate_basic(session)
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "room-1"
    session.last_consolidated = 3
    sqlite_manager.save(session)

    sqlite_manager.invalidate(key)
    loaded = sqlite_manager.get_or_create(key)

    assert loaded.metadata["origin_channel"] == "discord"
    assert loaded.metadata["origin_chat_id"] == "room-1"
    assert loaded.last_consolidated == 3


def test_list_sessions_returns_sqlite_sessions(sqlite_manager: SQLiteSessionManager) -> None:
    a = sqlite_manager.get_or_create("cli:a")
    a.add_message("user", "one")
    sqlite_manager.save(a)

    b = sqlite_manager.get_or_create("cli:b")
    b.add_message("user", "two")
    sqlite_manager.save(b)

    sessions = sqlite_manager.list_sessions()
    session_keys = {item["key"] for item in sessions}

    assert {"cli:a", "cli:b"} <= session_keys
    assert all(str(item["path"]).endswith(".jsonl") for item in sessions if item["key"].startswith("cli:"))


def test_orphaned_tool_pairs_are_sanitized_after_jsonl_migration(tmp_path: Path) -> None:
    key = "cli:orphaned"
    jsonl_manager = SessionManager(tmp_path)
    session = jsonl_manager.get_or_create(key)
    session.checkpoint(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "other-call", "name": "exec", "content": "orphan"},
            {"role": "assistant", "content": "final"},
        ]
    )
    jsonl_manager.save(session)

    sqlite_manager = SQLiteSessionManager(tmp_path)
    migrated = sqlite_manager.get_or_create(key)
    history, _ = migrated.get_history(max_messages=20, prune_tool_results=False)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "final"},
    ]


def test_large_session_roundtrips_from_sqlite(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:large"
    session = sqlite_manager.get_or_create(key)
    batch = []
    for index in range(1_200):
        role = "user" if index % 2 == 0 else "assistant"
        batch.append({"role": role, "content": f"msg-{index}"})
    session.checkpoint(batch)

    sqlite_manager.invalidate(key)
    loaded = sqlite_manager.get_or_create(key)
    assert isinstance(loaded.messages, LazyMessageList)
    assert not loaded.messages.is_loaded
    history, _ = loaded.get_history(max_messages=10, prune_tool_results=False)

    assert len(loaded.messages) == 1_200
    assert not loaded.messages.is_loaded
    assert history[0]["content"] == "msg-1190"
    assert history[-1]["content"] == "msg-1199"


def test_prune_old_tool_results_persists_across_reload(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:persist-prune"
    _build_tool_session(sqlite_manager, key)

    first = sqlite_manager.prune_old_tool_results(
        key,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    rows = _fetch_all(
        sqlite_manager.db_path,
        """
        SELECT seq, pruned_at, raw_json
        FROM message
        WHERE session_key = ? AND role = 'tool'
        ORDER BY seq ASC
        """,
        (key,),
    )

    assert first.messages_pruned > 0
    assert any(row["pruned_at"] for row in rows)
    assert any("[Tool output cleared to save context" in json.loads(row["raw_json"])["content"] for row in rows)

    sqlite_manager.invalidate(key)
    reloaded = sqlite_manager.get_or_create(key)
    history, prune_result = reloaded.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    second = sqlite_manager.prune_old_tool_results(
        key,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    assert any(
        msg["role"] == "tool" and "[Tool output cleared to save context" in str(msg["content"])
        for msg in history
    )
    assert prune_result is not None
    assert prune_result.messages_pruned == first.messages_pruned
    assert second.messages_pruned == 0


def test_get_or_create_uses_lazy_message_loading(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:lazy"
    session = sqlite_manager.get_or_create(key)
    session.checkpoint(
        [{"role": "user" if idx % 2 == 0 else "assistant", "content": f"m-{idx}"} for idx in range(20)]
    )

    sqlite_manager.invalidate(key)
    loaded = sqlite_manager.get_or_create(key)

    assert isinstance(loaded.messages, LazyMessageList)
    assert not loaded.messages.is_loaded
    assert len(loaded.messages) == 20
    assert loaded.messages[-1]["content"] == "m-19"
    assert loaded.messages.is_loaded


def test_concurrent_get_or_create_returns_single_cached_session(sqlite_manager: SQLiteSessionManager) -> None:
    key = "cli:shared"

    def _worker(_: int) -> int:
        return id(sqlite_manager.get_or_create(key))

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(_worker, range(32)))

    rows = _fetch_all(
        sqlite_manager.db_path,
        "SELECT COUNT(*) AS count FROM session WHERE key = ?",
        (key,),
    )

    assert len(set(ids)) == 1
    assert rows[0]["count"] == 1


@pytest.mark.asyncio
async def test_router_uses_sqlite_session_manager_when_profile_requests_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: dict[str, Any] = {}

    class FakeLoop:
        def __init__(self, **kwargs: Any):
            created["session_manager"] = kwargs["session_manager"]

    monkeypatch.setattr("nanobot.agent.router.AgentLoop", FakeLoop)

    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "workspace": str(tmp_path / "workspace"),
                    "model": "stub-model",
                },
                "profiles": {
                    "sqlite-agent": {
                        "model": "stub-model",
                        "sessionStore": "sqlite",
                    }
                },
            }
        }
    )
    router = AgentRouter(
        front_bus=MessageBus(),
        config=config,
        provider=SimpleNamespace(),
    )
    profile = config.agents.profiles["sqlite-agent"].resolve(config.agents.defaults)

    await router._create_agent_instance("sqlite-agent", profile)

    assert isinstance(created["session_manager"], SQLiteSessionManager)


def test_rewrite_does_not_destroy_unloaded_lazy_messages(sqlite_manager: SQLiteSessionManager) -> None:
    """Regression: _rewrite_session_locked must hydrate LazyMessageList BEFORE
    deleting rows.  Without the fix, DELETE runs first, then _ensure_loaded()
    queries the now-empty table, and the commit persists zero messages."""
    session = sqlite_manager.get_or_create("test:lazy-rewrite")
    msgs = [
        {"role": "user", "content": f"message {i}", "timestamp": f"2026-01-01T00:00:{i:02d}"}
        for i in range(50)
    ]
    session.checkpoint(msgs)
    assert len(session.messages) == 50

    # Simulate a gateway restart: drop the cache so the next load
    # creates a fresh LazyMessageList that hasn't been hydrated yet.
    sqlite_manager._cache.clear()
    reloaded = sqlite_manager.get_or_create("test:lazy-rewrite")
    assert isinstance(reloaded.messages, LazyMessageList)
    assert not reloaded.messages.is_loaded
    assert len(reloaded.messages) == 50  # total_count from metadata

    # Force _messages_changed to return True by bumping the version
    reloaded.messages._version += 1

    # This is the dangerous call — save() triggers _rewrite_session_locked
    # with rewrite_messages=True while the lazy list is unloaded.
    sqlite_manager.save(reloaded)

    # Verify messages survived
    sqlite_manager._cache.clear()
    final = sqlite_manager.get_or_create("test:lazy-rewrite")
    assert len(final.messages) == 50
    # Hydrate and verify content
    assert final.messages[0]["content"] == "message 0"
    assert final.messages[49]["content"] == "message 49"


def test_save_all_skips_clean_cached_sessions_and_preserves_external_message_count(
    tmp_path: Path,
) -> None:
    key = "cli:stale-clean"
    writer = SQLiteSessionManager(tmp_path)
    session = writer.get_or_create(key)
    session.checkpoint([{"role": "user", "content": f"msg-{i}"} for i in range(5)])

    stale_manager = SQLiteSessionManager(tmp_path)
    stale = stale_manager.get_or_create(key)
    assert isinstance(stale.messages, LazyMessageList)
    assert not stale.messages.is_loaded
    assert len(stale.messages) == 5

    _insert_message_rows(stale_manager.db_path, session_key=key, start_seq=5, count=15)
    with sqlite3.connect(stale_manager.db_path) as conn:
        conn.execute(
            "UPDATE session SET message_count = ?, updated_at = ? WHERE key = ?",
            (20, datetime.now().isoformat(), key),
        )
        conn.commit()

    stale_manager.save_all()

    row = _fetch_all(
        stale_manager.db_path,
        """
        SELECT message_count, (SELECT COUNT(*) FROM message WHERE session_key = ?) AS actual_count
        FROM session
        WHERE key = ?
        """,
        (key, key),
    )[0]

    assert row["message_count"] == 20
    assert row["actual_count"] == 20


def test_get_or_create_repairs_message_count_drift_from_actual_rows(
    tmp_path: Path,
) -> None:
    key = "cli:repair-drift"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint([{"role": "user", "content": f"msg-{i}"} for i in range(5)])

    _insert_message_rows(manager.db_path, session_key=key, start_seq=5, count=15)
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            "UPDATE session SET message_count = ?, last_consolidated_seq = ? WHERE key = ?",
            (5, 99, key),
        )
        conn.commit()

    manager.invalidate(key)
    loaded = manager.get_or_create(key)
    history, _ = loaded.get_history(max_messages=max(1, len(loaded.messages)), prune_tool_results=False)
    row = _fetch_all(
        manager.db_path,
        "SELECT message_count, last_consolidated_seq FROM session WHERE key = ?",
        (key,),
    )[0]

    assert isinstance(loaded.messages, LazyMessageList)
    assert not loaded.messages.is_loaded
    assert len(loaded.messages) == 20
    assert len(history) == 20
    assert row["message_count"] == 20
    assert row["last_consolidated_seq"] == 20


def test_get_or_create_reloads_clean_cached_session_when_db_fingerprint_changes(
    tmp_path: Path,
) -> None:
    key = "cli:reload-clean-cache"
    manager_a = SQLiteSessionManager(tmp_path)
    manager_b = SQLiteSessionManager(tmp_path)

    session_a = manager_a.get_or_create(key)
    session_a.checkpoint([{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}])

    cached_b = manager_b.get_or_create(key)
    assert isinstance(cached_b.messages, LazyMessageList)
    assert len(cached_b.messages) == 2

    session_a.checkpoint([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}])

    refreshed_b = manager_b.get_or_create(key)
    history, _ = refreshed_b.get_history(max_messages=max(1, len(refreshed_b.messages)), prune_tool_results=False)
    revision = _fetch_all(
        manager_a.db_path,
        "SELECT revision FROM session WHERE key = ?",
        (key,),
    )[0]["revision"]

    assert refreshed_b is not cached_b
    assert len(refreshed_b.messages) == 4
    assert [msg["content"] for msg in history] == ["u0", "a0", "u1", "a1"]
    assert getattr(refreshed_b, "_sqlite_revision", None) == revision


def test_sqlite_session_count_helpers_use_visible_window_after_compaction(
    tmp_path: Path,
) -> None:
    key = "cli:count-helpers"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint(
        [{"role": "user" if idx % 2 == 0 else "assistant", "content": f"msg-{idx}"} for idx in range(20)]
    )
    session.append_compaction(
        summary="summary",
        first_kept_index=12,
        tokens_before=100,
        file_ops={"read_files": [], "modified_files": []},
    )

    manager.invalidate(key)
    loaded = manager.get_or_create(key)

    assert loaded.get_message_count() == 20
    assert loaded.get_visible_message_count() == 8


def test_sqlite_session_revision_increments_on_persisted_mutations(
    tmp_path: Path,
) -> None:
    key = "cli:revision"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)

    initial_revision = _fetch_all(
        manager.db_path,
        "SELECT revision FROM session WHERE key = ?",
        (key,),
    )[0]["revision"]

    session.checkpoint([{"role": "user", "content": "u0"}])
    after_checkpoint = _fetch_all(
        manager.db_path,
        "SELECT revision FROM session WHERE key = ?",
        (key,),
    )[0]["revision"]

    session.metadata["tag"] = "x"
    manager.save(session)
    after_metadata_save = _fetch_all(
        manager.db_path,
        "SELECT revision FROM session WHERE key = ?",
        (key,),
    )[0]["revision"]

    session.append_compaction(
        summary="summary",
        first_kept_index=1,
        tokens_before=50,
        file_ops={"read_files": [], "modified_files": []},
    )
    after_compaction = _fetch_all(
        manager.db_path,
        "SELECT revision FROM session WHERE key = ?",
        (key,),
    )[0]["revision"]

    assert after_checkpoint > initial_revision
    assert after_metadata_save > after_checkpoint
    assert after_compaction > after_metadata_save


def test_save_state_persists_metadata_and_cursor_without_rewriting_messages(
    tmp_path: Path,
) -> None:
    key = "cli:save-state"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint([{"role": "user", "content": f"msg-{i}"} for i in range(4)])
    baseline_revision = _fetch_all(
        manager.db_path,
        "SELECT revision, message_count FROM session WHERE key = ?",
        (key,),
    )[0]

    session.metadata["flag"] = "on"
    session.last_consolidated = 3
    manager.save_state(session)

    row = _fetch_all(
        manager.db_path,
        "SELECT revision, message_count, last_consolidated_seq, metadata_json FROM session WHERE key = ?",
        (key,),
    )[0]
    messages = _fetch_all(
        manager.db_path,
        "SELECT COUNT(*) AS count FROM message WHERE session_key = ?",
        (key,),
    )[0]

    assert row["message_count"] == 4
    assert messages["count"] == 4
    assert row["last_consolidated_seq"] == 3
    assert json.loads(row["metadata_json"])["flag"] == "on"
    assert row["revision"] > baseline_revision["revision"]


def test_save_state_falls_back_to_full_save_when_messages_changed(
    tmp_path: Path,
) -> None:
    key = "cli:save-state-fallback"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint([{"role": "user", "content": "before"}])

    manager.invalidate(key)
    loaded = manager.get_or_create(key)
    loaded.messages[0]["content"] = "after"

    manager.save_state(loaded)

    manager.invalidate(key)
    reloaded = manager.get_or_create(key)
    assert reloaded.messages[0]["content"] == "after"


def test_apply_state_updates_metadata_removes_keys_and_advances_cursor(
    tmp_path: Path,
) -> None:
    key = "cli:apply-state"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint([{"role": "user", "content": f"msg-{i}"} for i in range(4)])
    session.metadata["old"] = "gone"

    removed = manager.apply_state(
        session,
        metadata_updates={"new": "value"},
        metadata_remove=["old"],
        last_consolidated=3,
    )

    manager.invalidate(key)
    reloaded = manager.get_or_create(key)

    assert removed == {"old": "gone"}
    assert reloaded.metadata["new"] == "value"
    assert "old" not in reloaded.metadata
    assert reloaded.last_consolidated == 3


def test_session_state_helper_methods_persist_semantic_fields(
    tmp_path: Path,
) -> None:
    key = "cli:state-helpers"
    manager = SQLiteSessionManager(tmp_path)
    session = manager.get_or_create(key)
    session.checkpoint([{"role": "user", "content": f"msg-{i}"} for i in range(5)])

    manager.set_usage_snapshot(session, total_input_tokens=1234, source="test")
    manager.set_compaction_plan(
        session,
        {"extract_start": 1, "extract_end": 4, "cut_point_type": "clean"},
    )
    removed_plan = manager.pop_compaction_plan(session)
    manager.advance_last_consolidated(session, 4)
    manager.clear_usage_snapshot(session)

    manager.invalidate(key)
    reloaded = manager.get_or_create(key)

    assert removed_plan == {"extract_start": 1, "extract_end": 4, "cut_point_type": "clean"}
    assert "usage_snapshot" not in reloaded.metadata
    assert "_structured_compaction_plan" not in reloaded.metadata
    assert reloaded.last_consolidated == 4
