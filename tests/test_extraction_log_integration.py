from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.session.extraction_log import load_extraction_log


class _StubProvider:
    async def chat(self, *args, **kwargs):  # pragma: no cover - not used in this test
        raise AssertionError("provider.chat should not be called")


@pytest.mark.asyncio
async def test_hybrid_consolidation_writes_extraction_sidecar(tmp_path: Path) -> None:
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
    )
    agent._CONSOLIDATION_KEEP_COUNT = 5
    agent._memory_graph_config = {
        "consolidation": {"engine": "hybrid", "batch_messages": 30}
    }

    history_file = tmp_path / "memory" / "history" / "2026-03-06.md"
    compact_result = SimpleNamespace(
        success=True,
        error=None,
        entries=["entry-1", "entry-2"],
        history_file=history_file,
        transcript_chars=222,
        llm_response_chars=120,
        llm_response_preview="preview",
        llm_usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 3,
        },
        llm_duration_ms=42,
        retry_needed=False,
        retry_llm_usage=None,
        finish_reason="end_turn",
        items_before_cap=2,
        existing_memories_recalled=[
            {
                "id": "mem-old",
                "content": "Existing memory",
                "memory_type": "fact",
                "importance": 0.7,
                "score": 0.81,
            }
        ],
        existing_memories_recall_duration_ms=8,
        existing_memories_query_text_chars=50,
        existing_memories_results_requested=15,
        extracted_items=[
            {
                "content": "Tracked add.",
                "type": "fact",
                "importance": 0.6,
                "action": "add",
                "scope": "shared",
            },
            {
                "content": "Tracked supersede.",
                "type": "decision",
                "importance": 0.9,
                "action": "supersede",
                "scope": "shared",
            },
        ],
        indexed_items=[
            {
                "memory_id": "mem-add",
                "content": "Tracked add.",
                "action": "add",
                "type": "fact",
                "importance": 0.6,
            },
            {
                "memory_id": "mem-supersede",
                "content": "Tracked supersede.",
                "action": "supersede",
                "type": "decision",
                "importance": 0.9,
                "superseded_memory_id": "mem-old",
                "edge_id": "edge-1",
            },
        ],
        indexing_duration_ms=11,
        memories_added=1,
        memories_updated=0,
        memories_superseded=1,
        items_skipped=0,
        edges_created=2,
        context_overlap_count=5,
        user_turns_in_window=4,
        prompt_system_chars=20,
        prompt_context_section_chars=30,
        prompt_extraction_section_chars=40,
        prompt_existing_memories_chars=50,
        prompt_full_hash="sha256:abc123",
        max_tokens_budget=2048,
        memory_md_rewrite_triggered=False,
        memory_md_before_chars=0,
        memory_md_after_chars=0,
        memory_md_duration_ms=0,
    )
    hybrid = SimpleNamespace(compact=AsyncMock(return_value=compact_result))
    llm_adapter = SimpleNamespace(
        estimate_tokens=lambda text: 1000,
        get_context_window=lambda model: 200000,
    )
    consolidator_mock = SimpleNamespace(
        llm=llm_adapter,
        model="claude-haiku-4-5",
        build_extraction_messages=lambda window: [{"role": "user", "content": "window"}],
        get_extraction_max_tokens=lambda window: 2048,
    )
    agent._memory_module = SimpleNamespace(initialized=True, hybrid=hybrid, consolidator=consolidator_mock)

    session = agent.sessions.get_or_create("cli:test")
    for index in range(12):
        session.add_message("user", f"msg-{index}")
    agent.sessions.save(session)

    ok = await agent._consolidate_memory(session, archive_all=False)

    assert ok is True

    entries = load_extraction_log(agent.sessions.get_session_path(session.key))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["session_key"] == "cli:test"
    assert entry["trigger"] == {
        "reason": "compaction",
        "compaction_event_id": None,
        "total_session_messages": 12,
        "last_consolidated_before": 0,
    }
    assert entry["message_window"] == {
        "start_index": 0,
        "end_index": 7,
        "message_count": 7,
        "user_turns": 4,
        "context_overlap": {
            "start_index": 0,
            "message_count": 5,
        },
    }
    assert entry["existing_memories_recall"]["memories"][0]["id"] == "mem-old"
    assert entry["prompt"]["full_prompt_hash"] == "sha256:abc123"
    assert entry["llm_call"] == {
        "model": "claude-haiku-4-5",
        "temperature": 0.0,
        "duration_ms": 42,
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 3,
        "retry_needed": False,
        "retry_input_tokens": None,
        "retry_output_tokens": None,
        "finish_reason": "end_turn",
        "error": None,
    }
    assert entry["extraction_raw"] == {
        "response_chars": 120,
        "response_preview": "preview",
        "parse_ok": True,
        "parse_error": None,
        "items_before_cap": 2,
        "items_after_cap": 2,
        "items_by_action": {"add": 1, "supersede": 1},
        "items_by_type": {"fact": 1, "decision": 1},
        "items_by_scope": {"shared": 2},
    }
    assert entry["graph_indexing"]["indexed_items"] == compact_result.indexed_items
    assert entry["file_ops"] == {
        "history_file": "memory/history/2026-03-06.md",
        "history_entries_written": 2,
        "memory_md_rewrite": {
            "triggered": False,
            "before_chars": 0,
            "after_chars": 0,
            "duration_ms": 0,
        },
    }
    assert entry["result"]["success"] is True
    assert entry["result"]["total_items_extracted"] == 2
    assert entry["result"]["total_items_indexed"] == 2
    assert entry["result"]["new_last_consolidated"] == 7
