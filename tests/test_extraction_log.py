"""Tests for the extraction sidecar log."""

import json
from pathlib import Path

from nanobot.session.extraction_log import ExtractionEvent, ExtractionLogger, load_extraction_log


def test_extraction_event_builder_serializes_full_schema() -> None:
    event = ExtractionEvent(session_key="discord:123", agent_id="nanobot-dev")

    event.set_trigger(
        reason="compaction",
        compaction_event_id="comp-1",
        total_session_messages=4684,
        last_consolidated_before=4086,
    )
    event.set_message_window(
        start_index=4086,
        end_index=4654,
        user_turns=42,
        context_overlap_start=4081,
        context_overlap_count=5,
    )
    event.set_existing_memories(
        query_text_chars=500,
        results_requested=15,
        results_returned=1,
        recall_duration_ms=45,
        memories=[
            {
                "id": "mem-1",
                "content": "User prefers dark mode",
                "memory_type": "preference",
                "importance": 0.8,
                "score": 0.72,
            }
        ],
    )
    event.set_prompt_stats(
        system_prompt_chars=2847,
        transcript_chars=24500,
        context_section_chars=1200,
        extraction_section_chars=23300,
        existing_memories_section_chars=850,
        max_tokens_budget=2048,
        full_prompt_hash="sha256:abc123",
    )
    event.set_llm_call(
        model="claude-haiku-4-5",
        temperature=0.0,
        duration_ms=3200,
        input_tokens=8500,
        output_tokens=1200,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        retry_needed=False,
        retry_input_tokens=None,
        retry_output_tokens=None,
        finish_reason="end_turn",
        error=None,
    )
    event.set_extraction_raw(
        response_chars=1200,
        response_preview="x" * 700,
        parse_ok=True,
        parse_error=None,
        items_before_cap=8,
        items_after_cap=8,
        items_by_action={"add": 6, "supersede": 1, "reinforce": 1},
        items_by_type={"fact": 3, "decision": 2},
        items_by_scope={"shared": 7, "agent": 1},
    )
    event.set_extracted_items(
        [
            {
                "content": "The API uses RS256 JWT tokens with 24-hour expiry",
                "type": "decision",
                "importance": 0.8,
                "action": "add",
            }
        ]
    )
    event.set_graph_indexing(
        duration_ms=120,
        memories_added=6,
        memories_updated=1,
        memories_superseded=1,
        edges_created=3,
        items_skipped=0,
        indexed_items=[
            {
                "memory_id": "mem-new",
                "content": "The API uses RS256 JWT tokens with 24-hour expiry",
                "action": "add",
                "type": "decision",
                "importance": 0.8,
            }
        ],
    )
    event.set_file_ops(
        history_file="memory/history/2026-03-06.md",
        history_entries_written=8,
        memory_md_rewrite_triggered=True,
        memory_md_before_chars=14561,
        memory_md_after_chars=15200,
        memory_md_duration_ms=2800,
    )
    event.finalize(
        success=True,
        error=None,
        total_duration_ms=6200,
        total_items_extracted=8,
        total_items_indexed=8,
        new_last_consolidated=4654,
    )

    payload = event.data
    assert payload["session_key"] == "discord:123"
    assert payload["agent_id"] == "nanobot-dev"
    assert payload["trigger"] == {
        "reason": "compaction",
        "compaction_event_id": "comp-1",
        "total_session_messages": 4684,
        "last_consolidated_before": 4086,
    }
    assert payload["message_window"] == {
        "start_index": 4086,
        "end_index": 4654,
        "message_count": 568,
        "user_turns": 42,
        "context_overlap": {
            "start_index": 4081,
            "message_count": 5,
        },
    }
    assert payload["existing_memories_recall"]["memories"][0]["score"] == 0.72
    assert payload["prompt"]["full_prompt_hash"] == "sha256:abc123"
    assert payload["llm_call"]["finish_reason"] == "end_turn"
    assert payload["extraction_raw"]["response_preview"] == "x" * 500
    assert payload["graph_indexing"]["memories_superseded"] == 1
    assert payload["file_ops"]["memory_md_rewrite"]["duration_ms"] == 2800
    assert payload["result"]["new_last_consolidated"] == 4654

    serialized = json.dumps(payload)
    assert '"extracted_items"' in serialized
    assert '"graph_indexing"' in serialized


def test_extraction_logger_writes_and_loads_entries(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    logger = ExtractionLogger(session_path)
    event = ExtractionEvent(session_key="discord:123", agent_id="nanobot-dev")
    event.finalize(
        success=True,
        error=None,
        total_duration_ms=10,
        total_items_extracted=1,
        total_items_indexed=1,
        new_last_consolidated=2,
    )

    logger.write(event)

    assert logger.path == tmp_path / "session.extraction.jsonl"
    assert logger.path.exists()

    entries = load_extraction_log(session_path)
    assert len(entries) == 1
    assert entries[0]["extraction_id"] == event.data["extraction_id"]
    assert entries[0]["result"]["total_items_indexed"] == 1


def test_load_extraction_log_skips_malformed_lines(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    log_path = tmp_path / "session.extraction.jsonl"
    log_path.write_text(
        '\n'.join(
            [
                '{"timestamp":"2026-03-06T00:00:00+00:00","result":{"success":true}}',
                "not json",
                '{"timestamp":"2026-03-05T00:00:00+00:00","result":{"success":false}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = load_extraction_log(session_path)

    assert [entry["result"]["success"] for entry in entries] == [False, True]
