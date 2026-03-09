from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.session.compaction import compact_session, estimate_message_tokens, should_compact
from nanobot.session.manager import Session


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


def _build_session(tool_names: list[str], *, output_chars: int = 18_000) -> Session:
    session = Session(key="cli:test-prune")
    for idx, name in enumerate(tool_names, start=1):
        session.messages.extend(
            _tool_turn(idx, tool_name=name, output_chars=output_chars)
        )
    return session


def _tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [msg for msg in messages if msg.get("role") == "tool"]


def _is_pruned_tool(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    return (
        isinstance(content, str)
        and content.startswith("[Tool output pruned from prompt:")
    )


def test_basic_pruning_prunes_older_tool_results() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"])

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=2_000,
        prune_minimum_tokens=100,
    )
    tools = _tool_messages(history)

    assert len(tools) == 5
    assert all(_is_pruned_tool(msg) for msg in tools[:3])
    assert not _is_pruned_tool(tools[3])
    assert not _is_pruned_tool(tools[4])
    assert "Use session_search to retrieve details." in str(tools[0].get("content"))


def test_protected_tools_never_pruned() -> None:
    session = _build_session(
        ["exec", "memory_recall", "exec", "exec", "exec"],
    )

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    tools = _tool_messages(history)

    memory_tool = next(msg for msg in tools if msg.get("name") == "memory_recall")
    assert not _is_pruned_tool(memory_tool)
    assert str(memory_tool.get("content")).startswith("memory_recall-2-")
    assert _is_pruned_tool(tools[0])
    assert _is_pruned_tool(tools[2])


def test_turn_protection_preserves_last_two_user_turns() -> None:
    session = _build_session(["exec", "exec", "exec", "exec"])

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    tools = _tool_messages(history)

    assert _is_pruned_tool(tools[0])
    assert _is_pruned_tool(tools[1])
    assert not _is_pruned_tool(tools[2])
    assert not _is_pruned_tool(tools[3])


def test_minimum_threshold_blocks_small_prunes() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=1_500)

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=5_000,
    )

    assert not any(_is_pruned_tool(msg) for msg in _tool_messages(history))


def test_default_thresholds_are_fixed_open_code_style_values() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=60_000)

    baseline_window, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
    )
    larger_window, _ = session.get_history(
        max_messages=500,
        context_window=300_000,
    )

    baseline_pruned = sum(1 for msg in _tool_messages(baseline_window) if _is_pruned_tool(msg))
    larger_pruned = sum(1 for msg in _tool_messages(larger_window) if _is_pruned_tool(msg))

    assert baseline_pruned > 0
    assert larger_pruned == baseline_pruned


@pytest.mark.asyncio
async def test_pruned_view_affects_compaction_decision() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=18_000)

    raw_history, _ = session.get_history(
        max_messages=500,
        context_window=40_000,
        prune_tool_results=False,
    )
    pruned_history, _ = session.get_history(
        max_messages=500,
        context_window=40_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    raw_tokens = sum(estimate_message_tokens(msg) for msg in raw_history)
    pruned_tokens = sum(estimate_message_tokens(msg) for msg in pruned_history)

    assert raw_tokens > pruned_tokens
    assert should_compact(
        raw_history,
        context_window=40_000,
        reserve_tokens=5_000,
        last_input_tokens=raw_tokens,
    )
    # Old/stale snapshot path: short-circuits to "compact" even on pruned history.
    assert should_compact(
        pruned_history,
        context_window=40_000,
        reserve_tokens=5_000,
        last_input_tokens=raw_tokens,
    )
    pruning_applied = pruned_tokens < raw_tokens
    if pruning_applied:
        decision_tokens: int | None = pruned_tokens
    elif raw_tokens > 0:
        decision_tokens = raw_tokens
    else:
        decision_tokens = None
    assert not should_compact(
        pruned_history,
        context_window=40_000,
        reserve_tokens=5_000,
        last_input_tokens=decision_tokens,
    )


def test_get_history_is_non_destructive() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=18_000)
    before = copy.deepcopy(session.messages)

    _, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    assert session.messages == before


def test_pruning_noop_without_tool_messages() -> None:
    session = Session(key="cli:test-no-tools")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session.add_message("user", "again")
    session.add_message("assistant", "done")

    history, _ = session.get_history(
        max_messages=50,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    assert [msg["role"] for msg in history] == ["user", "assistant", "user", "assistant"]
    assert not any(_is_pruned_tool(msg) for msg in history)


def test_compaction_summary_stays_out_of_history_pruning() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=18_000)
    summary = (
        "## Goal\nKeep summary\n\n"
        "## Progress\n### Done\n- [x] one\n\n"
        "## Next Steps\n1. Continue"
    )
    session.append_compaction(
        summary=summary,
        first_kept_index=0,
        tokens_before=12345,
        file_ops={"read_files": [], "modified_files": []},
    )

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )

    assert not any(
        msg.get("role") == "system" and msg.get("content") == summary
        for msg in history
    )
    assert any(_is_pruned_tool(msg) for msg in _tool_messages(history))


def test_all_eligible_old_tool_messages_can_be_pruned() -> None:
    session = _build_session(["exec", "exec", "exec", "exec", "exec"], output_chars=12_000)

    history, _ = session.get_history(
        max_messages=500,
        context_window=50_000,
        prune_protect_tokens=0,
        prune_minimum_tokens=100,
    )
    tools = _tool_messages(history)
    pruned_count = sum(1 for msg in tools if _is_pruned_tool(msg))

    assert len(tools) == 5
    assert pruned_count == 3
