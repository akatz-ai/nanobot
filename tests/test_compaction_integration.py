from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.session.compaction import (
    SUMMARIZATION_SYSTEM_PROMPT,
    compact_session,
    estimate_message_tokens,
)
from nanobot.session.manager import SessionManager


def _structured_summary(label: str = "Summary") -> str:
    return (
        f"## Goal\n{label}\n\n"
        "## Constraints & Preferences\n- keep paths\n\n"
        "## Progress\n"
        "### Done\n- [x] done\n"
        "### In Progress\n- [ ] in progress\n"
        "### Blocked\n- none\n\n"
        "## Key Decisions\n- **A**: B\n\n"
        "## Next Steps\n1. Continue\n\n"
        "## Critical Context\n- context"
    )


def _add_long_messages(session, turns: int = 40) -> None:
    payload = "x" * 900
    for i in range(turns):
        session.add_message("user", f"Question {i} {payload}")
        session.add_message("assistant", f"Answer {i} {payload}")


def _make_structured_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


@pytest.mark.asyncio
async def test_full_structured_compaction_flow(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)

    loop.provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=_structured_summary("First")),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )

    extraction_calls: list[tuple[int, int] | None] = []

    async def _fake_consolidate(sess, archive_all: bool = False, extraction_range=None):
        extraction_calls.append(extraction_range)
        if extraction_range:
            sess.last_consolidated = extraction_range[1]
        return True

    loop._consolidate_memory = _fake_consolidate

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }
    loop.sessions.save(session)

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="next")
    )

    assert len(session.compactions) == 1
    usage_snapshot = session.metadata.get("usage_snapshot")
    assert isinstance(usage_snapshot, dict)
    assert int(usage_snapshot.get("total_input_tokens", 0)) > 0
    assert int(loop._last_input_tokens.get(session.key, 0)) > 0
    assert extraction_calls and extraction_calls[0] is not None
    assert session.last_consolidated > 0

    history, _ = session.get_history(max_messages=200)
    assert history[0]["role"] == "system"
    assert "## Goal" in history[0]["content"]

    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    reloaded = loop.sessions.get_or_create(session.key)
    assert len(reloaded.compactions) == 1
    assert reloaded.get_last_compaction() is not None


@pytest.mark.asyncio
async def test_iterative_structured_compaction_uses_previous_summary(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)

    async def _chat_side_effect(*args, **kwargs):
        messages = kwargs.get("messages") or []
        if messages and messages[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT:
            # Return different summaries for each compaction pass.
            count = sum(
                1
                for call in loop.provider.chat.await_args_list
                if call.kwargs.get("messages", [{}])[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT
            )
            return LLMResponse(content=_structured_summary(f"Pass {count + 1}"))
        return LLMResponse(content="ok", tool_calls=[])

    loop.provider.chat = AsyncMock(side_effect=_chat_side_effect)

    async def _fake_consolidate(sess, archive_all: bool = False, extraction_range=None):
        if extraction_range:
            sess.last_consolidated = extraction_range[1]
        return True

    loop._consolidate_memory = _fake_consolidate

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="first")
    )

    _add_long_messages(session, turns=12)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 185_000,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="second")
    )

    assert len(session.compactions) == 2
    assert session.compactions[1]["previous_summary"]

    summary_calls = [
        call
        for call in loop.provider.chat.await_args_list
        if call.kwargs.get("messages", [{}])[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT
    ]
    assert len(summary_calls) >= 2
    second_prompt = summary_calls[1].kwargs["messages"][1]["content"]
    assert "<previous_summary>" in second_prompt

    history, _ = session.get_history(max_messages=200)
    assert history[0]["role"] == "system"
    assert "## Goal" in history[0]["content"]


@pytest.mark.asyncio
async def test_noop_structured_compaction_under_threshold(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))

    consolidate = AsyncMock(return_value=True)
    loop._consolidate_memory = consolidate

    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 100,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="small")
    )

    assert session.compactions == []
    consolidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_compaction_check_uses_visible_history_when_usage_missing(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))

    consolidate = AsyncMock(return_value=True)
    loop._consolidate_memory = consolidate

    session = loop.sessions.get_or_create("cli:test")
    payload = "x" * 5000
    for i in range(120):
        session.add_message("user", f"old-question-{i} {payload}")
        session.add_message("assistant", f"old-answer-{i} {payload}")

    recent_start = len(session.messages)
    for i in range(6):
        session.add_message("user", f"recent-question-{i}")
        session.add_message("assistant", f"recent-answer-{i}")

    session.append_compaction(
        summary=_structured_summary("Previous"),
        first_kept_index=recent_start,
        tokens_before=180_000,
        file_ops={"read_files": [], "modified_files": []},
    )
    session.metadata.pop("usage_snapshot", None)
    loop._last_input_tokens.pop(session.key, None)

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="continue")
    )

    consolidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_prompt_budget_trims_history_before_chat(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    model_key = "preflight-budget-model"
    loop.model = model_key
    # Keep this above reserve/max_tokens so the compaction threshold remains valid.
    loop.MODEL_CONTEXT_WINDOWS[model_key] = 60_000
    loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))

    try:
        session = loop.sessions.get_or_create("cli:test")
        payload = "x" * 2500
        for i in range(80):
            session.add_message("user", f"question-{i} {payload}")
            session.add_message("assistant", f"answer-{i} {payload}")

        # Force compaction gate to use this low value so preflight trimming is exercised.
        session.metadata["usage_snapshot"] = {
            "total_input_tokens": 100,
            "message_index": len(session.messages),
        }

        await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="trim me")
        )

        sent_messages = loop.provider.chat.await_args.kwargs["messages"]
        sent_estimate = sum(estimate_message_tokens(msg) for msg in sent_messages)
        assert sent_estimate <= loop._get_prompt_input_budget()
    finally:
        loop.MODEL_CONTEXT_WINDOWS.pop(model_key, None)


@pytest.mark.asyncio
async def test_cut_point_edge_case_snaps_over_tool_chain(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "name": "read_file", "content": "ok"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    estimates = [estimate_message_tokens(m) for m in session.messages]

    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(content=_structured_summary("Edge"))

    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }
    entry = await compact_session(
        session=session,
        provider=provider,
        model="test-model",
        keep_recent_tokens=sum(estimates[2:]),
    )

    assert entry is not None
    assert entry.first_kept_index == 4


@pytest.mark.asyncio
async def test_failure_recovery_on_summary_error(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)

    async def _chat_side_effect(*args, **kwargs):
        messages = kwargs.get("messages") or []
        if messages and messages[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT:
            raise RuntimeError("summary failed")
        return LLMResponse(content="ok", tool_calls=[])

    loop.provider.chat = AsyncMock(side_effect=_chat_side_effect)
    loop._consolidate_memory = AsyncMock(return_value=True)

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="recover")
    )

    assert session.compactions == []
    usage_snapshot = session.metadata.get("usage_snapshot")
    assert isinstance(usage_snapshot, dict)
    assert usage_snapshot.get("source") == "estimated_visible_history"
    assert int(usage_snapshot.get("total_input_tokens", 0)) > 0


@pytest.mark.asyncio
async def test_memory_extraction_called_with_explicit_range(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    loop.provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=_structured_summary("Extract")),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )

    consolidate = AsyncMock(return_value=True)
    loop._consolidate_memory = consolidate

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="extract")
    )

    assert consolidate.await_count == 1
    kwargs = consolidate.await_args.kwargs
    assert kwargs.get("extraction_range") is not None


def test_backward_compat_old_sessions_without_compactions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    key = "cli:legacy"
    path = manager._get_session_path(key)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_type": "metadata",
                        "key": key,
                        "created_at": "2026-01-01T00:00:00",
                        "updated_at": "2026-01-01T00:00:00",
                        "metadata": {},
                        "last_consolidated": 0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"role": "user", "content": "hello"}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": "world"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = manager.get_or_create(key)
    history, _ = loaded.get_history(max_messages=10)
    assert loaded.compactions == []
    assert [m["content"] for m in history] == ["hello", "world"]


@pytest.mark.asyncio
async def test_concurrent_structured_compactions_are_serialized(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)

    async def _chat_side_effect(*args, **kwargs):
        messages = kwargs.get("messages") or []
        if messages and messages[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT:
            return LLMResponse(content=_structured_summary("Concurrent"))
        return LLMResponse(content="ok", tool_calls=[])

    loop.provider.chat = AsyncMock(side_effect=_chat_side_effect)

    active = 0
    max_active = 0
    calls = 0

    async def _slow_consolidate(sess, archive_all: bool = False, extraction_range=None):
        nonlocal active, max_active, calls
        calls += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        if extraction_range:
            sess.last_consolidated = extraction_range[1]
        active -= 1
        return True

    loop._consolidate_memory = _slow_consolidate

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=42)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 190_000,
        "message_index": len(session.messages),
    }

    await asyncio.gather(
        loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="one")
        ),
        loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="two")
        ),
    )

    assert calls >= 1
    assert max_active == 1


@pytest.mark.asyncio
async def test_save_reload_round_trip_with_compaction_and_messages(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    loop.provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=_structured_summary("Roundtrip")),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )

    async def _fake_consolidate(sess, archive_all: bool = False, extraction_range=None):
        if extraction_range:
            sess.last_consolidated = extraction_range[1]
        return True

    loop._consolidate_memory = _fake_consolidate

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="persist")
    )

    before = {
        "messages": list(session.messages),
        "compactions": list(session.compactions),
        "last_consolidated": session.last_consolidated,
    }

    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    loaded = loop.sessions.get_or_create(session.key)

    assert loaded.messages == before["messages"]
    assert loaded.compactions == before["compactions"]
    assert loaded.last_consolidated == before["last_consolidated"]


@pytest.mark.asyncio
async def test_process_message_pruning_uses_fresh_snapshot_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _make_structured_loop(tmp_path)
    loop._run_agent_loop = AsyncMock(return_value=("ok", [], []))
    loop._retrieve_memory_context = AsyncMock(return_value=None)

    original_window = loop.MODEL_CONTEXT_WINDOWS.get(loop.model)
    loop.MODEL_CONTEXT_WINDOWS[loop.model] = 30_000
    try:
        session = loop.sessions.get_or_create("cli:test")
        session.add_message("user", "old-user")
        session.add_message("assistant", "old-assistant")
        session.metadata["usage_snapshot"] = {
            "total_input_tokens": 8_000,
            "message_index": len(session.messages),
        }

        baseline_history = [{"role": "user", "content": "x" * 45_000}]
        pressure_history = [{"role": "user", "content": "y" * 35_000}]

        def _fake_get_history(*args, **kwargs):
            if kwargs.get("prune_tool_results") is False:
                return baseline_history, None
            return pressure_history, None

        monkeypatch.setattr(session, "get_history", _fake_get_history)
        compact_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nanobot.agent.loop.compact_session", compact_mock)

        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="next")
        )

        assert response is not None
        compact_mock.assert_not_awaited()
    finally:
        if original_window is None:
            loop.MODEL_CONTEXT_WINDOWS.pop(loop.model, None)
        else:
            loop.MODEL_CONTEXT_WINDOWS[loop.model] = original_window


@pytest.mark.asyncio
async def test_process_message_ignores_stale_snapshot_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _make_structured_loop(tmp_path)
    loop._run_agent_loop = AsyncMock(return_value=("ok", [], []))
    loop._retrieve_memory_context = AsyncMock(return_value=None)

    original_window = loop.MODEL_CONTEXT_WINDOWS.get(loop.model)
    loop.MODEL_CONTEXT_WINDOWS[loop.model] = 30_000
    try:
        session = loop.sessions.get_or_create("cli:test")
        session.add_message("user", "old-user")
        session.add_message("assistant", "old-assistant")
        session.metadata["usage_snapshot"] = {
            "total_input_tokens": 8_000,
            "message_index": len(session.messages) - 500,
        }
        loop._last_input_tokens.pop(session.key, None)

        baseline_history = [{"role": "user", "content": "x" * 45_000}]
        pressure_history = [{"role": "user", "content": "y" * 35_000}]

        def _fake_get_history(*args, **kwargs):
            if kwargs.get("prune_tool_results") is False:
                return baseline_history, None
            return pressure_history, None

        monkeypatch.setattr(session, "get_history", _fake_get_history)
        compact_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nanobot.agent.loop.compact_session", compact_mock)

        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="test", content="next")
        )

        assert response is not None
        assert compact_mock.await_count == 1
    finally:
        if original_window is None:
            loop.MODEL_CONTEXT_WINDOWS.pop(loop.model, None)
        else:
            loop.MODEL_CONTEXT_WINDOWS[loop.model] = original_window


@pytest.mark.asyncio
async def test_compact_session_pruning_uses_fresh_snapshot_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 8_000,
        "message_index": len(session.messages),
    }

    baseline_history = [{"role": "user", "content": "x" * 45_000}]
    pressure_history = [{"role": "user", "content": "y" * 35_000}]

    def _fake_get_history(*args, **kwargs):
        if kwargs.get("prune_tool_results") is False:
            return baseline_history, None
        return pressure_history, None

    monkeypatch.setattr(session, "get_history", _fake_get_history)

    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content=_structured_summary("Fresh")))

    entry = await compact_session(
        session=session,
        provider=provider,
        model="test-model",
        context_window=30_000,
        reserve_tokens=16_384,
        keep_recent_tokens=8,
    )

    assert entry is None
    provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_compact_session_ignores_stale_snapshot_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 8_000,
        "message_index": len(session.messages) - 500,
    }

    baseline_history = [{"role": "user", "content": "x" * 45_000}]
    pressure_history = [{"role": "user", "content": "y" * 35_000}]

    def _fake_get_history(*args, **kwargs):
        if kwargs.get("prune_tool_results") is False:
            return baseline_history, None
        return pressure_history, None

    monkeypatch.setattr(session, "get_history", _fake_get_history)

    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content=_structured_summary("Stale")))

    entry = await compact_session(
        session=session,
        provider=provider,
        model="test-model",
        context_window=30_000,
        reserve_tokens=16_384,
        keep_recent_tokens=8,
    )

    assert entry is not None
    assert provider.chat.await_count >= 1


@pytest.mark.asyncio
async def test_new_command_clears_last_input_token_cache(tmp_path: Path) -> None:
    loop = _make_structured_loop(tmp_path)
    loop.provider.chat = AsyncMock(return_value=LLMResponse(content=_structured_summary("Archive")))
    loop._consolidate_memory = AsyncMock(return_value=True)

    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 10_000,
        "message_index": len(session.messages),
    }
    loop._last_input_tokens[session.key] = 10_000

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
    )

    assert response is not None
    assert "new session started" in response.content.lower()
    assert session.key not in loop._last_input_tokens
