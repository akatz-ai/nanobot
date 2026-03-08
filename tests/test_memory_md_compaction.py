from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.compaction_log import load_compaction_log


def _large_memory(*, repeats: int = 2400) -> str:
    repeated = "active project detail " * repeats
    return f"""# MEMORY

## Identity & Preferences
- User prefers concise summaries.

## Active Projects
- Phase 3 MEMORY.md auto-compaction rollout.

## Decisions
- Keep personal working memory lean.

## Reference Facts
- MEMORY.md is loaded on every turn.

## Recent Context
- {repeated}
"""


def _compacted_memory() -> str:
    return """# MEMORY

## Identity & Preferences
- User prefers concise summaries.

## Active Projects
- Phase 3 MEMORY.md auto-compaction rollout.

## Decisions
- Keep personal working memory lean.

## Reference Facts
- MEMORY.md is loaded on every turn.

## Recent Context
- Implementing standalone MEMORY.md distillation.
"""


def _tool_response(memory_update: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call-1",
                name="save_memory",
                arguments={
                    "history_entry": "",
                    "memory_update": memory_update,
                },
            )
        ],
    )


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def _structured_summary() -> str:
    return (
        "## Goal\nContinue the task\n\n"
        "## Constraints & Preferences\n- keep context tight\n\n"
        "## Progress\n"
        "### Done\n- [x] reviewed code\n"
        "### In Progress\n- [ ] wire compaction\n"
        "### Blocked\n- none\n\n"
        "## Key Decisions\n- **Use Haiku**: low-cost background compaction\n\n"
        "## Next Steps\n1. Finish compaction\n\n"
        "## Critical Context\n- memory.py and loop.py"
    )


def _add_long_messages(session, turns: int = 40) -> None:
    payload = "x" * 900
    for i in range(turns):
        session.add_message("user", f"Question {i} {payload}")
        session.add_message("assistant", f"Answer {i} {payload}")


@pytest.mark.asyncio
async def test_compact_memory_md_under_threshold(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_long_term(_compacted_memory())

    provider = MagicMock()
    provider.count_tokens = AsyncMock(return_value=3000)
    provider.chat = AsyncMock()

    result = await store.compact_memory_md(provider=provider, model="test-model")

    assert result is None
    provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_compact_memory_md_over_threshold(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    original = _large_memory()
    store.write_long_term(original)
    expected = store._coerce_to_canonical(_compacted_memory())

    provider = MagicMock()
    provider.count_tokens = AsyncMock(side_effect=[9000, 4500])
    provider.chat = AsyncMock(return_value=_tool_response(_compacted_memory()))

    result = await store.compact_memory_md(provider=provider, model="test-model")

    assert result is not None
    assert result["success"] is True
    assert result["before_tokens"] == 9000
    assert result["after_tokens"] == 4500
    assert result["after_chars"] < result["before_chars"]
    assert store.read_long_term() == expected
    provider.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_compact_memory_md_preserves_structure(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_long_term(_large_memory())

    provider = MagicMock()
    provider.count_tokens = AsyncMock(side_effect=[8500, 4200])
    provider.chat = AsyncMock(
        return_value=_tool_response(f"```markdown\n{_compacted_memory()}```")
    )

    result = await store.compact_memory_md(provider=provider, model="test-model")

    memory_text = store.read_long_term()
    assert result is not None
    assert result["success"] is True
    assert store._is_valid_canonical_structure(memory_text)
    for section in store._CANONICAL_SECTIONS:
        assert f"## {section}" in memory_text


@pytest.mark.asyncio
async def test_compact_memory_md_llm_failure(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    original = _large_memory()
    store.write_long_term(original)

    provider = MagicMock()
    provider.count_tokens = AsyncMock(return_value=9500)
    provider.chat = AsyncMock(side_effect=RuntimeError("haiku failed"))

    result = await store.compact_memory_md(provider=provider, model="test-model")

    assert result is not None
    assert result["success"] is False
    assert result["before_tokens"] == 9500
    assert result["after_tokens"] == 9500
    assert store.read_long_term() == original


@pytest.mark.asyncio
async def test_compact_memory_md_token_counting_fallback(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    original = _large_memory(repeats=2400)
    store.write_long_term(original)
    expected = store._coerce_to_canonical(_compacted_memory())

    provider = MagicMock()
    provider.count_tokens = AsyncMock(side_effect=RuntimeError("count failed"))
    provider.chat = AsyncMock(return_value=_tool_response(_compacted_memory()))

    result = await store.compact_memory_md(provider=provider, model="test-model")

    assert result is not None
    assert result["success"] is True
    assert result["before_tokens"] == store._estimate_tokens(original)
    assert result["after_tokens"] == store._estimate_tokens(expected)


@pytest.mark.asyncio
async def test_compaction_pipeline_includes_memory_md(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop._run_agent_loop = AsyncMock(return_value=("ok", [], []))
    expected = loop.context.memory._coerce_to_canonical(_compacted_memory())

    loop.provider.count_tokens = AsyncMock(side_effect=[9000, 4800])
    loop.provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(content=_structured_summary()),
            _tool_response(_compacted_memory()),
        ]
    )

    async def _fake_consolidate(sess, archive_all: bool = False, extraction_range=None):
        if extraction_range:
            sess.last_consolidated = extraction_range[1]
        return True

    loop._consolidate_memory = _fake_consolidate
    loop.context.memory.write_long_term(_large_memory())

    session = loop.sessions.get_or_create("cli:test")
    _add_long_messages(session, turns=40)
    session.metadata["usage_snapshot"] = {
        "total_input_tokens": 180_000,
        "message_index": len(session.messages),
    }
    loop.sessions.save(session)

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="continue")
    )

    session_path = loop.sessions._get_session_path(session.key)
    entries = load_compaction_log(session_path)
    assert entries
    memory_compaction = entries[-1].get("memory_md_compaction")
    assert isinstance(memory_compaction, dict)
    assert memory_compaction["success"] is True
    assert memory_compaction["before_tokens"] == 9000
    assert memory_compaction["after_tokens"] == 4800
    assert loop.context.memory.read_long_term() == expected
