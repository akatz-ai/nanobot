from __future__ import annotations

import copy
import json
from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent import context as context_module
from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.compaction import SUMMARIZATION_SYSTEM_PROMPT, estimate_message_tokens
from nanobot.session.manager import SessionManager


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 3, 6, 21, 18)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


class _CapturingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key=None, api_base=None)
        self.regular_calls: list[list[dict]] = []
        self.summary_calls: list[list[dict]] = []
        self._reply_counter = 0
        self._summary_counter = 0

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        _ = (tools, model, max_tokens, temperature, reasoning_effort)
        snapshot = copy.deepcopy(messages)
        if messages and messages[0].get("content") == SUMMARIZATION_SYSTEM_PROMPT:
            self.summary_calls.append(snapshot)
            self._summary_counter += 1
            return LLMResponse(content=_structured_summary(f"Summary {self._summary_counter}"))

        self.regular_calls.append(snapshot)
        self._reply_counter += 1
        prompt_tokens = sum(estimate_message_tokens(msg) for msg in messages)
        return LLMResponse(
            content=f"assistant-{self._reply_counter} " + ("z" * 4000),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 200,
            },
        )

    def get_default_model(self) -> str:
        return "capture-model"


def _structured_summary(label: str = "Summary") -> str:
    return (
        f"## Goal\n{label}\n\n"
        "## Constraints & Preferences\n- preserve context\n\n"
        "## Progress\n"
        "### Done\n- [x] previous work summarized\n"
        "### In Progress\n- [ ] continue session\n"
        "### Blocked\n- none\n\n"
        "## Key Decisions\n- **Cache layout**: move turn context inline\n\n"
        "## Next Steps\n1. Continue from preserved context\n\n"
        "## Critical Context\n- Keep exact paths and decisions"
    )


def _write_workspace(workspace: Path) -> None:
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text("# Agent rules\n", encoding="utf-8")
    (workspace / "SOUL.md").write_text("# Soul\nStay focused.\n", encoding="utf-8")
    (workspace / "USER.md").write_text("# User\nPrefers concise output.\n", encoding="utf-8")
    (workspace / "TOOLS.md").write_text("# Tools\nUse tools when needed.\n", encoding="utf-8")
    (workspace / "IDENTITY.md").write_text(
        "- **Name:** nanobot\n- **Emoji:** :)\n",
        encoding="utf-8",
    )
    (workspace / "memory" / "MEMORY.md").write_text(
        "## Memory\n- Stable fact\n- Cache optimization matters\n",
        encoding="utf-8",
    )


def _build_v1_messages(
    builder: ContextBuilder,
    *,
    history: list[dict],
    current_message: str,
    channel: str,
    chat_id: str,
    memory_context: str | None,
) -> list[dict]:
    messages = [{"role": "system", "content": builder.build_system_prompt()}]
    runtime_ctx = builder._build_runtime_context(channel, chat_id)
    messages.append({"role": "system", "content": f"## Current Session\n{runtime_ctx}"})

    long_term_memory = builder.memory.read_long_term()
    if long_term_memory:
        messages.append(
            {
                "role": "system",
                "content": builder._build_memory_data_message(
                    "Long-term Memory (file)",
                    long_term_memory,
                    "memory_file_data",
                ),
            }
        )

    messages.extend(history)

    memory_block = builder._build_retrieved_memory_block(memory_context)
    if memory_block:
        messages.append(
            {
                "role": "system",
                "content": memory_block.replace(
                    "[Relevant Retrieved Memory]",
                    "## Relevant Retrieved Memory (facts)",
                ),
            }
        )

    messages.append({"role": "user", "content": current_message})
    return messages


def _system_contents(messages: list[dict]) -> list[str]:
    return [msg["content"] for msg in messages if msg.get("role") == "system"]


def test_v2_user_turn_context_includes_live_models(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="what model are you?",
        channel="discord",
        chat_id="123",
        model="openai-codex/gpt-5.4",
        background_model="anthropic-direct/claude-haiku-4-5",
        memory_context="[fact] The default model is anthropic-direct/claude-opus-4-6.",
    )

    user_message = messages[-1]["content"]
    assert isinstance(user_message, str)
    assert "Active Model: openai-codex/gpt-5.4" in user_message
    assert "Background Model: anthropic-direct/claude-haiku-4-5" in user_message
    assert "what model are you?" in user_message


def test_context_v1_vs_v2_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)
    builder = ContextBuilder(workspace)
    history = [{"role": "assistant", "content": "previous reply"}]

    monkeypatch.setattr(context_module, "datetime", _FakeDatetime)

    _FakeDatetime.current = real_datetime(2026, 3, 6, 21, 18)
    v1_turn1 = _build_v1_messages(
        builder,
        history=history,
        current_message="hello",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact A",
    )
    v2_turn1 = builder.build_messages(
        history=history,
        current_message="hello",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact A",
    )

    _FakeDatetime.current = real_datetime(2026, 3, 6, 21, 19)
    v1_turn2 = _build_v1_messages(
        builder,
        history=history,
        current_message="goodbye",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact B",
    )
    v2_turn2 = builder.build_messages(
        history=history,
        current_message="goodbye",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact B",
    )

    assert len(_system_contents(v2_turn1)) < len(_system_contents(v1_turn1))
    assert _system_contents(v2_turn1) == _system_contents(v2_turn2)
    assert _system_contents(v1_turn1) != _system_contents(v1_turn2)


def test_cache_stability_across_turns(tmp_path: Path) -> None:
    builder = ContextBuilder(tmp_path)

    turn1 = builder.build_messages(
        history=[],
        current_message="hello",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact A",
    )
    turn2 = builder.build_messages(
        history=[],
        current_message="goodbye",
        channel="discord",
        chat_id="123",
        memory_context="- [fact] Fact B",
    )

    assert _system_contents(turn1) == _system_contents(turn2)
    assert turn1[-1]["content"] != turn2[-1]["content"]
    assert "[Relevant Retrieved Memory]" in turn1[-1]["content"]
    assert "[Relevant Retrieved Memory]" in turn2[-1]["content"]


def test_retrieved_memory_prepended_to_user_message(tmp_path: Path) -> None:
    builder = ContextBuilder(tmp_path)

    messages = builder.build_messages(
        history=[],
        current_message="what's the API auth?",
        memory_context="- [fact] API uses RS256 JWT\n- [decision] 24h token expiry",
        channel="discord",
        chat_id="123",
    )

    system_msgs = [m for m in messages if m["role"] == "system"]
    for sm in system_msgs:
        assert "Retrieved Memory" not in sm["content"]

    user_msg = [m for m in messages if m["role"] == "user"][-1]
    assert "[Relevant Retrieved Memory]" in user_msg["content"]
    assert "RS256 JWT" in user_msg["content"]
    assert "what's the API auth?" in user_msg["content"]


def test_no_daily_history_in_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)
    history_dir = workspace / "memory" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{real_datetime.now().date().isoformat()}.md").write_text(
        "# Today\nDaily content should not appear.",
        encoding="utf-8",
    )

    builder = ContextBuilder(
        workspace,
        memory_graph_config={"consolidation": {"engine": "hybrid"}},
    )
    messages = builder.build_messages(history=[], current_message="hello")

    assert not any("Daily History" in str(msg.get("content", "")) for msg in messages)
    assert not any("<daily_history_data>" in str(msg.get("content", "")) for msg in messages)


def test_compaction_summary_in_static_prefix(tmp_path: Path) -> None:
    builder = ContextBuilder(tmp_path)
    summary = (
        "## Goal\nBuild extraction pipeline\n\n"
        "## Progress\n### Done\n- [x] prefix moved\n\n"
        "## Next Steps\n1. Continue"
    )

    messages = builder.build_messages(
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "previous reply"},
        ],
        current_message="continue",
        extra_system_messages=[summary],
    )

    history_start = next(i for i, msg in enumerate(messages) if msg["role"] != "system")
    summary_idx = next(
        i
        for i, msg in enumerate(messages)
        if msg["role"] == "system" and "## Goal" in msg["content"]
    )
    assert summary_idx < history_start


@pytest.mark.asyncio
async def test_full_session_e2e_with_compaction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)

    provider = _CapturingProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model="capture-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    retrieval_map: dict[str, str] = {}
    loop._retrieve_memory_context = AsyncMock(side_effect=lambda session, text: retrieval_map.get(text))

    def _payload(label: str) -> str:
        return f"{label}: " + ("x" * 10000)

    for i in range(1, 6):
        message = _payload(f"turn-{i}")
        retrieval_map[message] = f"- [fact] retrieval for turn {i}"
        response = await loop._process_message(
            InboundMessage(
                channel="discord",
                sender_id="user",
                chat_id="123",
                content=message,
            )
        )
        assert response is not None
        sent = provider.regular_calls[-1]
        system_messages = _system_contents(sent)
        assert len(system_messages) == 2
        assert not any("Daily History" in content for content in system_messages)
        assert "[Current Session]" not in "\n".join(system_messages)
        user_messages = [msg for msg in sent if msg["role"] == "user"]
        assert user_messages[-1]["content"].startswith("[Current Session]\n")
        assert f"retrieval for turn {i}" in user_messages[-1]["content"]
        assert "Time:" in user_messages[-1]["content"]

    pre_compaction_prefixes = [_system_contents(call) for call in provider.regular_calls[:5]]
    assert all(prefix == pre_compaction_prefixes[0] for prefix in pre_compaction_prefixes[1:])

    session = loop.sessions.get_or_create("discord:123")
    pressure_payload = "y" * 40000
    for i in range(6, 36):
        session.add_message("user", f"pressure-user-{i} {pressure_payload}")
        session.add_message("assistant", f"pressure-assistant-{i} {pressure_payload}")

    turn6 = _payload("turn-6")
    retrieval_map[turn6] = "- [fact] retrieval for turn 6"
    response = await loop._process_message(
        InboundMessage(
            channel="discord",
            sender_id="user",
            chat_id="123",
            content=turn6,
        )
    )
    assert response is not None
    assert provider.summary_calls

    post_compaction_call = provider.regular_calls[-1]
    post_compaction_system = _system_contents(post_compaction_call)
    assert len(post_compaction_system) == 3
    assert any("## Goal" in content for content in post_compaction_system)
    assert post_compaction_system[:2] == pre_compaction_prefixes[0]
    assert any(msg["role"] == "assistant" for msg in post_compaction_call[3:-1])
    assert post_compaction_call[-1]["role"] == "user"
    assert "[Current Session]" in post_compaction_call[-1]["content"]
    assert "retrieval for turn 6" in post_compaction_call[-1]["content"]

    turn7 = _payload("turn-7")
    retrieval_map[turn7] = "- [fact] retrieval for turn 7"
    response = await loop._process_message(
        InboundMessage(
            channel="discord",
            sender_id="user",
            chat_id="123",
            content=turn7,
        )
    )
    assert response is not None

    assert _system_contents(provider.regular_calls[-2]) == _system_contents(provider.regular_calls[-1])
    assert provider.regular_calls[-2][-1]["content"] != provider.regular_calls[-1][-1]["content"]
    assert not any(
        "Daily History" in str(msg.get("content", ""))
        for call in provider.regular_calls
        for msg in call
    )

    persisted_session = loop.sessions.get_or_create("discord:123")
    persisted_user_messages = [
        msg for msg in persisted_session.messages if msg.get("role") == "user"
    ]
    prefixed_persisted_user_messages = [
        msg for msg in persisted_user_messages if isinstance(msg.get("content"), str) and msg["content"].startswith("[Current Session]\n")
    ]
    assert prefixed_persisted_user_messages
    assert any("retrieval for turn 6" in msg["content"] for msg in prefixed_persisted_user_messages)
    assert any("retrieval for turn 7" in msg["content"] for msg in prefixed_persisted_user_messages)


def test_old_sessions_with_system_messages_still_work(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    key = "cli:legacy"
    path = manager.get_session_path(key)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_type": "metadata",
                        "key": key,
                        "created_at": "2026-03-01T00:00:00",
                        "updated_at": "2026-03-01T00:00:00",
                        "metadata": {},
                        "last_consolidated": 0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"role": "system", "content": "## Current Session\nlegacy"}, ensure_ascii=False),
                json.dumps({"role": "system", "content": "<daily_history_data>legacy</daily_history_data>"}, ensure_ascii=False),
                json.dumps({"role": "user", "content": "message 1"}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": "response 1"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = manager.get_or_create(key)
    history, _ = session.get_history(max_messages=10)

    assert history[0]["role"] == "system"
    assert history[1]["role"] == "system"

    builder = ContextBuilder(tmp_path)
    messages = builder.build_messages(
        history=history,
        current_message="next question",
        channel="cli",
        chat_id="legacy",
        memory_context="- [fact] Legacy sessions still work",
    )

    assert any(msg.get("role") == "system" and "legacy" in msg.get("content", "") for msg in messages)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("[Current Session]\n")


def test_get_history_no_summary_injection(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:test")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session.append_compaction(
        summary=_structured_summary("Injected?"),
        first_kept_index=1,
        tokens_before=50_000,
        file_ops={"read_files": [], "modified_files": []},
    )

    history, _ = session.get_history(max_messages=100, context_window=200_000)

    for msg in history:
        if msg["role"] == "system" and "## Goal" in msg.get("content", ""):
            pytest.fail("get_history() should not inject compaction summary in V2")


def test_prompt_assembly_result_classifies_static_dynamic_and_history_sections(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)
    builder = ContextBuilder(workspace)

    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    messages = builder.build_messages(
        history=history,
        current_message="current question",
        channel="discord",
        chat_id="123",
        model="openai-codex/gpt-5.4",
        background_model="openai-codex/gpt-5.3-codex-spark",
        memory_context="- [fact] Prior decision",
        extra_system_messages=[_structured_summary("Session Summary")],
    )

    assembly = builder.build_prompt_assembly_result(
        session_key="discord:123",
        provider_name="FakeProvider",
        model_name="openai-codex/gpt-5.4",
        messages=messages,
        context_window=200_000,
        reserve_tokens=16_384,
        compaction_threshold_ratio=0.75,
        emergency_trim_ratio=0.95,
        current_message="current question",
    )

    assert assembly.session_key == "discord:123"
    assert assembly.budget.compaction_trigger_tokens == 150_000
    assert assembly.pre_compaction_snapshot["trigger_snapshot"] == "pre_compaction"
    assert assembly.pre_compaction_snapshot["stable_cached_prefix_tokens"] > 0
    assert assembly.pre_compaction_snapshot["dynamic_turn_tokens"] > 0
    assert assembly.pre_compaction_snapshot["visible_conversation_slice_tokens"] > 0

    kinds = [section.kind for section in assembly.sections]
    assert "system_base" in kinds
    assert "memory_md" in kinds
    assert "session_summary" in kinds
    assert "current_user" in kinds
    assert "history_user" in kinds
    assert "history_assistant" in kinds
