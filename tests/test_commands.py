import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.agent.context import ContextBuilder
from nanobot.cli.commands import app
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.claude_code_provider import ClaudeCodeProvider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import (
    _convert_messages,
    _prompt_cache_key,
    _strip_model_prefix,
)
from nanobot.providers.registry import find_by_model
from nanobot.session.manager import Session

runner = CliRunner()


class _StubProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)
        self._calls = 0

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self._calls += 1
        return LLMResponse(content=f"reply-{self._calls}")

    def get_default_model(self) -> str:
        return "stub-model"


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_config_matches_github_copilot_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "github-copilot/gpt-5.3-codex"

    assert config.get_provider_name() == "github_copilot"


def test_config_matches_openai_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.1-codex"

    assert config.get_provider_name() == "openai_codex"


def test_find_by_model_prefers_explicit_prefix_over_generic_codex_keyword():
    spec = find_by_model("github-copilot/gpt-5.3-codex")

    assert spec is not None
    assert spec.name == "github_copilot"


def test_litellm_provider_canonicalizes_github_copilot_hyphen_prefix():
    provider = LiteLLMProvider(default_model="github-copilot/gpt-5.3-codex")

    resolved = provider._resolve_model("github-copilot/gpt-5.3-codex")

    assert resolved == "github_copilot/gpt-5.3-codex"


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


def test_context_build_messages_places_retrieved_memory_in_separate_system_message(tmp_path: Path):
    builder = ContextBuilder(tmp_path)

    messages = builder.build_messages(
        history=[{"role": "assistant", "content": "previous"}],
        current_message="current question",
        channel="cli",
        chat_id="test",
        memory_context="- Fact A\n- fact a\n- Fact B",
    )

    assert messages[0]["role"] == "system"
    assert "Relevant Retrieved Memory" not in messages[0]["content"]
    assert messages[-2]["role"] == "system"
    assert "Relevant Retrieved Memory (facts)" in messages[-2]["content"]
    assert messages[-1] == {"role": "user", "content": "current question"}

    bullet_lines = [line for line in messages[-2]["content"].splitlines() if line.startswith("- ")]
    assert bullet_lines == ["- Fact A", "- Fact B"]


def test_context_base_system_prompt_is_stable_across_turns(tmp_path: Path):
    builder = ContextBuilder(tmp_path)

    turn1 = builder.build_messages(
        history=[],
        current_message="hello",
        channel="cli",
        chat_id="test",
        memory_context="first retrieval",
    )
    turn2 = builder.build_messages(
        history=[],
        current_message="hello again",
        channel="cli",
        chat_id="test",
        memory_context="second retrieval with different content",
    )

    assert turn1[0]["content"] == turn2[0]["content"]
    assert "Current Time" not in turn1[0]["content"]
    assert turn1[-2]["content"] != turn2[-2]["content"]


def test_retrieved_memory_guardrails_cap_and_truncate():
    lines = [f"- repeated fact line {i} " + ("x" * 200) for i in range(30)]
    memory_context = "\n".join(lines)

    rendered = ContextBuilder._build_retrieved_memory_message(memory_context)

    assert rendered is not None
    assert rendered.count("\n- ") <= 13  # max 12 bullets + optional "(truncated)"
    assert "(truncated)" in rendered


def test_codex_concatenates_all_system_messages_in_order():
    messages = [
        {"role": "system", "content": "Base system"},
        {"role": "system", "content": "Retrieved memory"},
        {"role": "user", "content": "hello"},
    ]

    system_prompt, input_items = _convert_messages(messages)

    assert system_prompt == "Base system\n\nRetrieved memory"
    assert input_items == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def test_codex_prompt_cache_key_uses_stable_prefix_only():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    turn1 = [
        {"role": "system", "content": "Static instructions"},
        {"role": "system", "content": "Retrieved memory A"},
        {"role": "user", "content": "question A"},
    ]
    turn2 = [
        {"role": "system", "content": "Static instructions"},
        {"role": "system", "content": "Retrieved memory B"},
        {"role": "user", "content": "question B"},
        {"role": "assistant", "content": "answer B"},
    ]

    key1 = _prompt_cache_key(turn1, tools)
    key2 = _prompt_cache_key(turn2, tools)
    key3 = _prompt_cache_key(
        [{"role": "system", "content": "Different static instructions"}, {"role": "user", "content": "q"}],
        tools,
    )

    assert key1 == key2
    assert key1 != key3


def test_litellm_cache_control_marks_only_first_system_message():
    provider = LiteLLMProvider(default_model="anthropic/claude-opus-4-5")
    messages = [
        {"role": "system", "content": "Static system"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "Dynamic retrieved memory"},
    ]
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}},
    ]

    new_messages, new_tools = provider._apply_cache_control(messages, tools)

    assert isinstance(new_messages[0]["content"], list)
    assert new_messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert new_messages[2]["content"] == "Dynamic retrieved memory"
    assert new_tools is not None
    assert new_tools[-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_agent_loop_stores_each_user_message_once(tmp_path: Path):
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
    )

    await agent.process_direct("first", session_key="cli:test", channel="cli", chat_id="test")
    await agent.process_direct("second", session_key="cli:test", channel="cli", chat_id="test")

    session = agent.sessions.get_or_create("cli:test")
    roles = [m.get("role") for m in session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]

    user_messages = [m.get("content") for m in session.messages if m.get("role") == "user"]
    assert user_messages == ["first", "second"]


@pytest.mark.asyncio
async def test_consolidate_memory_uses_hybrid_engine_path(tmp_path: Path):
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
        memory_window=10,
    )
    agent._memory_graph_config = {"consolidation": {"engine": "hybrid"}}

    hybrid = SimpleNamespace(compact=AsyncMock(return_value=SimpleNamespace()))
    agent._memory_module = SimpleNamespace(initialized=True, hybrid=hybrid, consolidator=None)

    session = Session(key="cli:test")
    for i in range(12):
        session.add_message("user", f"msg-{i}")

    with patch("nanobot.agent.loop.MemoryStore.consolidate", new_callable=AsyncMock) as legacy_consolidate:
        ok = await agent._consolidate_memory(session, archive_all=False)

    assert ok is True
    legacy_consolidate.assert_not_called()
    hybrid.compact.assert_awaited_once()
    assert session.last_consolidated == 7


@pytest.mark.asyncio
async def test_consolidate_memory_keeps_legacy_path(tmp_path: Path):
    agent = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
        memory_window=10,
    )
    agent._memory_graph_config = {"consolidation": {"engine": "legacy"}}
    consolidator = SimpleNamespace(consolidate_session=AsyncMock(return_value={"added": 1}))
    agent._memory_module = SimpleNamespace(initialized=True, hybrid=None, consolidator=consolidator)

    session = Session(key="cli:test")
    for i in range(12):
        session.add_message("user", f"msg-{i}")

    with patch(
        "nanobot.agent.loop.MemoryStore.consolidate",
        new_callable=AsyncMock,
        return_value=True,
    ) as legacy_consolidate:
        ok = await agent._consolidate_memory(session, archive_all=False)

    assert ok is True
    legacy_consolidate.assert_awaited_once()
    consolidator.consolidate_session.assert_awaited_once()


def test_context_builder_includes_daily_history_in_hybrid_mode(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    history_dir = memory_dir / "history"
    history_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text("## Memory\n- Long-term fact", encoding="utf-8")

    today = datetime.now().date().isoformat()
    (history_dir / f"{today}.md").write_text(
        f"# {today}\n\n## 14:30 — Compaction (telegram:1)\n\n- [fact] Daily event",
        encoding="utf-8",
    )

    builder = ContextBuilder(
        tmp_path,
        memory_graph_config={"consolidation": {"engine": "hybrid"}},
    )
    messages = builder.build_messages(history=[], current_message="hello")

    system_messages = [m["content"] for m in messages if m.get("role") == "system"]
    assert any("Long-term Memory (file)" in content for content in system_messages)
    assert any("Daily History (today)" in content for content in system_messages)


@pytest.mark.asyncio
async def test_claude_code_provider_persists_session_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    provider = ClaudeCodeProvider(
        default_model="claude-code/claude-sonnet-4-5",
        workspace=tmp_path,
    )

    async def _fake_run_query(prompt, options):
        return "ok", "stop", {"total_tokens": 1}, "session-123"

    monkeypatch.setattr(provider, "_run_query", _fake_run_query)

    response = await provider.chat(
        messages=[
            {"role": "system", "content": "Channel: cli\nChat ID: 42"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert response.content == "ok"

    session_ids_path = tmp_path / ".claude_session_ids.json"
    assert session_ids_path.exists()
    data = json.loads(session_ids_path.read_text(encoding="utf-8"))
    assert data == {"cli:42": "session-123"}

    reloaded = ClaudeCodeProvider(
        default_model="claude-code/claude-sonnet-4-5",
        workspace=tmp_path,
    )
    assert reloaded._session_ids == {"cli:42": "session-123"}
