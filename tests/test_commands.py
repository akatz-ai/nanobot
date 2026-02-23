import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.agent.context import ContextBuilder
from nanobot.cli.commands import app
from nanobot.config.schema import Config
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import (
    _convert_messages,
    _prompt_cache_key,
    _strip_model_prefix,
)
from nanobot.providers.registry import find_by_model

runner = CliRunner()


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
