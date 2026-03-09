"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_turn_context_is_prepended_to_user_message(tmp_path) -> None:
    """Runtime metadata should be inline on the user message, not a system message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    # First message is the main system prompt (no session metadata baked in)
    assert messages[0]["role"] == "system"
    assert "[Current Session]" not in messages[0]["content"]

    system_contents = [m.get("content", "") for m in messages if m["role"] == "system"]
    assert not any("[Current Session]" in content for content in system_contents)

    # Last message should contain the prepended turn context and actual input
    assert messages[-1]["role"] == "user"
    assert "[Current Session]" in messages[-1]["content"]
    assert "Channel: cli | Chat ID: direct" in messages[-1]["content"]
    assert "Time:" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Return exactly: OK")


def test_system_prompt_cache_reuses_prompt_until_inputs_change(tmp_path, monkeypatch) -> None:
    workspace = _make_workspace(tmp_path)
    (workspace / "IDENTITY.md").write_text("- **Name:** cachebot\n", encoding="utf-8")
    (workspace / "memory").mkdir(parents=True, exist_ok=True)
    (workspace / "memory" / "MEMORY.md").write_text("stable memory\n", encoding="utf-8")

    builder = ContextBuilder(workspace)
    calls = {"identity": 0, "bootstrap": 0}

    original_identity = builder._get_identity
    original_bootstrap = builder._load_bootstrap_files

    def _count_identity() -> str:
        calls["identity"] += 1
        return original_identity()

    def _count_bootstrap() -> str:
        calls["bootstrap"] += 1
        return original_bootstrap()

    monkeypatch.setattr(builder, "_get_identity", _count_identity)
    monkeypatch.setattr(builder, "_load_bootstrap_files", _count_bootstrap)

    prompt1 = builder.build_system_prompt()
    prompt2 = builder.build_system_prompt()
    builder.invalidate_prompt_cache()
    prompt3 = builder.build_system_prompt()
    (workspace / "IDENTITY.md").write_text("- **Name:** cachebot-v2\n", encoding="utf-8")
    prompt4 = builder.build_system_prompt()

    assert prompt1 == prompt2
    assert prompt1 == prompt3
    assert "cachebot-v2" in prompt4
    assert calls == {"identity": 3, "bootstrap": 3}
