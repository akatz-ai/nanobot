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
