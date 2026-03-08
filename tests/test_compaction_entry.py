import json
from pathlib import Path

import pytest

from nanobot.session.manager import SessionManager


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path)


def _populate_messages(session) -> None:
    session.add_message("user", "u0")
    session.add_message("assistant", "a0")
    session.add_message("user", "u1")
    session.add_message("assistant", "a1")


def test_compaction_entry_roundtrip(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:compaction_roundtrip")
    _populate_messages(session)
    session.append_compaction(
        summary="## Goal\nPersist me\n\n## Progress\n### Done\n- [x] A\n\n## Next Steps\n1. B",
        first_kept_index=2,
        tokens_before=2500,
        file_ops={"read_files": ["/tmp/read.py"], "modified_files": ["/tmp/write.py"]},
    )
    session_manager.save(session)

    session_manager.invalidate(session.key)
    loaded = session_manager.get_or_create(session.key)
    compaction = loaded.get_last_compaction()

    assert compaction is not None
    assert compaction.first_kept_index == 2
    assert compaction.tokens_before == 2500
    assert "/tmp/read.py" in compaction.file_ops["read_files"]


def test_get_history_with_compaction_returns_conversation_only(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:history_with_compaction")
    _populate_messages(session)
    session.append_compaction(
        summary="## Goal\nCompact\n\n## Progress\n### Done\n- [x] A\n\n## Next Steps\n1. B",
        first_kept_index=2,
        tokens_before=100,
        file_ops={"read_files": [], "modified_files": []},
    )

    history, _ = session.get_history(max_messages=50)

    assert not any(
        m.get("role") == "system" and "## Goal" in m.get("content", "")
        for m in history
    )
    assert [m["content"] for m in history] == ["u1", "a1"]


def test_get_history_without_compaction_is_backward_compatible(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:history_without_compaction")
    _populate_messages(session)

    history, _ = session.get_history(max_messages=2)

    assert [m["content"] for m in history] == ["u1", "a1"]


def test_get_last_compaction_none_when_absent(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:no_compaction")
    _populate_messages(session)

    assert session.get_last_compaction() is None


def test_get_last_compaction_clamps_invalid_first_kept_index(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:clamp_compaction")
    _populate_messages(session)
    session.compactions.append(
        {
            "summary": "## Goal\nClamp\n\n## Progress\n### Done\n- [x] A\n\n## Next Steps\n1. B",
            "first_kept_index": 999,
            "tokens_before": 10,
            "file_ops": {"read_files": [], "modified_files": []},
            "previous_summary": None,
            "timestamp": "2026-01-01T00:00:00",
        }
    )

    compaction = session.get_last_compaction()

    assert compaction is not None
    assert compaction.first_kept_index == len(session.messages)



def test_append_compaction_writes_jsonl_directly(session_manager: SessionManager) -> None:
    session = session_manager.get_or_create("test:append_compaction_jsonl")
    _populate_messages(session)

    session.append_compaction(
        summary="## Goal\nWrite line\n\n## Progress\n### Done\n- [x] A\n\n## Next Steps\n1. B",
        first_kept_index=1,
        tokens_before=500,
        file_ops={"read_files": ["a"], "modified_files": ["b"]},
    )

    path = session_manager._get_session_path(session.key)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[-1].get("_type") == "compaction"
    assert lines[-1]["first_kept_index"] == 1
    assert lines[-1]["tokens_before"] == 500
