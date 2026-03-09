from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import LLMResponse
from nanobot.session.compaction import (
    MAX_PREVIOUS_SUMMARY_CHARS,
    MAX_SUMMARY_CHARS,
    TARGET_SUMMARY_CHARS,
    UPDATE_SUMMARIZATION_PROMPT,
    extract_file_ops,
    generate_compaction_summary,
    serialize_conversation,
)


def test_serialize_conversation_mixed_messages() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Let me check."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "tc1", "content": "contents"},
        {"role": "assistant", "content": "Found it."},
    ]

    transcript = serialize_conversation(messages)
    assert "[User]: Hello" in transcript
    assert "[Assistant]: Let me check." in transcript
    assert "[Assistant]: (tool calls: read_file)" in transcript
    assert "[Tool:read_file]: contents" in transcript


def test_serialize_conversation_handles_none_assistant_content() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "tc2", "function": {"name": "list_dir", "arguments": "{}"}},
            ],
        }
    ]

    transcript = serialize_conversation(messages)
    assert "(tool calls: read_file, list_dir)" in transcript


def test_serialize_conversation_handles_user_image_blocks() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "Please inspect"},
            ],
        }
    ]

    transcript = serialize_conversation(messages)
    assert "[image attached]" in transcript
    assert "Please inspect" in transcript


def test_serialize_conversation_truncates_from_beginning() -> None:
    messages = [{"role": "user", "content": f"line-{i}"} for i in range(200)]
    transcript = serialize_conversation(messages, max_transcript_chars=200)

    assert len(transcript) <= 200
    assert "line-0" not in transcript
    assert "line-199" in transcript


def test_extract_file_ops_categorizes_and_requires_tool_result() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_read",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                },
                {
                    "id": "tc_write",
                    "function": {"name": "write_file", "arguments": '{"path": "b.py"}'},
                },
                {
                    "id": "tc_edit",
                    "function": {"name": "edit_file", "arguments": '{"path": "c.py"}'},
                },
                {
                    "id": "tc_exec",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command": "echo ok > out.txt"}',
                    },
                },
                {
                    "id": "tc_missing",
                    "function": {"name": "write_file", "arguments": '{"path": "missing.py"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc_read", "name": "read_file", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc_write", "name": "write_file", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc_edit", "name": "edit_file", "content": "ok"},
        {"role": "tool", "tool_call_id": "tc_exec", "name": "exec", "content": "ok"},
    ]

    ops = extract_file_ops(messages)
    assert "a.py" in ops["read_files"]
    assert "b.py" in ops["modified_files"]
    assert "c.py" in ops["modified_files"]
    assert "out.txt" in ops["modified_files"]
    assert "missing.py" not in ops["modified_files"]


@pytest.mark.asyncio
async def test_generate_compaction_summary_returns_structured_output() -> None:
    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(
        content=(
            "## Goal\nTest goal\n\n"
            "## Constraints & Preferences\n- none\n\n"
            "## Progress\n### Done\n- [x] Done\n### In Progress\n- [ ] Work\n### Blocked\n- none\n\n"
            "## Key Decisions\n- **A**: B\n\n"
            "## Next Steps\n1. Continue\n\n"
            "## Critical Context\n- Keep X"
        )
    )

    summary = await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="test-model",
    )

    assert "## Goal" in summary
    assert "## Progress" in summary
    assert "## Next Steps" in summary


@pytest.mark.asyncio
async def test_generate_compaction_summary_uses_update_prompt_with_previous_summary() -> None:
    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(
        content=(
            "## Goal\nUpdated\n\n"
            "## Constraints & Preferences\n- C\n\n"
            "## Progress\n### Done\n- [x] D\n### In Progress\n- [ ] I\n### Blocked\n- B\n\n"
            "## Key Decisions\n- **K**: R\n\n"
            "## Next Steps\n1. N\n\n"
            "## Critical Context\n- X"
        )
    )

    previous_summary = "## Goal\nOld goal"
    await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="test-model",
        previous_summary=previous_summary,
    )

    assert provider.chat.await_count == 1
    call = provider.chat.await_args
    prompt = call.kwargs["messages"][1]["content"]
    assert "<previous_summary>" in prompt
    assert previous_summary in prompt
    assert UPDATE_SUMMARIZATION_PROMPT.splitlines()[0] in prompt


@pytest.mark.asyncio
async def test_generate_compaction_summary_forwards_reasoning_effort() -> None:
    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(
        content=(
            "## Goal\nUpdated\n\n"
            "## Constraints & Preferences\n- C\n\n"
            "## Progress\n### Done\n- [x] D\n### In Progress\n- [ ] I\n### Blocked\n- B\n\n"
            "## Key Decisions\n- **K**: R\n\n"
            "## Next Steps\n1. N\n\n"
            "## Critical Context\n- X"
        )
    )

    await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="openai-codex/gpt-5.3-codex-spark",
        reasoning_effort="high",
    )

    assert provider.chat.await_count == 1
    assert provider.chat.await_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_generate_compaction_summary_retries_and_falls_back_on_malformed_output() -> None:
    provider = AsyncMock()
    provider.chat.side_effect = [
        LLMResponse(content="not structured"),
        LLMResponse(content="still bad"),
    ]

    summary = await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="test-model",
    )

    assert provider.chat.await_count == 2
    assert "## Goal" in summary
    assert "## Progress" in summary
    assert "## Next Steps" in summary
    assert "fallback" in summary.lower()


@pytest.mark.asyncio
async def test_generate_compaction_summary_compresses_oversized_previous_summary() -> None:
    provider = AsyncMock()
    compressed_previous = (
        "## Goal\nCompacted old summary\n\n"
        "## Constraints & Preferences\n- concise\n\n"
        "## Progress\n### Done\n- [x] prior work grouped\n### In Progress\n- [ ] continue\n### Blocked\n- none\n\n"
        "## Key Decisions\n- **Keep**: essentials only\n\n"
        "## Next Steps\n1. Continue\n\n"
        "## Critical Context\n- carry forward key identifiers"
    )
    updated_summary = (
        "## Goal\nUpdated\n\n"
        "## Constraints & Preferences\n- C\n\n"
        "## Progress\n### Done\n- [x] D\n### In Progress\n- [ ] I\n### Blocked\n- B\n\n"
        "## Key Decisions\n- **K**: R\n\n"
        "## Next Steps\n1. N\n\n"
        "## Critical Context\n- X"
    )
    provider.chat.side_effect = [
        LLMResponse(content=compressed_previous),
        LLMResponse(content=updated_summary),
    ]

    previous_summary = "LEGACY_DETAIL " * ((MAX_PREVIOUS_SUMMARY_CHARS // 14) + 200)
    summary = await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="test-model",
        previous_summary=previous_summary,
    )

    assert provider.chat.await_count == 2
    second_prompt = provider.chat.await_args_list[1].kwargs["messages"][1]["content"]
    assert compressed_previous in second_prompt
    assert len(second_prompt) < len(previous_summary)
    assert len(summary) <= MAX_SUMMARY_CHARS
    assert "## Goal" in summary
    assert "## Progress" in summary
    assert "## Next Steps" in summary


@pytest.mark.asyncio
async def test_generate_compaction_summary_recompresses_oversized_output() -> None:
    provider = AsyncMock()
    oversized_summary = (
        "## Goal\nBig\n\n"
        "## Constraints & Preferences\n- cap\n\n"
        "## Progress\n### Done\n- [x] "
        + ("very long detail " * 1500)
        + "\n### In Progress\n- [ ] keep\n### Blocked\n- none\n\n"
        "## Key Decisions\n- **A**: B\n\n"
        "## Next Steps\n1. Continue\n\n"
        "## Critical Context\n- data"
    )
    recompressed = (
        "## Goal\nBounded\n\n"
        "## Constraints & Preferences\n- concise\n\n"
        "## Progress\n### Done\n- [x] grouped\n### In Progress\n- [ ] next\n### Blocked\n- none\n\n"
        "## Key Decisions\n- **A**: B\n\n"
        "## Next Steps\n1. Continue\n\n"
        "## Critical Context\n- key refs"
    )
    provider.chat.side_effect = [
        LLMResponse(content=oversized_summary),
        LLMResponse(content=recompressed),
    ]

    summary = await generate_compaction_summary(
        messages=[{"role": "user", "content": "hello"}],
        provider=provider,
        model="test-model",
        previous_summary=None,
    )

    assert provider.chat.await_count == 2
    assert len(summary) <= MAX_SUMMARY_CHARS
    assert len(summary) <= TARGET_SUMMARY_CHARS
    assert "## Goal" in summary
    assert "## Progress" in summary
    assert "## Next Steps" in summary
