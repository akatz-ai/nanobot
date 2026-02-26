from __future__ import annotations

from typing import Any

import pytest

from nanobot.providers.anthropic_direct_provider import (
    SYSTEM_PREFIX,
    AnthropicDirectProvider,
)


def _provider() -> AnthropicDirectProvider:
    return AnthropicDirectProvider(
        oauth_token="sk-ant-oat01-test-token",
        default_model="anthropic/claude-opus-4-6",
    )


def test_build_body_basic():
    provider = _provider()
    body = provider._build_body(
        messages=[
            {"role": "system", "content": "System rule"},
            {"role": "user", "content": "Hello"},
        ],
        tools=None,
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    assert body["model"] == "claude-opus-4-6"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["system"][0]["text"] == SYSTEM_PREFIX
    assert body["system"][-1]["text"] == "System rule"


def test_build_body_multi_turn():
    provider = _provider()
    body = provider._build_body(
        messages=[
            {"role": "system", "content": "System rule"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        tools=None,
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    assert body["messages"][0] == {"role": "user", "content": "Q1"}
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][2] == {"role": "user", "content": "Q2"}


def test_build_body_tool_calls():
    provider = _provider()
    body = provider._build_body(
        messages=[
            {"role": "user", "content": "Use a tool"},
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
        ],
        tools=None,
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    assistant_blocks = body["messages"][1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "Calling tool"}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert assistant_blocks[1]["id"] == "call_1"
    assert assistant_blocks[1]["name"] == "read_file"
    assert assistant_blocks[1]["input"] == {"path": "README.md"}


def test_build_body_tool_results():
    provider = _provider()
    body = provider._build_body(
        messages=[
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": "file content"},
        ],
        tools=None,
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "file content"}
            ],
        }
    ]


def test_build_body_images():
    provider = _provider()
    body = provider._build_body(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBB"}},
                ],
            }
        ],
        tools=None,
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    blocks = body["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Look"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAABBBB"},
    }


def test_build_body_tools_definition():
    provider = _provider()
    body = provider._build_body(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file content",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        model="claude-opus-4-6",
        max_tokens=256,
        temperature=0.3,
    )

    assert body["tools"] == [
        {
            "name": "read_file",
            "description": "Read file content",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_cache_control_on_system():
    provider = _provider()
    system_blocks, _ = provider._apply_cache_control(
        system_parts=[SYSTEM_PREFIX, "Dynamic system"],
        messages=[{"role": "user", "content": "question"}],
    )

    assert system_blocks[0]["text"] == SYSTEM_PREFIX
    assert "cache_control" not in system_blocks[0]
    assert system_blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_on_conversation_prefix():
    provider = _provider()
    _, messages = provider._apply_cache_control(
        system_parts=[SYSTEM_PREFIX],
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
            {"role": "user", "content": "latest"},
        ],
    )

    assert messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_parse_response_text():
    provider = _provider()
    response = provider._parse_response(
        {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
    )

    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.tool_calls == []


def test_parse_response_tool_use():
    provider = _provider()
    response = provider._parse_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "list_dir",
                    "input": {"path": "."},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 2},
        }
    )

    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].name == "list_dir"
    assert response.tool_calls[0].arguments == {"path": "."}


def test_parse_response_usage():
    provider = _provider()
    response = provider._parse_response(
        {
            "content": [{"type": "text", "text": "done"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 10,
            },
        }
    )

    assert response.usage["prompt_tokens"] == 100  # raw (non-cached) input tokens
    assert response.usage["completion_tokens"] == 20
    # total_tokens = total_input (100 + 80 + 10) + output (20) = 210
    assert response.usage["total_tokens"] == 210
    assert response.usage["cache_read_input_tokens"] == 80
    assert response.usage["cache_creation_input_tokens"] == 10


def test_model_prefix_stripping():
    provider = AnthropicDirectProvider(
        oauth_token="sk-ant-oat01-test-token",
        default_model="anthropic/claude-opus-4-6",
    )

    assert provider.get_default_model() == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_chat_401_handling(monkeypatch: pytest.MonkeyPatch):
    provider = _provider()

    class _FakeResponse:
        status_code = 401
        text = "unauthorized"

        def json(self) -> dict[str, Any]:
            return {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr("nanobot.providers.anthropic_direct_provider.httpx.AsyncClient", _FakeClient)

    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content is not None
    assert "claude login" in response.content
