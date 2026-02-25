import json
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse | Exception]):
        super().__init__(api_key=None, api_base=None)
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = (tools, model, max_tokens, temperature)
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("No scripted response left")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def get_default_model(self) -> str:
        return "stub-model"


def _checkpoint_mid_tool(session: Session) -> None:
    session.checkpoint(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})},
                    }
                ],
            },
        ]
    )


def _checkpoint_mid_loop(session: Session) -> None:
    session.checkpoint(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "name": "list_dir", "content": "[]"},
        ]
    )


def test_detect_resume_state_variants() -> None:
    clean = Session(key="cli:clean")
    assert clean.detect_resume_state() == "clean"

    clean.checkpoint([
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ])
    assert clean.detect_resume_state() == "clean"

    mid_tool = Session(key="cli:mid_tool")
    _checkpoint_mid_tool(mid_tool)
    assert mid_tool.detect_resume_state() == "mid_tool"

    mid_loop = Session(key="cli:mid_loop")
    _checkpoint_mid_loop(mid_loop)
    assert mid_loop.detect_resume_state() == "mid_loop"


@pytest.mark.asyncio
async def test_resume_notice_is_injected_for_mid_flight_state(tmp_path: Path) -> None:
    provider = _ScriptedProvider([LLMResponse(content="done", tool_calls=[])])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="stub-model",
    )

    session = loop.sessions.get_or_create("cli:test")
    _checkpoint_mid_loop(session)

    msg = InboundMessage(channel="cli", sender_id="u", chat_id="test", content="continue")
    response = await loop._process_message(msg)

    assert response is not None
    assert response.content == "done"
    assert provider.calls, "Provider should be called"
    first_call = provider.calls[0]
    assert any(
        m.get("role") == "system"
        and "interrupted mid-turn by a system restart" in m.get("content", "")
        for m in first_call
    )


@pytest.mark.asyncio
async def test_end_to_end_interruption_reload_and_auto_resume(tmp_path: Path) -> None:
    provider1 = _ScriptedProvider(
        [
            LLMResponse(
                content="I will inspect the workspace",
                tool_calls=[
                    ToolCallRequest(id="tc_1", name="list_dir", arguments={"path": "."}),
                ],
            ),
            RuntimeError("simulated crash"),
        ]
    )
    loop1 = AgentLoop(
        bus=MessageBus(),
        provider=provider1,
        workspace=tmp_path,
        model="stub-model",
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        await loop1._process_message(
            InboundMessage(channel="cli", sender_id="u", chat_id="test", content="check files")
        )

    interrupted = loop1.sessions.get_or_create("cli:test")
    assert interrupted.detect_resume_state() == "mid_loop"

    provider2 = _ScriptedProvider([LLMResponse(content="resumed answer", tool_calls=[])])
    loop2 = AgentLoop(
        bus=MessageBus(),
        provider=provider2,
        workspace=tmp_path,
        model="stub-model",
    )

    resumed = await loop2.resume_inflight_sessions()
    assert resumed == 1
    assert loop2.bus.outbound_size == 1

    outbound = await loop2.bus.consume_outbound()
    assert outbound.content == "resumed answer"

    resumed_session = loop2.sessions.get_or_create("cli:test")
    assert resumed_session.detect_resume_state() == "clean"

    assert provider2.calls, "Provider should be invoked during auto-resume"
    auto_resume_messages = provider2.calls[0]
    assert any(
        m.get("role") == "system"
        and "interrupted mid-turn by a system restart" in m.get("content", "")
        for m in auto_resume_messages
    )
