import asyncio
import json
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
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


@pytest.mark.asyncio
async def test_auto_resume_cron_session_uses_origin_routing(tmp_path: Path) -> None:
    setup_loop = AgentLoop(
        bus=MessageBus(),
        provider=_ScriptedProvider([]),
        workspace=tmp_path,
        model="stub-model",
    )
    session = setup_loop.sessions.get_or_create("cron:job-123")
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "thread-42"
    _checkpoint_mid_loop(session)
    setup_loop.sessions.save(session)

    provider = _ScriptedProvider([LLMResponse(content="resumed cron", tool_calls=[])])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="stub-model",
    )

    resumed = await loop.resume_inflight_sessions()

    assert resumed == 1
    outbound = await loop.bus.consume_outbound()
    assert outbound.channel == "discord"
    assert outbound.chat_id == "thread-42"
    assert outbound.content == "resumed cron"


async def _shutdown_run(loop: AgentLoop, run_task: asyncio.Task) -> None:
    loop.stop()
    run_task.cancel()
    with suppress(asyncio.CancelledError):
        await run_task


@pytest.mark.asyncio
async def test_run_starts_loop_immediately_with_pending_resume(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ScriptedProvider([]),
        workspace=tmp_path,
        model="stub-model",
    )
    session = loop.sessions.get_or_create("cron:job-123")
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "thread-42"
    _checkpoint_mid_loop(session)
    loop.sessions.save(session)

    resume_started = asyncio.Event()

    async def _blocking_resume(_session: Session, _channel: str, _chat_id: str) -> OutboundMessage:
        resume_started.set()
        await asyncio.Event().wait()
        return OutboundMessage(channel="discord", chat_id="thread-42", content="never")

    loop._resume_session = _blocking_resume  # type: ignore[method-assign]
    loop._connect_mcp = AsyncMock()

    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(resume_started.wait(), timeout=1.0)
        await loop.bus.publish_inbound(
            InboundMessage(channel="discord", sender_id="u", chat_id="thread-42", content="/stop")
        )
        out = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=1.0)
        assert "stopped" in out.content.lower()
    finally:
        await _shutdown_run(loop, run_task)


@pytest.mark.asyncio
async def test_run_schedules_cron_resumes_as_background_tasks(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ScriptedProvider([]),
        workspace=tmp_path,
        model="stub-model",
    )
    session = loop.sessions.get_or_create("cron:job-123")
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "thread-42"
    _checkpoint_mid_loop(session)
    loop.sessions.save(session)

    resume_started = asyncio.Event()
    allow_finish = asyncio.Event()

    async def _blocking_resume(_session: Session, channel: str, chat_id: str) -> OutboundMessage:
        resume_started.set()
        await allow_finish.wait()
        return OutboundMessage(channel=channel, chat_id=chat_id, content="resumed cron")

    loop._resume_session = _blocking_resume  # type: ignore[method-assign]
    loop._connect_mcp = AsyncMock()

    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(resume_started.wait(), timeout=1.0)
        assert "cron:job-123" in loop._active_tasks
        assert "discord:thread-42" in loop._active_tasks
        assert any(not t.done() for t in loop._active_tasks["cron:job-123"])

        allow_finish.set()
        outbound = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=1.0)
        assert outbound.channel == "discord"
        assert outbound.chat_id == "thread-42"
        assert outbound.content == "resumed cron"
    finally:
        await _shutdown_run(loop, run_task)


@pytest.mark.asyncio
async def test_stop_cancels_running_resume_task(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ScriptedProvider([]),
        workspace=tmp_path,
        model="stub-model",
    )
    session = loop.sessions.get_or_create("cron:job-123")
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "thread-42"
    _checkpoint_mid_loop(session)
    loop.sessions.save(session)

    resume_started = asyncio.Event()
    resume_cancelled = asyncio.Event()

    async def _blocking_resume(_session: Session, _channel: str, _chat_id: str) -> OutboundMessage:
        resume_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            resume_cancelled.set()
            raise
        return OutboundMessage(channel="discord", chat_id="thread-42", content="unexpected")

    loop._resume_session = _blocking_resume  # type: ignore[method-assign]
    loop._connect_mcp = AsyncMock()

    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(resume_started.wait(), timeout=1.0)
        await loop.bus.publish_inbound(
            InboundMessage(channel="discord", sender_id="u", chat_id="thread-42", content="/stop")
        )
        out = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=1.0)
        assert "stopped" in out.content.lower()
        assert resume_cancelled.is_set()
    finally:
        await _shutdown_run(loop, run_task)


@pytest.mark.asyncio
async def test_normal_messages_process_while_resume_is_running(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_ScriptedProvider([]),
        workspace=tmp_path,
        model="stub-model",
    )
    session = loop.sessions.get_or_create("cron:job-123")
    session.metadata["origin_channel"] = "discord"
    session.metadata["origin_chat_id"] = "thread-42"
    _checkpoint_mid_loop(session)
    loop.sessions.save(session)

    resume_started = asyncio.Event()

    async def _blocking_resume(_session: Session, _channel: str, _chat_id: str) -> OutboundMessage:
        resume_started.set()
        await asyncio.Event().wait()
        return OutboundMessage(channel="discord", chat_id="thread-42", content="never")

    async def _mock_process(msg: InboundMessage, **_kwargs: object) -> OutboundMessage:
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=f"echo:{msg.content}")

    loop._resume_session = _blocking_resume  # type: ignore[method-assign]
    loop._process_message = AsyncMock(side_effect=_mock_process)
    loop._connect_mcp = AsyncMock()

    run_task = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(resume_started.wait(), timeout=1.0)
        await loop.bus.publish_inbound(
            InboundMessage(channel="discord", sender_id="u", chat_id="room-2", content="hello")
        )
        outbound = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=1.0)
        assert outbound.content == "echo:hello"

        await loop.bus.publish_inbound(
            InboundMessage(channel="discord", sender_id="u", chat_id="thread-42", content="/stop")
        )
        stop_out = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=1.0)
        assert "stopped" in stop_out.content.lower()
    finally:
        await _shutdown_run(loop, run_task)
