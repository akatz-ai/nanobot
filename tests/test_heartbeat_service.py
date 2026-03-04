import asyncio

import pytest

from nanobot.heartbeat.service import HeartbeatAgent, HeartbeatService
from nanobot.providers.base import LLMResponse, ToolCallRequest


class DummyProvider:
    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses or [])

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="nothing to do", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_tick_skip_when_no_heartbeat_file(tmp_path) -> None:
    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        interval_s=9999,
        enabled=True,
    )
    await service._tick()


@pytest.mark.asyncio
async def test_tick_executes_on_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("Check the queue", encoding="utf-8")
    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1",
                        name="heartbeat",
                        arguments={"action": "run", "tasks": "Process the queue"},
                    )
                ],
            )
        ]
    )

    executed: list[str] = []

    async def on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "Done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        on_execute=on_execute,
        interval_s=9999,
        enabled=True,
    )
    await service._tick()
    assert executed == ["Process the queue"]


@pytest.mark.asyncio
async def test_decide_returns_skip_when_no_tool_call(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
    )

    action, tasks = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""


@pytest.mark.asyncio
async def test_trigger_now_executes_when_decision_is_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1",
                        name="heartbeat",
                        arguments={"action": "run", "tasks": "check open tasks"},
                    )
                ],
            )
        ]
    )

    called_with: list[str] = []

    async def on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        on_execute=on_execute,
    )

    result = await service.trigger_now()
    assert result == "done"
    assert called_with == ["check open tasks"]


@pytest.mark.asyncio
async def test_trigger_now_returns_none_when_decision_is_skip(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1",
                        name="heartbeat",
                        arguments={"action": "skip"},
                    )
                ],
            )
        ]
    )

    async def on_execute(tasks: str) -> str:
        return tasks

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        on_execute=on_execute,
    )

    assert await service.trigger_now() is None


# --- Per-agent heartbeat tests ---


@pytest.mark.asyncio
async def test_per_agent_tick_only_fires_agents_with_heartbeat_file(tmp_path) -> None:
    """Only agents with HEARTBEAT.md get Phase 1 decision calls."""
    agent_a_dir = tmp_path / "agent_a"
    agent_b_dir = tmp_path / "agent_b"
    agent_a_dir.mkdir()
    agent_b_dir.mkdir()
    # Only agent_a has a heartbeat file
    (agent_a_dir / "HEARTBEAT.md").write_text("Check deployments", encoding="utf-8")

    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1",
                        name="heartbeat",
                        arguments={"action": "run", "tasks": "Deploy check"},
                    )
                ],
            )
        ]
    )

    executed: list[tuple[str, str]] = []

    def _make_execute(agent_id: str):
        async def on_execute(tasks: str) -> str:
            executed.append((agent_id, tasks))
            return f"Done by {agent_id}"
        return on_execute

    async def _noop_notify(response: str) -> None:
        pass

    service = HeartbeatService(provider=provider, model="test-model")
    service.register_agent("agent_a", agent_a_dir, _make_execute("agent_a"), _noop_notify)
    service.register_agent("agent_b", agent_b_dir, _make_execute("agent_b"), _noop_notify)

    await service._tick()

    # Only agent_a should have executed
    assert executed == [("agent_a", "Deploy check")]


@pytest.mark.asyncio
async def test_per_agent_tick_multiple_agents_execute(tmp_path) -> None:
    """Multiple agents with HEARTBEAT.md all get checked."""
    agent_a_dir = tmp_path / "agent_a"
    agent_b_dir = tmp_path / "agent_b"
    agent_a_dir.mkdir()
    agent_b_dir.mkdir()
    (agent_a_dir / "HEARTBEAT.md").write_text("Task A", encoding="utf-8")
    (agent_b_dir / "HEARTBEAT.md").write_text("Task B", encoding="utf-8")

    # Two "run" decisions
    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="hb_1", name="heartbeat", arguments={"action": "run", "tasks": "A tasks"}),
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="hb_2", name="heartbeat", arguments={"action": "run", "tasks": "B tasks"}),
                ],
            ),
        ]
    )

    executed: list[tuple[str, str]] = []

    def _make_execute(agent_id: str):
        async def on_execute(tasks: str) -> str:
            executed.append((agent_id, tasks))
            return f"Done by {agent_id}"
        return on_execute

    notified: list[tuple[str, str]] = []

    def _make_notify(agent_id: str):
        async def on_notify(response: str) -> None:
            notified.append((agent_id, response))
        return on_notify

    service = HeartbeatService(provider=provider, model="test-model")
    service.register_agent("agent_a", agent_a_dir, _make_execute("agent_a"), _make_notify("agent_a"))
    service.register_agent("agent_b", agent_b_dir, _make_execute("agent_b"), _make_notify("agent_b"))

    await service._tick()

    assert ("agent_a", "A tasks") in executed
    assert ("agent_b", "B tasks") in executed
    assert ("agent_a", "Done by agent_a") in notified
    assert ("agent_b", "Done by agent_b") in notified


@pytest.mark.asyncio
async def test_per_agent_one_failure_doesnt_block_others(tmp_path) -> None:
    """If one agent's execute callback fails, others still run."""
    agent_a_dir = tmp_path / "agent_a"
    agent_b_dir = tmp_path / "agent_b"
    agent_a_dir.mkdir()
    agent_b_dir.mkdir()
    (agent_a_dir / "HEARTBEAT.md").write_text("Task A", encoding="utf-8")
    (agent_b_dir / "HEARTBEAT.md").write_text("Task B", encoding="utf-8")

    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="hb_1", name="heartbeat", arguments={"action": "run", "tasks": "A tasks"}),
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="hb_2", name="heartbeat", arguments={"action": "run", "tasks": "B tasks"}),
                ],
            ),
        ]
    )

    executed_b: list[str] = []

    async def _fail_execute(tasks: str) -> str:
        raise RuntimeError("agent_a exploded")

    async def _ok_execute(tasks: str) -> str:
        executed_b.append(tasks)
        return "B done"

    async def _noop_notify(response: str) -> None:
        pass

    service = HeartbeatService(provider=provider, model="test-model")
    service.register_agent("agent_a", agent_a_dir, _fail_execute, _noop_notify)
    service.register_agent("agent_b", agent_b_dir, _ok_execute, _noop_notify)

    await service._tick()

    # agent_b should still execute despite agent_a's failure
    assert executed_b == ["B tasks"]


@pytest.mark.asyncio
async def test_trigger_now_specific_agent(tmp_path) -> None:
    """trigger_now(agent_id) only triggers the specified agent."""
    agent_a_dir = tmp_path / "agent_a"
    agent_b_dir = tmp_path / "agent_b"
    agent_a_dir.mkdir()
    agent_b_dir.mkdir()
    (agent_a_dir / "HEARTBEAT.md").write_text("Task A", encoding="utf-8")
    (agent_b_dir / "HEARTBEAT.md").write_text("Task B", encoding="utf-8")

    provider = DummyProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="hb_1", name="heartbeat", arguments={"action": "run", "tasks": "B only"}),
                ],
            ),
        ]
    )

    executed: list[tuple[str, str]] = []

    def _make_execute(agent_id: str):
        async def on_execute(tasks: str) -> str:
            executed.append((agent_id, tasks))
            return f"Done by {agent_id}"
        return on_execute

    async def _noop_notify(response: str) -> None:
        pass

    service = HeartbeatService(provider=provider, model="test-model")
    service.register_agent("agent_a", agent_a_dir, _make_execute("agent_a"), _noop_notify)
    service.register_agent("agent_b", agent_b_dir, _make_execute("agent_b"), _noop_notify)

    result = await service.trigger_now(agent_id="agent_b")
    assert result == "Done by agent_b"
    assert executed == [("agent_b", "B only")]


@pytest.mark.asyncio
async def test_trigger_now_nonexistent_agent(tmp_path) -> None:
    """trigger_now with unknown agent_id returns None."""
    service = HeartbeatService(provider=DummyProvider(), model="test-model")
    assert await service.trigger_now(agent_id="ghost") is None


@pytest.mark.asyncio
async def test_start_no_agents_registered() -> None:
    """Start with no agents registered is a no-op (no task created)."""
    service = HeartbeatService(provider=DummyProvider(), model="test-model", enabled=True)
    await service.start()
    assert service._task is None
