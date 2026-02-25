import asyncio

import pytest

from nanobot.heartbeat.service import HeartbeatService


class _FakeProvider:
    """Minimal provider stub for heartbeat tests."""

    def __init__(self, action: str = "skip", tasks: str = ""):
        self._action = action
        self._tasks = tasks

    async def chat(self, **kwargs):
        from nanobot.providers.base import LLMResponse, ToolCallRequest
        if self._action == "skip":
            return LLMResponse(content="nothing to do")
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="tc_1",
                    name="heartbeat",
                    arguments={"action": self._action, "tasks": self._tasks},
                )
            ],
        )

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    provider = _FakeProvider()
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
    """Tick should be a no-op when HEARTBEAT.md doesn't exist."""
    provider = _FakeProvider()
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        interval_s=9999,
        enabled=True,
    )
    # No HEARTBEAT.md — tick should not call the provider
    await service._tick()


@pytest.mark.asyncio
async def test_tick_executes_on_run(tmp_path) -> None:
    """When the LLM decides 'run', on_execute should be called."""
    (tmp_path / "HEARTBEAT.md").write_text("Check the queue")
    provider = _FakeProvider(action="run", tasks="Process the queue")

    executed = []

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
