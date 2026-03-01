import copy
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse


class _SequenceProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
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
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        _ = (tools, model, max_tokens, temperature, reasoning_effort)
        self.calls.append(copy.deepcopy(messages))
        if not self._responses:
            raise AssertionError("No scripted response left")
        return self._responses.pop(0)

    def get_default_model(self) -> str:
        return "stub-model"


@pytest.mark.asyncio
async def test_length_finish_reason_continues_and_concatenates(tmp_path: Path) -> None:
    provider = _SequenceProvider(
        [
            LLMResponse(content="Part 1", finish_reason="length"),
            LLMResponse(content="Part 2", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="stub-model",
    )

    final_content, _, messages = await loop._run_agent_loop(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "write"}]
    )

    assert final_content == "Part 1\nPart 2"
    assert len(provider.calls) == 2
    assert provider.calls[1][-1] == {"role": "user", "content": "Continue from where you left off."}
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "Part 2"


@pytest.mark.asyncio
async def test_length_finish_reason_caps_after_two_continuations(tmp_path: Path) -> None:
    provider = _SequenceProvider(
        [
            LLMResponse(content="Part 1", finish_reason="length"),
            LLMResponse(content="Part 2", finish_reason="length"),
            LLMResponse(content="Part 3", finish_reason="length"),
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="stub-model",
    )

    final_content, _, _ = await loop._run_agent_loop(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "write"}]
    )

    assert final_content == "Part 1\nPart 2\nPart 3"
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_stop_finish_reason_unchanged_without_continuations(tmp_path: Path) -> None:
    provider = _SequenceProvider([LLMResponse(content="Done", finish_reason="stop")])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="stub-model",
    )

    final_content, _, _ = await loop._run_agent_loop(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "write"}]
    )

    assert final_content == "Done"
    assert len(provider.calls) == 1
