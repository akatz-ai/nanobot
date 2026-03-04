import json
from pathlib import Path

import pytest
import yaml

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.cron import CronTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _record_cron_inbox_event
from nanobot.cron.service import CronService
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.inbox import InboxEvent, SessionInbox


class _StubProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key=None, api_base=None)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        _ = (messages, tools, model, max_tokens, temperature)
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "stub-model"


def _event(summary: str, content: str = "payload", *, source: str = "cron") -> InboxEvent:
    return InboxEvent.create(
        source=source,
        summary=summary,
        content=content,
        source_meta={"k": "v"},
    )


def test_session_inbox_append_creates_jsonl(tmp_path: Path) -> None:
    session_path = tmp_path / "sessions" / "cli_test.jsonl"
    inbox = SessionInbox(session_path)

    event = _event("cron-1", "hello world")
    inbox.append(event)

    assert inbox.path.exists()
    lines = inbox.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_id"].startswith("evt_")
    assert payload["source"] == "cron"
    assert payload["summary"] == "cron-1"
    assert payload["content"] == "hello world"


def test_session_inbox_drain_returns_all_and_clears(tmp_path: Path) -> None:
    inbox = SessionInbox(tmp_path / "sessions" / "cli_test.jsonl")
    inbox.append(_event("one"))
    inbox.append(_event("two"))

    drained = inbox.drain()

    assert [evt.summary for evt in drained] == ["one", "two"]
    assert inbox.drain() == []
    assert not inbox.path.exists() or inbox.path.read_text(encoding="utf-8") == ""


def test_session_inbox_drain_empty_when_missing_file(tmp_path: Path) -> None:
    inbox = SessionInbox(tmp_path / "sessions" / "cli_test.jsonl")
    assert inbox.drain() == []


def test_session_inbox_clear_removes_sidecar(tmp_path: Path) -> None:
    inbox = SessionInbox(tmp_path / "sessions" / "cli_test.jsonl")
    inbox.append(_event("one"))
    assert inbox.path.exists()

    inbox.clear()

    assert not inbox.path.exists()


def test_multiple_appends_single_drain_preserves_order(tmp_path: Path) -> None:
    inbox = SessionInbox(tmp_path / "sessions" / "cli_test.jsonl")
    for idx in range(5):
        inbox.append(_event(f"item-{idx}"))

    drained = inbox.drain()

    assert [evt.summary for evt in drained] == [f"item-{idx}" for idx in range(5)]


def test_append_after_drain_rotate_lands_in_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inbox = SessionInbox(tmp_path / "sessions" / "cli_test.jsonl")
    inbox.append(_event("first"))
    second = _event("second")

    original_replace = Path.replace

    def _patched_replace(self: Path, target: Path) -> Path:
        out = original_replace(self, target)
        if self == inbox.path:
            inbox.append(second)
        return out

    monkeypatch.setattr(Path, "replace", _patched_replace)

    drained = inbox.drain()
    assert [evt.summary for evt in drained] == ["first"]

    pending = inbox.drain()
    assert [evt.summary for evt in pending] == ["second"]


@pytest.mark.asyncio
async def test_drain_integration_persists_assistant_history_entry(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
    )
    inbox = loop.sessions.get_inbox("cli:test")
    inbox.append(_event("cron summary", "cron response body"))

    await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test",
            content="hello",
        )
    )

    session = loop.sessions.get_or_create("cli:test")
    assert session.messages[0]["role"] == "assistant"
    assert "[External Events]" in session.messages[0]["content"]
    assert "cron summary" in session.messages[0]["content"]
    assert "cron response body" in session.messages[0]["content"]


@pytest.mark.asyncio
async def test_legacy_recent_cron_metadata_migrates_on_turn_start(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
    )
    session = loop.sessions.get_or_create("cli:legacy")
    session.metadata["recent_cron_actions"] = [
        {
            "job_id": "legacy-job",
            "message": "legacy message",
            "response_preview": "legacy response",
            "timestamp": "2026-03-01T00:00:00",
        }
    ]
    loop.sessions.save(session)

    await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="legacy",
            content="what changed?",
        )
    )

    updated = loop.sessions.get_or_create("cli:legacy")
    assert "recent_cron_actions" not in updated.metadata
    injected = [
        m
        for m in updated.messages
        if m.get("role") == "assistant" and "[External Events]" in str(m.get("content", ""))
    ]
    assert len(injected) == 1
    assert "[legacy_cron evt_legacy_legacy-job_0]" in injected[0]["content"]
    assert "legacy response" in injected[0]["content"]


@pytest.mark.asyncio
async def test_new_command_clears_inbox_sidecar(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path,
        model="stub-model",
    )
    inbox = loop.sessions.get_inbox("cli:test")
    inbox.append(_event("pending"))
    assert inbox.path.exists()

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test",
            content="/new",
        )
    )

    assert response is not None
    assert response.content == "New session started."
    assert not inbox.path.exists()


@pytest.mark.asyncio
async def test_origin_session_key_flows_through_cron_creation_and_callback(tmp_path: Path) -> None:
    cron_dir = tmp_path / "cron" / "jobs"
    service = CronService(cron_dir)
    tool = CronTool(cron_service=service, agent_id="agent-a")
    tool.set_context("discord", "room-123")

    created = await tool.execute(action="add", message="check status", every_seconds=60)
    assert "Created job" in created

    jobs = service.list_jobs(include_disabled=True, agent_id="agent-a")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.payload.origin_session_key == "discord:room-123"

    payload = yaml.safe_load((cron_dir / f"{job.id}.yaml").read_text(encoding="utf-8"))
    assert payload["origin_session_key"] == "discord:room-123"

    # Ensure callback routing prefers origin_session_key over fallback channel/to.
    job.payload.channel = "discord"
    job.payload.to = "other-room"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=_StubProvider(),
        workspace=tmp_path / "workspace",
        model="stub-model",
    )
    _record_cron_inbox_event(loop, job, "done")

    origin_events = loop.sessions.get_inbox("discord:room-123").drain()
    fallback_events = loop.sessions.get_inbox("discord:other-room").drain()
    assert len(origin_events) == 1
    assert origin_events[0].summary == f"Cron job {job.id}: {job.name}"
    assert origin_events[0].content == "done"
    assert fallback_events == []
