"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks (required for run)",
                    },
                },
                "required": ["action"],
            },
        },
    }
]


@dataclass
class HeartbeatAgent:
    """An agent registered for heartbeat checks."""

    agent_id: str
    workspace: Path
    on_execute: Callable[[str], Coroutine[Any, Any, str]]
    on_notify: Callable[[str], Coroutine[Any, Any, None]]

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None


class HeartbeatService:
    """
    Periodic heartbeat service that wakes agents to check for tasks.

    A single timer iterates over all registered agents on each tick.
    Each agent has its own HEARTBEAT.md, execution callback, and
    notification callback.

    Phase 1 (decision): reads the agent's HEARTBEAT.md and asks the LLM
    — via a virtual tool call — whether there are active tasks.

    Phase 2 (execution): only triggered when Phase 1 returns ``run``.
    The agent's ``on_execute`` callback runs the task through its full
    agent loop and returns the result to deliver via ``on_notify``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        # Legacy single-agent interface (deprecated, use register_agent)
        workspace: Path | None = None,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.interval_s = interval_s
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None
        self._agents: dict[str, HeartbeatAgent] = {}

        # Support legacy single-agent construction
        if workspace is not None and on_execute is not None:
            self._agents["_legacy"] = HeartbeatAgent(
                agent_id="_legacy",
                workspace=workspace,
                on_execute=on_execute,
                on_notify=on_notify or (lambda _: asyncio.sleep(0)),
            )

    def register_agent(
        self,
        agent_id: str,
        workspace: Path,
        on_execute: Callable[[str], Coroutine[Any, Any, str]],
        on_notify: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        """Register an agent for heartbeat checks."""
        self._agents[agent_id] = HeartbeatAgent(
            agent_id=agent_id,
            workspace=workspace,
            on_execute=on_execute,
            on_notify=on_notify,
        )
        logger.debug("Heartbeat: registered agent '{}'", agent_id)

    async def _decide(self, content: str) -> tuple[str, str]:
        """Phase 1: ask LLM to decide skip/run via virtual tool call.

        Returns (action, tasks) where action is 'skip' or 'run'.
        """
        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a heartbeat agent. Call the heartbeat tool to report your decision."},
                {"role": "user", "content": (
                    "Review the following HEARTBEAT.md and decide whether there are active tasks.\n\n"
                    f"{content}"
                )},
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.model,
        )

        if not response.has_tool_calls:
            return "skip", ""

        args = response.tool_calls[0].arguments
        return args.get("action", "skip"), args.get("tasks", "")

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return
        if not self._agents:
            logger.info("Heartbeat: no agents registered, skipping")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        agent_ids = ", ".join(sorted(self._agents.keys()))
        logger.info("Heartbeat started (every {}s, agents: {})", self.interval_s, agent_ids)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    async def _tick(self) -> None:
        """Execute a single heartbeat tick across all registered agents."""
        for agent_id, agent in self._agents.items():
            try:
                await self._tick_agent(agent)
            except Exception:
                logger.exception("Heartbeat failed for agent '{}'", agent_id)

    async def _tick_agent(self, agent: HeartbeatAgent) -> None:
        """Execute a heartbeat tick for a single agent."""
        content = agent.read_heartbeat_file()
        if not content:
            logger.debug("Heartbeat: {}: HEARTBEAT.md missing or empty", agent.agent_id)
            return

        logger.info("Heartbeat: {}: checking for tasks...", agent.agent_id)

        action, tasks = await self._decide(content)

        if action != "run":
            logger.info("Heartbeat: {}: OK (nothing to report)", agent.agent_id)
            return

        logger.info("Heartbeat: {}: tasks found, executing...", agent.agent_id)
        response = await agent.on_execute(tasks)
        if response:
            logger.info("Heartbeat: {}: completed, delivering response", agent.agent_id)
            await agent.on_notify(response)

    async def trigger_now(self, agent_id: str | None = None) -> str | None:
        """Manually trigger a heartbeat for one or all agents.

        If *agent_id* is given, only that agent is triggered.
        Otherwise all agents are triggered and the first non-empty
        response is returned.
        """
        if agent_id is not None:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
            return await self._trigger_agent(agent)

        for agent in self._agents.values():
            result = await self._trigger_agent(agent)
            if result:
                return result
        return None

    async def _trigger_agent(self, agent: HeartbeatAgent) -> str | None:
        """Manually trigger a heartbeat for a single agent."""
        content = agent.read_heartbeat_file()
        if not content:
            return None
        action, tasks = await self._decide(content)
        if action != "run":
            return None
        return await agent.on_execute(tasks)
