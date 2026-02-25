"""Multi-agent router: routes inbound messages to the correct AgentLoop by channel mapping."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.agent.workspace import get_agent_workspace, init_agent_workspace
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentProfile, ResolvedAgentProfile
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, Config
    from nanobot.cron.service import CronService
    from nanobot.providers.base import LLMProvider


@dataclass
class AgentInstance:
    """Wraps an AgentLoop with its own MessageBus, workspace, and metadata."""

    agent_id: str
    profile: ResolvedAgentProfile
    loop: AgentLoop
    bus: MessageBus
    workspace: Path
    _run_task: asyncio.Task | None = field(default=None, repr=False)
    _forward_task: asyncio.Task | None = field(default=None, repr=False)

    async def start(self, front_bus: MessageBus) -> None:
        """Start the agent loop and outbound forwarder."""
        self._run_task = asyncio.create_task(
            self.loop.run(), name=f"agent-{self.agent_id}-run"
        )
        self._forward_task = asyncio.create_task(
            self._forward_outbound(front_bus), name=f"agent-{self.agent_id}-fwd"
        )
        logger.info("Agent '{}' started (model={})", self.agent_id, self.profile.model)

    async def stop(self) -> None:
        """Stop the agent loop and forwarder."""
        self.loop.stop()
        await self.loop.close_mcp()
        if self._forward_task:
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass
        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent '{}' stopped", self.agent_id)

    async def _forward_outbound(self, front_bus: MessageBus) -> None:
        """Forward outbound messages from this agent's bus to the front-door bus."""
        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)
                await front_bus.publish_outbound(msg)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


class AgentRouter:
    """Routes inbound messages from a shared front-door bus to per-agent AgentLoop instances.

    Channel mapping: Discord channel_id -> agent_id. Non-Discord messages go to the
    default agent. If no agent profiles are configured, a single "default" agent is
    created for backward compatibility.
    """

    def __init__(
        self,
        front_bus: MessageBus,
        config: Config,
        provider: LLMProvider,
        cron_service: CronService | None = None,
    ):
        self.front_bus = front_bus
        self.config = config
        self.provider = provider
        self.cron_service = cron_service

        self._agents: dict[str, AgentInstance] = {}
        self._channel_map: dict[str, str] = {}  # discord_channel_id -> agent_id
        self._default_agent_id: str = "default"
        self._route_task: asyncio.Task | None = None

    @property
    def agents(self) -> dict[str, AgentInstance]:
        return self._agents

    @property
    def default_agent(self) -> AgentInstance | None:
        return self._agents.get(self._default_agent_id)

    def get_agent(self, agent_id: str) -> AgentInstance | None:
        return self._agents.get(agent_id)

    async def initialize_from_config(self) -> None:
        """Create agent instances from config profiles. Falls back to single-agent mode."""
        profiles = self.config.agents.profiles

        if not profiles:
            # Backward compat: single "default" agent using defaults
            await self._create_agent_instance("default", self._make_default_profile())
            return

        # Create agents from profiles
        first_id = None
        for agent_id, profile in profiles.items():
            resolved = profile.resolve(self.config.agents.defaults)
            await self._create_agent_instance(agent_id, resolved)
            if first_id is None:
                first_id = agent_id

        # The first profile is the default (or "general" if it exists)
        if "general" in self._agents:
            self._default_agent_id = "general"
        elif first_id:
            self._default_agent_id = first_id

    def _make_default_profile(self) -> ResolvedAgentProfile:
        """Create a resolved profile from the global defaults."""
        d = self.config.agents.defaults
        return ResolvedAgentProfile(
            model=d.model,
            max_tokens=d.max_tokens,
            temperature=d.temperature,
            max_tool_iterations=d.max_tool_iterations,
            memory_window=d.memory_window,
        )

    async def _create_agent_instance(
        self, agent_id: str, profile: ResolvedAgentProfile
    ) -> AgentInstance:
        """Create and register an AgentInstance (does NOT start it)."""
        base_workspace = self.config.workspace_path
        workspace = init_agent_workspace(
            base_workspace, agent_id, system_identity=profile.system_identity
        )

        agent_bus = MessageBus()
        session_manager = SessionManager(workspace)

        loop = AgentLoop(
            bus=agent_bus,
            provider=self.provider,
            workspace=workspace,
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            max_iterations=profile.max_tool_iterations,
            memory_window=profile.memory_window,
            brave_api_key=self.config.tools.web.search.api_key or None,
            exec_config=self.config.tools.exec,
            cron_service=self.cron_service,
            restrict_to_workspace=self.config.tools.restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=self.config.tools.mcp_servers,
            channels_config=self.config.channels,
            memory_graph_config=self.config.memory_graph,
            skill_names=profile.skills,
        )

        instance = AgentInstance(
            agent_id=agent_id,
            profile=profile,
            loop=loop,
            bus=agent_bus,
            workspace=workspace,
        )

        self._agents[agent_id] = instance

        # Register channel mappings
        for ch_id in profile.discord_channels:
            self._channel_map[ch_id] = agent_id

        return instance

    async def create_agent(
        self, agent_id: str, profile: ResolvedAgentProfile
    ) -> AgentInstance:
        """Hot-add a new agent at runtime: create, register, and start."""
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already exists")

        instance = await self._create_agent_instance(agent_id, profile)
        await instance.start(self.front_bus)
        return instance

    async def remove_agent(self, agent_id: str) -> None:
        """Stop and remove an agent."""
        if agent_id == self._default_agent_id:
            raise ValueError("Cannot remove the default agent")
        instance = self._agents.pop(agent_id, None)
        if instance:
            # Remove channel mappings
            self._channel_map = {
                k: v for k, v in self._channel_map.items() if v != agent_id
            }
            await instance.stop()

    def map_channel(self, channel_id: str, agent_id: str) -> None:
        """Map a Discord channel to an agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' not found")
        self._channel_map[channel_id] = agent_id

    def _resolve_agent(self, msg: InboundMessage) -> AgentInstance | None:
        """Resolve which agent should handle a message."""
        if msg.channel == "discord":
            agent_id = self._channel_map.get(msg.chat_id)
            if agent_id and agent_id in self._agents:
                return self._agents[agent_id]
        # Non-Discord or unmapped Discord channels -> default agent
        return self._agents.get(self._default_agent_id)

    async def start(self) -> None:
        """Start all agents and the inbound routing loop."""
        for instance in self._agents.values():
            await instance.start(self.front_bus)

        self._route_task = asyncio.create_task(
            self._route_inbound(), name="agent-router-inbound"
        )
        logger.info(
            "AgentRouter started: {} agents, {} channel mappings",
            len(self._agents),
            len(self._channel_map),
        )

    async def stop(self) -> None:
        """Stop all agents and the routing loop."""
        if self._route_task:
            self._route_task.cancel()
            try:
                await self._route_task
            except asyncio.CancelledError:
                pass

        for instance in self._agents.values():
            await instance.stop()
        logger.info("AgentRouter stopped")

    async def _route_inbound(self) -> None:
        """Main routing loop: consume from front-door bus and dispatch to agents."""
        while True:
            try:
                msg = await asyncio.wait_for(
                    self.front_bus.consume_inbound(), timeout=1.0
                )
                instance = self._resolve_agent(msg)
                if instance:
                    await instance.bus.publish_inbound(msg)
                else:
                    logger.warning(
                        "No agent found for message on {}:{}", msg.channel, msg.chat_id
                    )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
