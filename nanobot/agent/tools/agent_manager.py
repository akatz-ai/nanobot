"""Tool for managing agents dynamically (create, list, info, remove)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.profile_manager import AgentProfileManager
    from nanobot.agent.router import AgentRouter
    from nanobot.channels.manager import ChannelManager


class AgentManagerTool(Tool):
    """Tool for the general/default agent to manage other agents."""

    def __init__(
        self,
        router: AgentRouter,
        profile_manager: AgentProfileManager,
        channel_manager: ChannelManager,
        guild_id: str,
    ):
        self._router = router
        self._profile_manager = profile_manager
        self._channel_manager = channel_manager
        self._guild_id = guild_id

    @property
    def name(self) -> str:
        return "manage_agents"

    @property
    def description(self) -> str:
        return (
            "Manage agent profiles. Actions: create (create a new agent with its own "
            "Discord channel, model, and identity), list (list all agents), "
            "info (get details about an agent), remove (stop and remove an agent)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "info", "remove"],
                    "description": "The action to perform.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier (required for create, info, remove). Use lowercase, no spaces.",
                },
                "model": {
                    "type": "string",
                    "description": "LLM model for the agent (e.g. 'anthropic-direct/claude-opus-4-6'). Falls back to defaults.",
                },
                "system_identity": {
                    "type": "string",
                    "description": "Custom system identity/personality text for the new agent.",
                },
                "channel_name": {
                    "type": "string",
                    "description": "Discord channel name to create (defaults to agent_id).",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skill whitelist for the agent (omit for all skills).",
                },
                "display_name": {
                    "type": "string",
                    "description": "Display name for the agent in Discord (shown via webhook).",
                },
                "avatar_url": {
                    "type": "string",
                    "description": "Avatar image URL for the agent in Discord.",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        agent_id = kwargs.get("agent_id", "")

        if action == "list":
            return await self._action_list()
        elif action == "info":
            if not agent_id:
                return "Error: agent_id is required for info action."
            return await self._action_info(agent_id)
        elif action == "create":
            if not agent_id:
                return "Error: agent_id is required for create action."
            return await self._action_create(
                agent_id=agent_id,
                model=kwargs.get("model"),
                system_identity=kwargs.get("system_identity"),
                channel_name=kwargs.get("channel_name"),
                skills=kwargs.get("skills"),
                display_name=kwargs.get("display_name"),
                avatar_url=kwargs.get("avatar_url"),
            )
        elif action == "remove":
            if not agent_id:
                return "Error: agent_id is required for remove action."
            return await self._action_remove(agent_id)
        else:
            return f"Error: Unknown action '{action}'. Use: create, list, info, remove."

    async def _action_list(self) -> str:
        agents = []
        for aid, instance in self._router.agents.items():
            p = instance.profile
            agents.append({
                "agent_id": aid,
                "model": p.model,
                "discord_channels": p.discord_channels,
                "is_default": aid == self._router._default_agent_id,
            })
        return json.dumps({"agents": agents}, indent=2)

    async def _action_info(self, agent_id: str) -> str:
        instance = self._router.get_agent(agent_id)
        if not instance:
            return f"Error: Agent '{agent_id}' not found."
        p = instance.profile
        info = {
            "agent_id": agent_id,
            "model": p.model,
            "max_tokens": p.max_tokens,
            "temperature": p.temperature,
            "max_tool_iterations": p.max_tool_iterations,
            "memory_window": p.memory_window,
            "skills": p.skills,
            "system_identity": p.system_identity[:200] if p.system_identity else None,
            "discord_channels": p.discord_channels,
            "display_name": p.display_name,
            "avatar_url": p.avatar_url,
            "webhook_enabled": p.discord_webhook_url is not None,
            "workspace": str(instance.workspace),
        }
        return json.dumps(info, indent=2)

    async def _action_create(
        self,
        agent_id: str,
        model: str | None = None,
        system_identity: str | None = None,
        channel_name: str | None = None,
        skills: list[str] | None = None,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> str:
        # Validate
        if self._router.get_agent(agent_id):
            return f"Error: Agent '{agent_id}' already exists."

        # Create Discord channel
        discord_channel_id = None
        webhook_url = None
        discord = self._channel_manager.get_discord_channel()
        if discord and self._guild_id:
            ch_name = channel_name or agent_id
            discord_channel_id = await discord.create_guild_channel(
                guild_id=self._guild_id,
                name=ch_name,
                topic=f"Agent: {agent_id}" + (f" | {model}" if model else ""),
            )
            if not discord_channel_id:
                return f"Error: Failed to create Discord channel '{ch_name}'."

            # Create webhook for custom name/avatar if display_name or avatar_url provided
            if display_name or avatar_url:
                webhook_url = await discord.create_channel_webhook(
                    channel_id=discord_channel_id,
                    name=display_name or agent_id,
                    avatar_url=avatar_url,
                )
                if webhook_url:
                    discord.register_webhook(
                        channel_id=discord_channel_id,
                        webhook_url=webhook_url,
                        display_name=display_name,
                        avatar_url=avatar_url,
                    )
                else:
                    logger.warning("Failed to create webhook for agent '{}', falling back to bot messages", agent_id)
        elif not discord:
            return "Error: Discord channel is not enabled. Cannot create agent channel."

        discord_channels = [discord_channel_id] if discord_channel_id else []

        # Create profile in config
        try:
            profile = self._profile_manager.create_profile(
                agent_id,
                model=model,
                system_identity=system_identity,
                skills=skills,
                discord_channels=discord_channels,
                display_name=display_name,
                avatar_url=avatar_url,
                discord_webhook_url=webhook_url,
            )
        except ValueError as e:
            return f"Error: {e}"

        # Resolve and start the agent
        resolved = profile.resolve(self._router.config.agents.defaults)
        try:
            await self._router.create_agent(agent_id, resolved)
        except ValueError as e:
            return f"Error starting agent: {e}"

        result = {
            "status": "created",
            "agent_id": agent_id,
            "model": resolved.model,
            "discord_channel_id": discord_channel_id,
            "channel_name": channel_name or agent_id,
            "display_name": display_name,
            "webhook_enabled": webhook_url is not None,
        }
        logger.info("Agent '{}' created via manage_agents tool", agent_id)
        return json.dumps(result, indent=2)

    async def _action_remove(self, agent_id: str) -> str:
        if agent_id == self._router._default_agent_id:
            return "Error: Cannot remove the default agent."
        if not self._router.get_agent(agent_id):
            return f"Error: Agent '{agent_id}' not found."

        try:
            await self._router.remove_agent(agent_id)
            self._profile_manager.delete_profile(agent_id)
        except ValueError as e:
            return f"Error: {e}"

        return json.dumps({"status": "removed", "agent_id": agent_id})
