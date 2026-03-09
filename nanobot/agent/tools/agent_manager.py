"""Tool for managing agents dynamically (create, clone, list, info, remove)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
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
            "Discord channel, webhook, model, and identity), clone (clone an existing "
            "agent into a new one with copied memory/history/skills), list (list all agents), "
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
                    "enum": ["create", "clone", "list", "info", "remove"],
                    "description": "The action to perform.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier (required for create, clone, info, remove). Use lowercase, no spaces.",
                },
                "source_agent_id": {
                    "type": "string",
                    "description": "Source agent identifier for clone action.",
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
                "copy_history": {
                    "type": "boolean",
                    "description": "For clone action: copy daily history and session history from the source agent.",
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
        elif action == "clone":
            if not agent_id:
                return "Error: agent_id is required for clone action."
            source_agent_id = kwargs.get("source_agent_id", "")
            if not source_agent_id:
                return "Error: source_agent_id is required for clone action."
            return await self._action_clone(
                source_agent_id=source_agent_id,
                agent_id=agent_id,
                model=kwargs.get("model"),
                system_identity=kwargs.get("system_identity"),
                channel_name=kwargs.get("channel_name"),
                skills=kwargs.get("skills"),
                display_name=kwargs.get("display_name"),
                avatar_url=kwargs.get("avatar_url"),
                copy_history=bool(kwargs.get("copy_history", False)),
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
            "skills": p.skills,
            "system_identity": p.system_identity[:200] if p.system_identity else None,
            "discord_channels": p.discord_channels,
            "display_name": p.display_name,
            "avatar_url": p.avatar_url,
            "webhook_enabled": p.discord_webhook_url is not None,
            "workspace": str(instance.workspace),
        }
        return json.dumps(info, indent=2)

    async def _provision_discord_channel(
        self,
        *,
        agent_id: str,
        channel_name: str | None,
        display_name: str | None,
        avatar_url: str | None,
        topic_model: str,
    ) -> tuple[str | None, str | None, str]:
        """Create the Discord channel + webhook for a new agent."""
        discord = self._channel_manager.get_discord_channel()
        if not discord:
            raise ValueError("Discord channel is not enabled. Cannot create agent channel.")
        if not self._guild_id:
            raise ValueError("Discord guild_id is not configured. Cannot create agent channel.")

        ch_name = channel_name or agent_id
        discord_channel_id = await discord.create_guild_channel(
            guild_id=self._guild_id,
            name=ch_name,
            topic=(topic_model or "")[:1024],
        )
        if not discord_channel_id:
            raise ValueError(f"Failed to create Discord channel '{ch_name}'.")

        webhook_display_name = display_name or agent_id
        webhook_url = await discord.create_channel_webhook(
            channel_id=discord_channel_id,
            name=webhook_display_name,
            avatar_url=avatar_url,
        )
        if webhook_url:
            discord.register_webhook(
                channel_id=discord_channel_id,
                webhook_url=webhook_url,
                display_name=webhook_display_name,
                avatar_url=avatar_url,
            )
        else:
            logger.warning("Failed to create webhook for agent '{}', falling back to bot messages", agent_id)

        return discord_channel_id, webhook_url, ch_name

    def _copy_agent_workspace(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        copy_history: bool,
    ) -> None:
        """Copy key workspace artifacts from one agent to another."""
        source_workspace = self._router.get_agent(source_agent_id).workspace
        target_workspace = self._router.get_agent(target_agent_id).workspace

        for rel in [Path('memory/MEMORY.md')]:
            src = source_workspace / rel
            dst = target_workspace / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        source_skills = source_workspace / 'skills'
        target_skills = target_workspace / 'skills'
        if source_skills.exists():
            for item in source_skills.iterdir():
                dst = target_skills / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    target_skills.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst)

        if copy_history:
            source_history = source_workspace / 'memory' / 'history'
            target_history = target_workspace / 'memory' / 'history'
            if source_history.exists():
                for item in source_history.iterdir():
                    dst = target_history / item.name
                    if item.is_dir():
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(item, dst)
                    else:
                        target_history.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dst)

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
        if self._router.get_agent(agent_id):
            return f"Error: Agent '{agent_id}' already exists."

        resolved_model = model or self._router.config.agents.defaults.model
        try:
            discord_channel_id, webhook_url, final_channel_name = await self._provision_discord_channel(
                agent_id=agent_id,
                channel_name=channel_name,
                display_name=display_name,
                avatar_url=avatar_url,
                topic_model=resolved_model,
            )
            profile = self._profile_manager.create_profile(
                agent_id,
                model=model,
                system_identity=system_identity,
                skills=skills,
                discord_channels=[discord_channel_id] if discord_channel_id else [],
                display_name=display_name or agent_id,
                avatar_url=avatar_url,
                discord_webhook_url=webhook_url,
            )
        except ValueError as e:
            return f"Error: {e}"

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
            "channel_name": final_channel_name,
            "display_name": resolved.display_name,
            "webhook_enabled": webhook_url is not None,
        }
        logger.info("Agent '{}' created via manage_agents tool", agent_id)
        return json.dumps(result, indent=2)

    async def _action_clone(
        self,
        *,
        source_agent_id: str,
        agent_id: str,
        model: str | None = None,
        system_identity: str | None = None,
        channel_name: str | None = None,
        skills: list[str] | None = None,
        display_name: str | None = None,
        avatar_url: str | None = None,
        copy_history: bool = False,
    ) -> str:
        if self._router.get_agent(agent_id):
            return f"Error: Agent '{agent_id}' already exists."
        source = self._router.get_agent(source_agent_id)
        if not source:
            return f"Error: Source agent '{source_agent_id}' not found."

        source_profile = self._profile_manager.get_profile(source_agent_id)
        if not source_profile:
            return f"Error: Source profile '{source_agent_id}' not found."

        try:
            discord_channel_id, webhook_url, final_channel_name = await self._provision_discord_channel(
                agent_id=agent_id,
                channel_name=channel_name,
                display_name=display_name or source.profile.display_name or agent_id,
                avatar_url=avatar_url if avatar_url is not None else source.profile.avatar_url,
                topic_model=model or source.profile.model or self._router.config.agents.defaults.model,
            )
            profile = self._profile_manager.create_profile(
                agent_id,
                model=model if model is not None else source_profile.model,
                background_model=source_profile.background_model,
                context_window=source_profile.context_window,
                background_context_window=source_profile.background_context_window,
                session_store=source_profile.session_store,
                max_tokens=source_profile.max_tokens,
                temperature=source_profile.temperature,
                max_tool_iterations=source_profile.max_tool_iterations,
                reasoning_effort=source_profile.reasoning_effort,
                skills=skills if skills is not None else source_profile.skills,
                system_identity=system_identity if system_identity is not None else source_profile.system_identity,
                discord_channels=[discord_channel_id] if discord_channel_id else [],
                display_name=display_name or source.profile.display_name or agent_id,
                avatar_url=avatar_url if avatar_url is not None else source.profile.avatar_url,
                discord_webhook_url=webhook_url,
            )
        except ValueError as e:
            return f"Error: {e}"

        resolved = profile.resolve(self._router.config.agents.defaults)
        try:
            await self._router.create_agent(agent_id, resolved)
            self._copy_agent_workspace(
                source_agent_id=source_agent_id,
                target_agent_id=agent_id,
                copy_history=copy_history,
            )
        except ValueError as e:
            return f"Error starting cloned agent: {e}"

        result = {
            "status": "cloned",
            "agent_id": agent_id,
            "source_agent_id": source_agent_id,
            "model": resolved.model,
            "discord_channel_id": discord_channel_id,
            "channel_name": final_channel_name,
            "display_name": resolved.display_name,
            "webhook_enabled": webhook_url is not None,
            "copied_history": copy_history,
        }
        logger.info("Agent '{}' cloned from '{}' via manage_agents tool", agent_id, source_agent_id)
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
