"""Profile manager: CRUD operations on agent profiles with config persistence."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.config.schema import AgentProfile, Config
from nanobot.config.state import StateStore


class AgentProfileManager:
    """Manages agent profiles in runtime state with persistence to state.json."""

    def __init__(self, config: Config, state_store: StateStore):
        self.config = config
        self.state_store = state_store

    def list_profiles(self) -> dict[str, AgentProfile]:
        """Return all configured profiles."""
        return dict(self.config.agents.profiles)

    def get_profile(self, agent_id: str) -> AgentProfile | None:
        """Get a single profile by agent_id."""
        return self.config.agents.profiles.get(agent_id)

    def create_profile(
        self,
        agent_id: str,
        *,
        model: str | None = None,
        background_model: str | None = None,
        context_window: int | None = None,
        background_context_window: int | None = None,
        session_store: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tool_iterations: int | None = None,
        reasoning_effort: str | None = None,
        skills: list[str] | None = None,
        system_identity: str | None = None,
        discord_channels: list[str] | None = None,
        display_name: str | None = None,
        avatar_url: str | None = None,
        discord_webhook_url: str | None = None,
    ) -> AgentProfile:
        """Create a new agent profile and persist to config."""
        if agent_id in self.config.agents.profiles:
            raise ValueError(f"Profile '{agent_id}' already exists")

        profile = AgentProfile(
            model=model,
            background_model=background_model,
            context_window=context_window,
            background_context_window=background_context_window,
            session_store=session_store,
            max_tokens=max_tokens,
            temperature=temperature,
            max_tool_iterations=max_tool_iterations,
            reasoning_effort=reasoning_effort,
            skills=skills,
            system_identity=system_identity,
            discord_channels=discord_channels or [],
            display_name=display_name,
            avatar_url=avatar_url,
            discord_webhook_url=discord_webhook_url,
        )
        self.config.agents.profiles[agent_id] = profile
        self.state_store.upsert_profile(agent_id, profile)
        logger.info("Created agent profile '{}'", agent_id)
        return profile

    def update_profile(self, agent_id: str, **kwargs: Any) -> AgentProfile:
        """Update fields on an existing profile and persist."""
        profile = self.config.agents.profiles.get(agent_id)
        if not profile:
            raise ValueError(f"Profile '{agent_id}' not found")

        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                raise ValueError(f"Unknown profile field: {key}")

        self.state_store.upsert_profile(agent_id, profile)
        logger.info("Updated agent profile '{}'", agent_id)
        return profile

    def delete_profile(self, agent_id: str) -> None:
        """Remove a profile from config and persist."""
        if agent_id not in self.config.agents.profiles:
            raise ValueError(f"Profile '{agent_id}' not found")
        del self.config.agents.profiles[agent_id]
        self.state_store.delete_profile(agent_id)
        logger.info("Deleted agent profile '{}'", agent_id)
