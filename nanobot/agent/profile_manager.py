"""Profile manager: CRUD operations on agent profiles with config persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.loader import save_config
from nanobot.config.schema import AgentProfile, Config


class AgentProfileManager:
    """Manages agent profiles in config with persistence to config.json."""

    def __init__(self, config: Config, config_path: Path | None = None):
        self.config = config
        self.config_path = config_path

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
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tool_iterations: int | None = None,
        memory_window: int | None = None,
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
            max_tokens=max_tokens,
            temperature=temperature,
            max_tool_iterations=max_tool_iterations,
            memory_window=memory_window,
            skills=skills,
            system_identity=system_identity,
            discord_channels=discord_channels or [],
            display_name=display_name,
            avatar_url=avatar_url,
            discord_webhook_url=discord_webhook_url,
        )
        self.config.agents.profiles[agent_id] = profile
        self._save()
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

        self._save()
        logger.info("Updated agent profile '{}'", agent_id)
        return profile

    def delete_profile(self, agent_id: str) -> None:
        """Remove a profile from config and persist."""
        if agent_id not in self.config.agents.profiles:
            raise ValueError(f"Profile '{agent_id}' not found")
        del self.config.agents.profiles[agent_id]
        self._save()
        logger.info("Deleted agent profile '{}'", agent_id)

    def _save(self) -> None:
        """Persist current config to disk."""
        save_config(self.config, self.config_path)
