import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.agent_manager import AgentManagerTool
from nanobot.config.schema import Config


class _FakeDiscordChannel:
    def __init__(self):
        self.created_channels = []
        self.created_webhooks = []
        self.registered_webhooks = []

    async def create_guild_channel(self, guild_id: str, name: str, topic: str | None = None, category_id: str | None = None, channel_type: int = 0):
        self.created_channels.append({
            'guild_id': guild_id,
            'name': name,
            'topic': topic,
            'category_id': category_id,
            'channel_type': channel_type,
        })
        return f'ch-{name}'

    async def create_channel_webhook(self, channel_id: str, name: str, avatar_url: str | None = None):
        self.created_webhooks.append({
            'channel_id': channel_id,
            'name': name,
            'avatar_url': avatar_url,
        })
        return f'https://discord.test/webhooks/{channel_id}/token'

    def register_webhook(self, channel_id: str, webhook_url: str, display_name: str | None = None, avatar_url: str | None = None):
        self.registered_webhooks.append({
            'channel_id': channel_id,
            'webhook_url': webhook_url,
            'display_name': display_name,
            'avatar_url': avatar_url,
        })


class _FakeChannelManager:
    def __init__(self, discord):
        self._discord = discord

    def get_discord_channel(self):
        return self._discord


class _FakeProfileManager:
    def __init__(self, config: Config):
        self.config = config

    def create_profile(self, agent_id: str, **kwargs):
        from nanobot.config.schema import AgentProfile

        profile = AgentProfile(**kwargs)
        self.config.agents.profiles[agent_id] = profile
        return profile

    def get_profile(self, agent_id: str):
        return self.config.agents.profiles.get(agent_id)

    def delete_profile(self, agent_id: str):
        self.config.agents.profiles.pop(agent_id, None)


class _FakeRouter:
    def __init__(self, base_workspace: Path, config: Config, source_instance=None):
        self.config = config
        self._default_agent_id = 'general'
        self._agents = {}
        self._base_workspace = base_workspace
        if source_instance is not None:
            self._agents[source_instance.agent_id] = source_instance

    @property
    def agents(self):
        return self._agents

    def get_agent(self, agent_id: str):
        return self._agents.get(agent_id)

    async def create_agent(self, agent_id: str, profile):
        workspace = self._base_workspace / 'agents' / agent_id
        (workspace / 'memory').mkdir(parents=True, exist_ok=True)
        (workspace / 'skills').mkdir(parents=True, exist_ok=True)
        self._agents[agent_id] = SimpleNamespace(agent_id=agent_id, profile=profile, workspace=workspace)
        return self._agents[agent_id]

    async def remove_agent(self, agent_id: str):
        self._agents.pop(agent_id, None)


@pytest.mark.asyncio
async def test_manage_agents_create_always_creates_webhook_and_model_topic(tmp_path: Path):
    config = Config.model_validate({'agents': {'defaults': {'workspace': str(tmp_path), 'model': 'anthropic-direct/claude-opus-4-6'}}})
    discord = _FakeDiscordChannel()
    router = _FakeRouter(tmp_path, config)
    profile_manager = _FakeProfileManager(config)
    tool = AgentManagerTool(router, profile_manager, _FakeChannelManager(discord), guild_id='guild-1')

    result = json.loads(await tool.execute(action='create', agent_id='helper', display_name='Helper'))

    assert result['status'] == 'created'
    assert result['agent']['webhook_enabled'] is True
    assert discord.created_channels[0]['name'] == 'helper'
    assert discord.created_channels[0]['topic'] == 'anthropic-direct/claude-opus-4-6'
    assert result['channel']['topic'] == 'anthropic-direct/claude-opus-4-6'
    assert discord.created_webhooks[0]['name'] == 'Helper'
    assert discord.registered_webhooks[0]['display_name'] == 'Helper'
    profile = config.agents.profiles['helper']
    assert profile.display_name == 'Helper'
    assert profile.discord_webhook_url is not None
    assert profile.discord_channels == ['ch-helper']


@pytest.mark.asyncio
async def test_manage_agents_clone_copies_profile_memory_skills_and_history(tmp_path: Path):
    config = Config.model_validate({'agents': {'defaults': {'workspace': str(tmp_path), 'model': 'anthropic-direct/claude-opus-4-6'}}})
    source_workspace = tmp_path / 'agents' / 'source'
    (source_workspace / 'memory' / 'history').mkdir(parents=True, exist_ok=True)
    (source_workspace / 'skills' / 'private-skill').mkdir(parents=True, exist_ok=True)
    (source_workspace / 'sessions').mkdir(parents=True, exist_ok=True)
    (source_workspace / 'memory' / 'MEMORY.md').write_text('source memory', encoding='utf-8')
    (source_workspace / 'memory' / 'history' / '2026-03-09.md').write_text('history entry', encoding='utf-8')
    (source_workspace / 'skills' / 'private-skill' / 'SKILL.md').write_text('skill body', encoding='utf-8')
    (source_workspace / 'sessions' / 'source-session.jsonl').write_text('{"role":"user","content":"hello"}\n', encoding='utf-8')
    (source_workspace / 'sessions' / 'source-session.context.jsonl').write_text('ignored\n', encoding='utf-8')

    source_profile = config.agents.profiles['source'] = SimpleNamespace(
        model='openai-codex/gpt-5.4',
        background_model='openai-codex/gpt-5.3-codex-spark',
        context_window=1000000,
        background_context_window=128000,
        session_store='sqlite',
        max_tokens=100000,
        temperature=0.7,
        max_tool_iterations=50,
        reasoning_effort='high',
        skills=['memory', 'github'],
        system_identity='source identity',
        discord_channels=['ch-source'],
        display_name='Source Agent',
        avatar_url=None,
        discord_webhook_url='https://discord.test/webhooks/ch-source/token',
    )
    source_instance = SimpleNamespace(agent_id='source', profile=source_profile, workspace=source_workspace)

    discord = _FakeDiscordChannel()
    router = _FakeRouter(tmp_path, config, source_instance=source_instance)
    profile_manager = _FakeProfileManager(config)
    tool = AgentManagerTool(router, profile_manager, _FakeChannelManager(discord), guild_id='guild-1')

    result = json.loads(await tool.execute(action='clone', source_agent_id='source', agent_id='clone', display_name='Clone Agent', copy_history=True, copy_sessions=True))

    assert result['status'] == 'cloned'
    assert result['agent']['source_agent_id'] == 'source'
    assert result['agent']['webhook_enabled'] is True
    assert discord.created_channels[0]['topic'] == 'openai-codex/gpt-5.4'
    assert result['channel']['topic'] == 'openai-codex/gpt-5.4'
    assert result['copied'] == {'memory': True, 'skills': True, 'history': True, 'sessions': True}
    clone_profile = config.agents.profiles['clone']
    assert clone_profile.model == 'openai-codex/gpt-5.4'
    assert clone_profile.background_model == 'openai-codex/gpt-5.3-codex-spark'
    assert clone_profile.session_store == 'sqlite'
    assert clone_profile.skills == ['memory', 'github']
    assert clone_profile.display_name == 'Clone Agent'
    clone_workspace = router.get_agent('clone').workspace
    assert (clone_workspace / 'memory' / 'MEMORY.md').read_text(encoding='utf-8') == 'source memory'
    assert (clone_workspace / 'memory' / 'history' / '2026-03-09.md').read_text(encoding='utf-8') == 'history entry'
    assert (clone_workspace / 'skills' / 'private-skill' / 'SKILL.md').read_text(encoding='utf-8') == 'skill body'
    assert (clone_workspace / 'sessions' / 'source-session.jsonl').exists()
    assert not (clone_workspace / 'sessions' / 'source-session.context.jsonl').exists()
