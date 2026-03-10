from types import SimpleNamespace

from nanobot.discord.system_status import _channel_topic, collect_system_status


class _FakeSessions:
    def __init__(self, sessions: dict[str, dict]):
        self._sessions = sessions

    def list_sessions(self):
        return [
            {
                "key": key,
                "updated_at": payload.get("updated_at", "2026-03-09T00:00:00"),
                "path": payload.get("path", ""),
            }
            for key, payload in self._sessions.items()
        ]

    def get_or_create(self, key: str):
        payload = self._sessions[key]
        return SimpleNamespace(metadata=payload.get("metadata", {}))


class _FakeLoop:
    def __init__(self, *, model: str, context_window: int, sessions: dict[str, dict], live_tokens: dict[str, int] | None = None):
        self.model = model
        self._context_window = context_window
        self._compaction_token_threshold = int(context_window * 0.75)
        self.sessions = _FakeSessions(sessions)
        self._last_input_tokens = live_tokens or {}

    def _get_context_window_size(self) -> int:
        return self._context_window


def test_collect_system_status_uses_persisted_usage_snapshot_after_restart():
    router = SimpleNamespace(
        agents={
            "nanobot-dev": SimpleNamespace(
                profile=SimpleNamespace(model="openai-codex/gpt-5.4", discord_channels=["123"]),
                loop=_FakeLoop(
                    model="openai-codex/gpt-5.4",
                    context_window=100000,
                    sessions={
                        "discord:123": {
                            "metadata": {
                                "usage_snapshot": {
                                    "total_input_tokens": 25000,
                                    "message_index": 10,
                                }
                            }
                        }
                    },
                    live_tokens={},
                ),
            )
        }
    )

    status = collect_system_status(router)
    agent = status.agents[0]
    assert agent.agent_id == "nanobot-dev"
    assert agent.current_input_tokens == 25000
    assert round(agent.utilization_pct, 2) == 0.25
    assert agent.is_idle is False


def test_collect_system_status_prefers_snapshot_over_stale_live_tokens():
    router = SimpleNamespace(
        agents={
            "nanobot-dev": SimpleNamespace(
                profile=SimpleNamespace(model="openai-codex/gpt-5.4", discord_channels=["123"]),
                loop=_FakeLoop(
                    model="openai-codex/gpt-5.4",
                    context_window=100000,
                    sessions={
                        "discord:123": {
                            "metadata": {
                                "usage_snapshot": {
                                    "total_input_tokens": 25000,
                                    "message_index": 10,
                                }
                            }
                        }
                    },
                    live_tokens={"discord:123": 30000},
                ),
            )
        }
    )

    status = collect_system_status(router)
    agent = status.agents[0]
    assert agent.current_input_tokens == 25000
    assert round(agent.utilization_pct, 2) == 0.25


def test_channel_topic_uses_model_string_only():
    assert _channel_topic("nanobot-dev", "openai-codex/gpt-5.4") == "openai-codex/gpt-5.4"
