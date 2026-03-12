from types import SimpleNamespace

from nanobot.discord.system_status import _channel_topic, _current_snapshot_tokens, collect_system_status, render_dashboard


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
        return SimpleNamespace(
            metadata=payload.get("metadata", {}),
            _sqlite_revision=payload.get("revision"),
            get_message_count=lambda: payload.get("message_count", 0),
        )


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
                            "message_count": 10,
                            "revision": 7,
                            "metadata": {
                                "usage_snapshot": {
                                    "total_input_tokens": 25000,
                                    "message_index": 10,
                                    "source": "provider_usage",
                                    "revision": 7,
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
                            "message_count": 10,
                            "revision": 4,
                            "metadata": {
                                "usage_snapshot": {
                                    "total_input_tokens": 25000,
                                    "message_index": 10,
                                    "source": "estimated_current_prompt",
                                    "revision": 4,
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


def test_current_snapshot_tokens_rejects_non_current_context_sources():
    session = SimpleNamespace(
        metadata={
            "usage_snapshot": {
                "total_input_tokens": 99999,
                "message_index": 10,
                "source": "compaction_trigger",
                "revision": 1,
            }
        },
        _sqlite_revision=1,
        get_message_count=lambda: 10,
    )
    assert _current_snapshot_tokens(SimpleNamespace(), session) == 0


def test_current_snapshot_tokens_rejects_stale_revision():
    session = SimpleNamespace(
        metadata={
            "usage_snapshot": {
                "total_input_tokens": 25000,
                "message_index": 10,
                "source": "provider_usage",
                "revision": 1,
            }
        },
        _sqlite_revision=2,
        get_message_count=lambda: 10,
    )
    assert _current_snapshot_tokens(SimpleNamespace(), session) == 0


def test_channel_topic_uses_model_string_only():
    assert _channel_topic("nanobot-dev", "openai-codex/gpt-5.4") == "openai-codex/gpt-5.4"


def test_render_dashboard_shows_per_agent_thresholds_and_current_snapshot_disclaimer():
    status = collect_system_status(
        SimpleNamespace(
            agents={
                "alpha": SimpleNamespace(
                    profile=SimpleNamespace(model="openai-codex/gpt-5.4", discord_channels=["123"]),
                    loop=_FakeLoop(
                        model="openai-codex/gpt-5.4",
                        context_window=200_000,
                        sessions={
                            "discord:123": {
                                "message_count": 10,
                                "revision": 1,
                                "metadata": {
                                    "usage_snapshot": {
                                        "total_input_tokens": 90_000,
                                        "message_index": 10,
                                        "source": "provider_usage",
                                        "revision": 1,
                                    }
                                },
                            }
                        },
                    ),
                ),
                "beta": SimpleNamespace(
                    profile=SimpleNamespace(model="openai-codex/gpt-5.4", discord_channels=["456"]),
                    loop=_FakeLoop(
                        model="openai-codex/gpt-5.4",
                        context_window=100_000,
                        sessions={
                            "discord:456": {
                                "message_count": 8,
                                "revision": 2,
                                "metadata": {
                                    "usage_snapshot": {
                                        "total_input_tokens": 25_000,
                                        "message_index": 8,
                                        "source": "provider_usage",
                                        "revision": 2,
                                    }
                                },
                            }
                        },
                    ),
                ),
            }
        )
    )

    rendered = render_dashboard(status)
    content = rendered[0]["components"][0]["content"]

    assert "Compaction threshold: 150k (75% of window)" in content
    assert "Compaction threshold: 75k (75% of window)" in content
    assert "Thresholds are shown per agent" in content
    assert "not a historical trigger value" in content
