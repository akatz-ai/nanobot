import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from nanobot.dashboard import app as dashboard_app
from nanobot.session import context_inspection


def test_api_version_reports_runtime_metadata(monkeypatch) -> None:
    monkeypatch.setattr(dashboard_app, "_load_config", lambda: {"schemaVersion": 7})

    client = TestClient(dashboard_app.app)
    response = client.get("/api/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"]
    assert payload["api_version"] == "1"
    assert payload["schema_version"] == 7
    assert payload["python"]


def test_overview_spa_requests_and_displays_version_metadata() -> None:
    index_html = (Path(__file__).resolve().parents[1] / "nanobot" / "dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "api('/api/version')" in index_html
    assert "API Version" in index_html
    assert "Schema Version" in index_html


def test_agents_spa_contains_create_and_clone_flows() -> None:
    index_html = (Path(__file__).resolve().parents[1] / "nanobot" / "dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "showAgentMutationModal({ mode: 'create' })" in index_html
    assert "showAgentMutationModal({ mode: 'clone', sourceAgent: agent })" in index_html
    assert "api('/api/agents'" in index_html


def test_context_spa_renders_canonical_prompt_assembly_labels() -> None:
    index_html = (Path(__file__).resolve().parents[1] / "nanobot" / "dashboard" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "buildPromptAssemblyCard(data.promptAssembly, data.context_window)" in index_html
    assert "Prompt Assembly" in index_html
    assert "Provider Observed" in index_html
    assert "Reserved Headroom" in index_html
    assert "Visible Conversation Slice" in index_html
    assert "Current conversation only. This is not the full context window." in index_html
    assert "Snapshot Provenance" in index_html
    assert "Pre-Compaction Snapshot" in index_html
    assert "Post-Compaction Snapshot" in index_html
    assert "Prompt Sections" in index_html


def test_build_prompt_assembly_payload_maps_required_contract_fields() -> None:
    payload = context_inspection.build_prompt_assembly_payload(
        {
            "prompt_assembly": {
                "estimated_total_tokens": 58000,
                "provider_observed_total_tokens": None,
                "should_compact": True,
                "budget": {
                    "compaction_trigger_tokens": 70000,
                    "reserve_tokens": 18000,
                    "compaction_threshold_ratio": 0.7,
                },
                "pre_compaction_snapshot": {
                    "trigger_snapshot": "pre_compaction",
                    "assembled_prompt_tokens": 58000,
                    "stable_cached_prefix_tokens": 22000,
                    "dynamic_turn_tokens": 36000,
                    "visible_conversation_slice_tokens": 33000,
                },
                "sections": [
                    {
                        "kind": "system_base",
                        "cache_scope": "static_prefix",
                        "estimated_tokens": 8000,
                        "source": "system_prompt",
                    },
                    {
                        "kind": "memory_md",
                        "cache_scope": "static_prefix",
                        "estimated_tokens": 9000,
                        "source": "MEMORY.md",
                    },
                    {
                        "kind": "current_user",
                        "cache_scope": "current_turn",
                        "estimated_tokens": 2000,
                        "source": "current_user",
                    },
                    {
                        "kind": "history_assistant",
                        "cache_scope": "history",
                        "estimated_tokens": 39000,
                        "source": "history:assistant:0",
                    },
                ],
            }
        },
        metadata={
            "post_compaction_snapshot": {
                "source": "post_compaction_rebuild",
                "assembled_prompt_tokens": 21000,
                "stable_cached_prefix_tokens": 15000,
                "dynamic_turn_tokens": 3000,
                "visible_conversation_slice_tokens": 3000,
            }
        },
        current_tokens=0,
        context_window=200000,
        compaction_events=[],
        user_message_index=0,
    )

    assert payload is not None
    assert payload["assembledPromptTokens"] == 58000
    assert payload["providerObservedPromptTokens"] is None
    assert payload["contextWindowTokens"] == 200000
    assert payload["compactionThresholdRatio"] == 0.7
    assert payload["compactionThresholdTokens"] == 70000
    assert payload["reservedHeadroomTokens"] == 18000
    assert payload["stablePrefixTokens"] == 22000
    assert payload["dynamicTurnTokens"] == 36000
    assert payload["visibleConversationSliceTokens"] == 33000
    assert payload["compactionTriggered"] is True
    assert payload["triggerSnapshot"] == "pre_compaction"
    assert payload["preCompactionSnapshot"]["snapshotId"] == "pre_compaction"
    assert payload["postCompactionSnapshot"]["snapshotId"] == "post_compaction_rebuild"
    assert payload["sections"] == [
        {
            "id": "system_prompt",
            "label": "system_base",
            "kind": "system",
            "stable": True,
            "turnScoped": False,
            "tokenEstimate": 8000,
            "source": "system_prompt",
        },
        {
            "id": "MEMORY.md",
            "label": "memory_md",
            "kind": "memory",
            "stable": True,
            "turnScoped": False,
            "tokenEstimate": 9000,
            "source": "MEMORY.md",
        },
        {
            "id": "current_user",
            "label": "current_user",
            "kind": "conversation",
            "stable": False,
            "turnScoped": True,
            "tokenEstimate": 2000,
            "source": "current_user",
        },
        {
            "id": "history:assistant:0",
            "label": "history_assistant",
            "kind": "conversation",
            "stable": False,
            "turnScoped": False,
            "tokenEstimate": 39000,
            "source": "history:assistant:0",
        },
    ]


def test_build_context_inspection_response_prefers_compaction_log_post_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "discord_123.jsonl"
    session_bundle = {
        "path": session_path,
        "metadata": {
            "key": "discord:123",
            "last_consolidated": 2,
            "post_compaction_snapshot": {
                "source": "metadata_post_compaction",
                "assembled_prompt_tokens": 12000,
                "stable_cached_prefix_tokens": 8000,
                "dynamic_turn_tokens": 2000,
                "visible_conversation_slice_tokens": 2000,
            },
        },
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "older"},
            {"role": "user", "content": "latest"},
            {"role": "assistant", "content": "reply"},
        ],
        "usage_summary": {"context_window": 100000},
        "key": "discord:123",
    }
    monkeypatch.setattr(
        context_inspection,
        "load_context_log",
        lambda path: [
            {
                "turn_index": 5,
                "user_message_index": 2,
                "system_prompt_hash": "new-hash",
                "prompt_assembly": {
                    "estimated_total_tokens": 64000,
                    "provider_observed_total_tokens": 65000,
                    "should_compact": True,
                    "budget": {
                        "compaction_trigger_tokens": 70000,
                        "reserve_tokens": 18000,
                        "compaction_threshold_ratio": 0.7,
                    },
                    "pre_compaction_snapshot": {
                        "assembled_prompt_tokens": 64000,
                        "stable_cached_prefix_tokens": 24000,
                        "dynamic_turn_tokens": 4000,
                        "visible_conversation_slice_tokens": 36000,
                    },
                    "sections": [
                        {
                            "kind": "system_base",
                            "cache_scope": "static_prefix",
                            "estimated_tokens": 9000,
                            "source": "system_prompt",
                        },
                        {
                            "kind": "resume_notice",
                            "cache_scope": "dynamic_system",
                            "estimated_tokens": 300,
                            "source": "resume_notice",
                        },
                    ],
                },
            }
        ],
    )
    monkeypatch.setattr(
        context_inspection,
        "load_compaction_log",
        lambda path: [
            {
                "pre_context": {
                    "prompt_assembly_snapshot": {
                        "trigger_snapshot": "pre_compaction",
                    },
                },
                "post_context": {
                    "first_kept_index": 2,
                    "new_last_consolidated": 2,
                    "prompt_assembly_snapshot": {
                        "trigger_snapshot": "post_compaction",
                        "source": "compaction_log_post_snapshot",
                        "assembled_prompt_tokens": 18000,
                        "stable_cached_prefix_tokens": 13000,
                        "dynamic_turn_tokens": 2000,
                        "visible_conversation_slice_tokens": 3000,
                    },
                }
            }
        ],
    )

    response = context_inspection.build_context_inspection_response(
        agent_name="alice",
        session_bundle=session_bundle,
        model="openai-codex/gpt-5.4",
    )

    assert response["sessionId"] == "discord:123"
    assert response["turnIndex"] == 5
    assert response["first_kept_index"] == 2
    assert response["messages_in_window"] == 2
    assert response["messages"] == session_bundle["messages"][2:]
    assert response["promptAssembly"]["triggerSnapshot"] == "pre_compaction"
    assert response["promptAssembly"]["postCompactionSnapshot"]["snapshotId"] == "compaction_log_post_snapshot"
    assert response["promptAssembly"]["providerObservedPromptTokens"] == 65000
    assert response["promptAssembly"]["sections"][0]["stable"] is True
    assert response["promptAssembly"]["sections"][1]["turnScoped"] is True


def test_build_context_inspection_response_raises_for_missing_turn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "discord_123.jsonl"
    session_bundle = {
        "path": session_path,
        "metadata": {"key": "discord:123", "last_consolidated": 0},
        "messages": [],
        "usage_summary": {},
        "key": "discord:123",
    }
    monkeypatch.setattr(context_inspection, "load_context_log", lambda path: [])
    monkeypatch.setattr(context_inspection, "load_compaction_log", lambda path: [])

    with pytest.raises(LookupError, match="Turn 7 not found for 'alice'"):
        context_inspection.build_context_inspection_response(
            agent_name="alice",
            session_bundle=session_bundle,
            model="openai-codex/gpt-5.4",
            requested_turn_index=7,
        )


def test_agent_context_live_exposes_prompt_assembly_contract(monkeypatch, tmp_path: Path) -> None:
    session_path = tmp_path / "discord_123.jsonl"
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        dashboard_app,
        "_load_config",
        lambda: {
            "agents": {
                "defaults": {"model": "openai-codex/gpt-5.4"},
                "profiles": {"alice": {"model": "openai-codex/gpt-5.4"}},
            }
        },
    )
    monkeypatch.setattr(
        dashboard_app,
        "_load_primary_session_bundle",
        lambda name, profile: {
            "path": session_path,
            "metadata": {
                "key": "discord:123",
                "last_consolidated": 40,
                "post_compaction_snapshot": {
                    "trigger_snapshot": "post_compaction",
                    "source": "post_compaction_rebuild",
                    "assembled_prompt_tokens": 21000,
                    "stable_cached_prefix_tokens": 15000,
                    "dynamic_turn_tokens": 3000,
                    "visible_conversation_slice_tokens": 3000,
                },
            },
            "messages": [{"role": "user", "content": "hi"}],
            "compactions": [],
            "usage_summary": {},
            "key": "discord:123",
            "channel_id": "123",
        },
    )
    monkeypatch.setattr(
        dashboard_app,
        "build_context_inspection_response",
        lambda **kwargs: captured.update(kwargs) or {
            "sessionId": "discord:123",
            "turnIndex": 0,
            "promptAssembly": {
                "assembledPromptTokens": 58000,
                "providerObservedPromptTokens": 59000,
                "contextWindowTokens": 200000,
                "compactionThresholdRatio": 0.7,
                "compactionThresholdTokens": 70000,
                "reservedHeadroomTokens": 18000,
                "stablePrefixTokens": 22000,
                "dynamicTurnTokens": 36000,
                "visibleConversationSliceTokens": 33000,
                "compactionTriggered": False,
                "triggerSnapshot": "pre_compaction",
                "preCompactionSnapshot": {
                    "snapshotId": "pre_compaction",
                    "assembledPromptTokens": 58000,
                    "stablePrefixTokens": 22000,
                    "dynamicTurnTokens": 36000,
                    "visibleConversationSliceTokens": 33000,
                },
                "postCompactionSnapshot": {
                    "snapshotId": "post_compaction_rebuild",
                    "assembledPromptTokens": 21000,
                    "stablePrefixTokens": 15000,
                    "dynamicTurnTokens": 3000,
                    "visibleConversationSliceTokens": 3000,
                },
                "sections": [
                    {
                        "id": "system_prompt",
                        "label": "system_base",
                        "kind": "system",
                        "stable": True,
                        "turnScoped": False,
                        "tokenEstimate": 8000,
                        "source": "system_prompt",
                    },
                    {
                        "id": "current_user",
                        "label": "current_user",
                        "kind": "conversation",
                        "stable": False,
                        "turnScoped": True,
                        "tokenEstimate": 2000,
                        "source": "current_user",
                    },
                ],
            },
        },
    )

    client = TestClient(dashboard_app.app)
    response = client.get("/api/agents/alice/context/live")

    assert response.status_code == 200
    payload = response.json()
    assert captured["agent_name"] == "alice"
    assert captured["model"] == "openai-codex/gpt-5.4"
    assert captured["requested_turn_index"] is None
    assert captured["session_bundle"]["path"] == session_path
    assert payload["sessionId"] == "discord:123"
    assert payload["turnIndex"] == 0
    prompt = payload["promptAssembly"]
    assert prompt["assembledPromptTokens"] == 58000
    assert prompt["providerObservedPromptTokens"] == 59000
    assert prompt["contextWindowTokens"] == 200000
    assert prompt["compactionThresholdRatio"] == 0.7
    assert prompt["compactionThresholdTokens"] == 70000
    assert prompt["reservedHeadroomTokens"] == 18000
    assert prompt["stablePrefixTokens"] == 22000
    assert prompt["dynamicTurnTokens"] == 36000
    assert prompt["visibleConversationSliceTokens"] == 33000
    assert prompt["triggerSnapshot"] == "pre_compaction"
    assert prompt["postCompactionSnapshot"]["snapshotId"] == "post_compaction_rebuild"
    assert any(section["stable"] for section in prompt["sections"])
    assert any(section["turnScoped"] for section in prompt["sections"])


def test_session_context_route_resolves_by_session_id(monkeypatch, tmp_path: Path) -> None:
    session_path = tmp_path / "discord_123.jsonl"
    calls: list[dict[str, object]] = []
    config = {
        "agents": {
            "defaults": {"model": "openai-codex/gpt-5.4"},
            "profiles": {
                "alice": {"model": "openai-codex/gpt-5.4"},
                "bob": {"model": "openai-codex/gpt-5.4"},
            },
        }
    }
    monkeypatch.setattr(dashboard_app, "_load_config", lambda: config)

    def _bundle(name, profile):
        key = "discord:123" if name == "alice" else "discord:999"
        return {
            "path": session_path,
            "metadata": {"key": key, "last_consolidated": 0},
            "messages": [],
            "compactions": [],
            "usage_summary": {},
            "key": key,
            "channel_id": key.split(":", 1)[1],
        }

    monkeypatch.setattr(dashboard_app, "_load_primary_session_bundle", _bundle)
    monkeypatch.setattr(
        dashboard_app,
        "build_context_inspection_response",
        lambda **kwargs: calls.append(kwargs) or {
            "sessionId": kwargs["session_bundle"]["key"],
            "turnIndex": None,
            "promptAssembly": None,
        },
    )

    client = TestClient(dashboard_app.app)
    response = client.get("/api/context/discord:123")
    missing = client.get("/api/context/discord:nope")

    assert response.status_code == 200
    assert response.json()["sessionId"] == "discord:123"
    assert len(calls) == 1
    assert calls[0]["agent_name"] == "alice"
    assert calls[0]["session_bundle"]["key"] == "discord:123"
    assert missing.status_code == 404
