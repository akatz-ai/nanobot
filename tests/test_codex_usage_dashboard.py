import json
from pathlib import Path

import httpx

from nanobot.discord.codex_usage_dashboard import CodexUsageDashboard, CodexUsageData, render_dashboard


def test_codex_usage_data_from_headers_parses_primary_secondary_and_spark() -> None:
    headers = httpx.Headers(
        {
            "x-codex-plan-type": "pro",
            "x-codex-active-limit": "codex",
            "x-codex-primary-used-percent": "12",
            "x-codex-secondary-used-percent": "34",
            "x-codex-primary-window-minutes": "300",
            "x-codex-secondary-window-minutes": "10080",
            "x-codex-primary-reset-at": "1773100631",
            "x-codex-secondary-reset-at": "1773535283",
            "x-codex-credits-has-credits": "False",
            "x-codex-bengalfox-primary-used-percent": "1",
            "x-codex-bengalfox-secondary-used-percent": "2",
            "x-codex-bengalfox-primary-window-minutes": "300",
            "x-codex-bengalfox-secondary-window-minutes": "10080",
            "x-codex-bengalfox-primary-reset-at": "1773102652",
            "x-codex-bengalfox-secondary-reset-at": "1773650375",
            "x-codex-bengalfox-limit-name": "GPT-5.3-Codex-Spark",
        }
    )

    data = CodexUsageData.from_headers(headers)

    assert data.plan_type == "pro"
    assert data.active_limit == "codex"
    assert len(data.buckets) == 2
    assert data.buckets[0].label == "Codex"
    assert data.buckets[0].used_primary_pct == 12
    assert data.buckets[0].used_secondary_pct == 34
    assert data.buckets[1].label == "GPT-5.3-Codex-Spark"
    assert data.buckets[1].used_primary_pct == 1
    assert data.buckets[1].used_secondary_pct == 2


def test_render_dashboard_includes_used_left_and_reset_lines() -> None:
    data = CodexUsageData.from_headers(
        httpx.Headers(
            {
                "x-codex-plan-type": "pro",
                "x-codex-active-limit": "codex",
                "x-codex-primary-used-percent": "10",
                "x-codex-secondary-used-percent": "25",
                "x-codex-primary-window-minutes": "300",
                "x-codex-secondary-window-minutes": "10080",
                "x-codex-primary-reset-at": "1773100631",
                "x-codex-secondary-reset-at": "1773535283",
            }
        )
    )

    components = render_dashboard(data)
    assert len(components) == 1
    content = components[0]["components"][0]["content"]
    assert "## 🤖 Codex Usage" in content
    assert "5h Window" in content
    assert "7d Window" in content
    assert "10.0% used" in content
    assert "90.0% left" in content
    assert "25.0% used" in content
    assert "75.0% left" in content
    assert "Plan: Pro" in content


def test_codex_dashboard_persists_message_id_to_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")

    dash = CodexUsageDashboard(
        discord_token="discord",
        channel_id="chan-codex",
        config_path=str(config_path),
    )
    dash.message_id = "codex-msg"
    dash._persist_message_id()

    state_data = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_data["channels"]["discord"]["codexUsage"]["messageId"] == "codex-msg"
