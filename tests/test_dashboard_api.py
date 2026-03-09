from pathlib import Path

from fastapi.testclient import TestClient

from nanobot.dashboard import app as dashboard_app


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
