import pytest

from nanobot.config.schema import Config
from nanobot.relay.security import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    build_safe_log_metadata,
    is_route_allowed,
    sanitize_forward_headers,
    validate_oauth_callback_url,
    validate_request_body_size,
)


def test_route_allowlist_matches_dynamic_paths_and_blocks_writes() -> None:
    assert is_route_allowed("GET", "/api/agents/alice/context/live")
    assert is_route_allowed("POST", "/api/graph/search")
    assert is_route_allowed("POST", "/api/agents")
    assert not is_route_allowed("DELETE", "/api/config")


def test_request_body_size_limit_enforced() -> None:
    validate_request_body_size("ok")

    with pytest.raises(ValueError):
        validate_request_body_size(
            b"x" * (DEFAULT_MAX_REQUEST_BODY_BYTES + 1),
            max_request_body_bytes=DEFAULT_MAX_REQUEST_BODY_BYTES,
        )


def test_header_sanitization_removes_sensitive_headers() -> None:
    sanitized = sanitize_forward_headers(
        {
            "Authorization": "Bearer secret",
            "Cookie": "session=secret",
            "X-Forwarded-For": "1.2.3.4",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )

    assert "Authorization" not in sanitized
    assert "Cookie" not in sanitized
    assert "X-Forwarded-For" not in sanitized
    assert sanitized["Content-Type"] == "application/json"
    assert sanitized["Accept"] == "application/json"


def test_log_metadata_never_includes_bodies() -> None:
    metadata = build_safe_log_metadata(
        method="post",
        path="/api/graph/search?q=test",
        status_code=200,
        headers={"Authorization": "secret", "Accept": "application/json"},
    )

    assert metadata["method"] == "POST"
    assert metadata["path"] == "/api/graph/search"
    assert metadata["log_request_body"] is False
    assert metadata["log_response_body"] is False
    assert metadata["headers"] == {"Accept": "application/json"}


def test_oauth_callback_url_validation_uses_registered_patterns() -> None:
    assert validate_oauth_callback_url(
        "https://relay.agentshq.io/api/i/nanobot-mom/integrations/google/callback"
    )
    assert validate_oauth_callback_url(
        "http://localhost:3000/api/i/nanobot-mom/integrations/google/callback"
    )
    assert not validate_oauth_callback_url(
        "https://relay.agentshq.io/api/i/nanobot-mom/integrations/google/token"
    )
    assert not validate_oauth_callback_url(
        "ftp://relay.agentshq.io/api/i/nanobot-mom/integrations/google/callback"
    )


def test_config_schema_accepts_relay_settings() -> None:
    config = Config.model_validate(
        {
            "relay": {
                "enabled": True,
                "url": "wss://relay.agentshq.io/ws/tunnel",
                "token": "inst_abc123",
                "instanceId": "nanobot-mom",
                "reconnectIntervalS": 3,
                "maxReconnectIntervalS": 30,
            }
        }
    )

    assert config.relay.enabled is True
    assert config.relay.instance_id == "nanobot-mom"
    assert config.relay.max_reconnect_interval_s == 30
