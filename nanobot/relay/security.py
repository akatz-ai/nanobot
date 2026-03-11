"""Security guardrails for the AgentsHQ relay."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

ALLOWED_ROUTES = [
    "GET /api/version",
    "GET /api/overview",
    "GET /api/agents",
    "POST /api/agents",
    "GET /api/context/{sessionId}",
    "GET /api/agents/{name}/context/live",
    "GET /api/agents/{name}/memory",
    "GET /api/agents/{name}/sessions",
    "GET /api/skills",
    "GET /api/channels",
    "GET /api/integrations",
    "GET /api/graph/stats",
    "GET /api/graph/memories",
    "POST /api/graph/search",
    "GET /api/logs",
    "GET /api/config",
]

DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024
DEFAULT_OAUTH_CALLBACK_PATTERNS = (
    r"^https://[^/]+/api/i/[^/]+/integrations/[A-Za-z0-9._-]+/callback$",
    r"^http://localhost(?::\d+)?/api/i/[^/]+/integrations/[A-Za-z0-9._-]+/callback$",
)
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "cf-connecting-ip",
        "fly-client-ip",
        "x-instance-token",
    }
)


@dataclass(frozen=True)
class RateLimitConfig:
    """Default per-tunnel rate limits for the read-only relay MVP."""

    requests_per_minute: int = 120
    burst: int = 20
    concurrent_requests: int = 10


def is_route_allowed(
    method: str,
    path: str,
    allowed_routes: list[str] | tuple[str, ...] = tuple(ALLOWED_ROUTES),
) -> bool:
    """Return True when the method/path pair matches the relay allowlist."""
    normalized_method = method.upper().strip()
    normalized_path = _strip_query(path)

    for route in allowed_routes:
        allowed_method, allowed_path = route.split(" ", 1)
        if normalized_method != allowed_method:
            continue
        pattern = "^" + re.sub(r"\{[^/]+\}", r"[^/]+", allowed_path) + "$"
        if re.fullmatch(pattern, normalized_path):
            return True
    return False


def validate_request_body_size(
    body: bytes | str | None,
    *,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
) -> None:
    """Raise when the forwarded request body exceeds the configured max size."""
    if body is None:
        return
    payload = body if isinstance(body, bytes) else body.encode("utf-8")
    if len(payload) > max_request_body_bytes:
        raise ValueError(
            f"Request body exceeds {max_request_body_bytes} bytes"
        )


def sanitize_forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Strip sensitive and hop-by-hop headers before forwarding."""
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        normalized = key.lower()
        if normalized in HOP_BY_HOP_HEADERS or normalized in SENSITIVE_HEADERS:
            continue
        sanitized[key] = value
    return sanitized


def build_safe_log_metadata(
    *,
    method: str,
    path: str,
    status_code: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build log-safe metadata without request or response bodies."""
    payload: dict[str, object] = {
        "method": method.upper().strip(),
        "path": _strip_query(path),
        "log_request_body": False,
        "log_response_body": False,
    }
    if status_code is not None:
        payload["status_code"] = status_code
    if headers is not None:
        payload["headers"] = sanitize_forward_headers(headers)
    return payload


def validate_oauth_callback_url(
    url: str,
    *,
    registered_patterns: tuple[str, ...] = DEFAULT_OAUTH_CALLBACK_PATTERNS,
) -> bool:
    """Return True when an OAuth callback URL matches a registered pattern."""
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"} or not parts.netloc or parts.fragment:
        return False
    normalized = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return any(re.fullmatch(pattern, normalized) for pattern in registered_patterns)


def _strip_query(path: str) -> str:
    """Drop any query string or fragment from a request path."""
    return path.split("?", 1)[0].split("#", 1)[0]
