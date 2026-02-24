"""Anthropic OAuth credential management with auto-refresh."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from loguru import logger

CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
TOKEN_ENDPOINT = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Refresh 10 minutes before expiry
REFRESH_BUFFER_MS = 600_000


def get_oauth_token(credentials_path: Path | None = None) -> str | None:
    """Read OAuth access token, auto-refreshing if expired or expiring soon.

    Priority:
    1. CLAUDE_CODE_OAUTH_TOKEN env var (for setup-token workflow)
    2. ~/.claude/.credentials.json (auto-populated by Claude CLI login)

    Returns None if no valid token found.
    """
    import os

    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if env_token:
        return env_token

    creds_path = credentials_path or CREDENTIALS_PATH
    if not creds_path.exists():
        return None

    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        token = oauth.get("accessToken", "")
        expires_at = oauth.get("expiresAt", 0)
        refresh_token = oauth.get("refreshToken", "")

        if not token:
            return None

        now_ms = int(time.time() * 1000)
        if expires_at and now_ms > (expires_at - REFRESH_BUFFER_MS):
            if refresh_token:
                logger.info("OAuth token expiring soon, attempting refresh...")
                new_token = _do_refresh(refresh_token, creds_path)
                if new_token:
                    return new_token
                logger.warning("Token refresh failed. Using existing token.")
            else:
                logger.warning("OAuth token expiring soon and no refresh token. Run: claude login")

        return token
    except Exception as e:
        logger.warning("Failed to read OAuth credentials: {}", e)
        return None


def refresh_oauth_token(credentials_path: Path | None = None) -> str | None:
    """Force-refresh the OAuth token using the stored refresh token.

    Returns the new access token, or None on failure.
    """
    creds_path = credentials_path or CREDENTIALS_PATH
    if not creds_path.exists():
        return None

    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        refresh_token = data.get("claudeAiOauth", {}).get("refreshToken", "")
        if not refresh_token:
            logger.warning("No refresh token available")
            return None
        return _do_refresh(refresh_token, creds_path)
    except Exception as e:
        logger.warning("Failed to refresh OAuth token: {}", e)
        return None


def _do_refresh(refresh_token: str, creds_path: Path) -> str | None:
    """Exchange refresh token for new access + refresh tokens."""
    try:
        resp = httpx.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=True,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error("OAuth refresh failed ({}): {}", resp.status_code, resp.text[:300])
            return None

        token_data = resp.json()
        new_access = token_data.get("access_token", "")
        new_refresh = token_data.get("refresh_token", "")
        expires_in = token_data.get("expires_in", 28800)  # default 8h

        if not new_access:
            logger.error("OAuth refresh returned no access_token")
            return None

        # Update credentials file
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        oauth["accessToken"] = new_access
        if new_refresh:
            oauth["refreshToken"] = new_refresh
        oauth["expiresAt"] = int(time.time() * 1000) + (expires_in * 1000)
        data["claudeAiOauth"] = oauth
        creds_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        logger.info("OAuth token refreshed successfully (expires in {}s)", expires_in)
        return new_access

    except Exception as e:
        logger.error("OAuth refresh error: {}", e)
        return None


def is_oauth_token(token: str) -> bool:
    """Check whether a token is an Anthropic OAuth token."""
    return token.startswith("sk-ant-oat")
