"""FastAPI dashboard application for nanobot memory system audit."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import time
import tomllib
from collections import deque
from datetime import date, datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yaml

import nanobot
from nanobot.agent.profile_manager import AgentProfileManager
from nanobot.agent.workspace import init_agent_workspace
from nanobot.bus.queue import MessageBus
from nanobot.channels.discord import DiscordChannel
from nanobot.config.loader import get_config_path, get_data_dir, get_state_path, load_config, load_config_data
from nanobot.config.state import StateStore
from nanobot.cron.service import CronService
from nanobot.session.context_log import load_context_log
from nanobot.session.extraction_log import load_extraction_log
from nanobot.session.manager import SessionManager
from nanobot.session.store import SQLiteSessionManager
from nanobot.session.usage_log import get_session_summary

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Nanobot Memory Dashboard", version=nanobot.__version__)

_default_cors_origins = [
    'http://localhost:4173',
    'http://127.0.0.1:4173',
    'http://localhost:4174',
    'http://127.0.0.1:4174',
    'http://localhost:9347',
    'http://127.0.0.1:9347',
]
_env_cors = [o.strip() for o in os.getenv('NANOBOT_DASHBOARD_CORS_ORIGINS', '').split(',') if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(set(_default_cors_origins + _env_cors)),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/dashboard/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROCESS_STARTED_AT = datetime.now(timezone.utc)
PROCESS_START_MONOTONIC = time.monotonic()

_SESSION_SIDEcar_SUFFIXES = (
    ".context",
    ".compaction",
    ".usage",
    ".extraction",
    ".inbox",
)
_DEFAULT_CONTEXT_WINDOW = 200_000
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-3-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}
_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\S+\s+\S+)\s+\|\s+"
    r"(?P<level>[A-Z]+)\s+\|\s+"
    r"(?P<source>[^|]+)\|\s+"
    r"(?P<message>.*)$"
)

# ---------------------------------------------------------------------------
# Config / workspace helpers
# ---------------------------------------------------------------------------

_config_cache: dict[str, Any] | None = None
_config_mtimes: tuple[float, float] = (0, 0)


def _load_config() -> dict[str, Any]:
    global _config_cache, _config_mtimes
    config_path = get_config_path()
    state_path = get_state_path(config_path)
    if not config_path.exists():
        raise RuntimeError(f"Config not found at {config_path}")
    mtimes = (
        config_path.stat().st_mtime,
        state_path.stat().st_mtime if state_path.exists() else 0,
    )
    if _config_cache is None or mtimes != _config_mtimes:
        _config_cache = load_config_data(config_path=config_path, state_path=state_path)
        _config_mtimes = mtimes
    return _config_cache


def _workspace() -> Path:
    cfg = _load_config()
    ws = cfg.get("agents", {}).get("defaults", {}).get("workspace", "~/.nanobot/workspace")
    return Path(ws).expanduser()


def _agent_dir(name: str) -> Path:
    ws = _workspace()
    d = ws / "agents" / name
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return d


def _agent_names() -> list[str]:
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    return list(profiles.keys())


def _base_config_and_store() -> tuple[Any, StateStore]:
    config_path = get_config_path()
    base_config = load_config(config_path)
    return base_config, StateStore.from_config_path(config_path)


def _discord_channel_category_id() -> str | None:
    state_path = get_state_path(get_config_path())
    if not state_path.exists():
        return None
    state = StateStore(state_path).load()
    return state.provisioning.discord.category_ids.get("AGENTS")


def _build_dashboard_discord_channel(config: Any) -> DiscordChannel:
    providers = config.providers
    return DiscordChannel(
        config.channels.discord,
        MessageBus(),
        groq_api_key=providers.groq.api_key or "",
        openai_api_key=providers.openai.api_key or "",
    )


def _copy_dashboard_workspace(
    *,
    workspace_root: Path,
    source_agent_id: str,
    target_agent_id: str,
    copy_history: bool,
    copy_sessions: bool,
) -> None:
    source_workspace = workspace_root / "agents" / source_agent_id
    target_workspace = workspace_root / "agents" / target_agent_id

    for rel in [Path("memory/MEMORY.md")]:
        src = source_workspace / rel
        dst = target_workspace / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    source_skills = source_workspace / "skills"
    target_skills = target_workspace / "skills"
    if source_skills.exists():
        for item in source_skills.iterdir():
            dst = target_skills / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                target_skills.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

    if copy_history:
        source_history = source_workspace / "memory" / "history"
        target_history = target_workspace / "memory" / "history"
        if source_history.exists():
            for item in source_history.iterdir():
                dst = target_history / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    target_history.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst)

    if copy_sessions:
        source_sessions = source_workspace / "sessions"
        target_sessions = target_workspace / "sessions"
        if source_sessions.exists():
            for item in source_sessions.iterdir():
                if item.name.endswith((
                    '.context.jsonl', '.usage.jsonl', '.extraction.jsonl',
                    '.evidence.sqlite', '.evidence.sqlite-wal', '.evidence.sqlite-shm',
                    '.db', '.db-wal', '.db-shm', '.sqlite', '.sqlite-wal', '.sqlite-shm',
                    '.inbox.jsonl',
                )):
                    continue
                dst = target_sessions / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    target_sessions.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dst)


def _read_text(p: Path) -> str | None:
    if p.exists() and p.is_file():
        return p.read_text(errors="replace")
    return None


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def _safe_filename(key: str) -> str:
    return key.replace(":", "_").replace("/", "_")


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _profile_value(profile: dict[str, Any], camel: str, snake: str) -> Any:
    if camel in profile:
        return profile.get(camel)
    return profile.get(snake)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso8601(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _message_timestamp(msg: dict[str, Any]) -> datetime | None:
    for key in ("timestamp", "_timestamp", "occurred_at"):
        ts = _parse_iso8601(msg.get(key))
        if ts is not None:
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def _is_base_session_file(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    stem = path.stem
    return not any(stem.endswith(suffix) for suffix in _SESSION_SIDEcar_SUFFIXES)


def _resolved_agent_model(profile: dict[str, Any], defaults: dict[str, Any]) -> str | None:
    return _profile_value(profile, "model", "model") or _profile_value(defaults, "model", "model")


def _resolved_agent_display_name(name: str, profile: dict[str, Any]) -> str:
    return _profile_value(profile, "displayName", "display_name") or name


def _context_window_for_model(
    model: str | None,
    usage_summary: dict[str, Any] | None = None,
) -> int:
    usage_window = _coerce_int((usage_summary or {}).get("context_window"), 0)
    if usage_window > 0:
        return usage_window

    model_lower = (model or "").lower()
    for key, size in _MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return size
    return _DEFAULT_CONTEXT_WINDOW


def _session_metadata_blob(meta: dict[str, Any]) -> dict[str, Any]:
    value = meta.get("metadata")
    if isinstance(value, dict):
        return value
    # SQLite-backed dashboard loading already returns the session metadata at top level.
    if isinstance(meta, dict):
        return meta
    return {}


def _session_usage_snapshot(meta: dict[str, Any]) -> dict[str, Any]:
    value = _session_metadata_blob(meta).get("usage_snapshot")
    return value if isinstance(value, dict) else {}


def _current_context_tokens(meta: dict[str, Any], usage_summary: dict[str, Any], context_window: int) -> int:
    """Prefer current prompt/window tokens, never cumulative session totals beyond context size."""
    usage_snapshot = _session_usage_snapshot(meta)
    snapshot_tokens = _coerce_int(usage_snapshot.get("total_input_tokens"), 0)
    if snapshot_tokens > 0:
        return snapshot_tokens

    summary_tokens = _coerce_int(usage_summary.get("total_input_tokens"), 0)
    if 0 < summary_tokens <= max(int(context_window), 1):
        return summary_tokens
    return 0


def _uptime_seconds() -> int:
    return int(max(0, time.monotonic() - PROCESS_START_MONOTONIC))


def _format_uptime(total_seconds: int) -> str:
    days, rem = divmod(max(total_seconds, 0), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nanobot_version() -> str:
    if nanobot.__version__:
        return nanobot.__version__

    for package_name in ("nanobot-ai", "nanobot"):
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue

    pyproject = _project_root() / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, "rb") as handle:
                data = tomllib.load(handle)
            return str(data.get("project", {}).get("version", "unknown"))
        except Exception:
            pass
    return "unknown"


def _sanitize_config_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_config_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_config_value(key, item) for item in value]
    if isinstance(value, str) and any(token in key.lower() for token in ("token", "secret", "password", "key")):
        if not value:
            return value
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "…" + value[-4:]
    return value


def _friendly_name(identifier: str) -> str:
    mapping = {
        "caldav": "CalDAV (iCloud)",
        "google-drive": "Google Drive",
        "filesystem": "Filesystem",
    }
    if identifier in mapping:
        return mapping[identifier]
    return identifier.replace("-", " ").replace("_", " ").title()


def _provider_display_name(identifier: str) -> str:
    mapping = {
        "anthropic": "Anthropic",
        "anthropic-direct": "Anthropic Direct",
        "anthropicDirect": "Anthropic Direct",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
        "google": "Google",
        "groq": "Groq",
        "ollama": "Ollama",
        "custom": "Custom Provider",
        "vllm": "vLLM",
        "deepseek": "DeepSeek",
        "gemini": "Gemini",
        "claudeCode": "Claude Code",
        "claude-code": "Claude Code",
        "openaiCodex": "OpenAI Codex",
        "githubCopilot": "GitHub Copilot",
    }
    if identifier in mapping:
        return mapping[identifier]
    return identifier.replace("-", " ").replace("_", " ").title()


def _google_drive_credentials_file() -> Path:
    return _workspace().parent / "credentials" / "google-drive.json"


def _google_drive_status(server_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    env = server_cfg.get("env", {})
    details: dict[str, Any] = {"oauth_supported": True}

    creds_file = _google_drive_credentials_file()
    if not creds_file.exists():
        return "error", {**details, "error": "Authorization required"}

    try:
        creds = json.loads(creds_file.read_text())
    except Exception as exc:
        return "error", {**details, "error": f"Invalid credentials file: {exc}"}

    has_access_token = bool(creds.get("token"))
    has_refresh_token = bool(creds.get("refresh_token"))
    details["has_refresh_token"] = has_refresh_token

    if has_access_token or has_refresh_token:
        return "connected", details
    return "error", {**details, "error": "Authorization required"}


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    frontmatter_text = text[4:end]
    body = text[end + 5:]
    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}, body


def _first_paragraph(markdown: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraph = " ".join(current).strip()
                if paragraph:
                    paragraphs.append(paragraph)
                current = []
            continue
        if line.startswith("#"):
            continue
        current.append(line)
    if current:
        paragraph = " ".join(current).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs[0] if paragraphs else ""


def _read_skill_info(skill_path: Path, location: str, agent: str | None = None) -> dict[str, Any]:
    text = skill_path.read_text(errors="replace")
    frontmatter, body = _split_frontmatter(text)
    name = str(frontmatter.get("name") or skill_path.parent.name)
    description = str(frontmatter.get("description") or _first_paragraph(body) or "")
    return {
        "name": name,
        "description": description,
        "location": location,
        "always": bool(frontmatter.get("always", False)),
        "available": True,
        "path": str(skill_path),
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _parse_session(path: Path) -> tuple[dict, list[dict]]:
    metadata, messages, _ = _parse_session_with_compactions(path)
    return metadata, messages


def _parse_session_with_compactions(path: Path) -> tuple[dict, list[dict], list[dict[str, Any]]]:
    metadata: dict = {}
    messages: list[dict] = []
    compactions: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("_type") == "metadata":
                metadata = data
            elif data.get("_type") == "compaction":
                compactions.append(data)
            else:
                messages.append(data)
    compactions.sort(key=lambda entry: (
        _coerce_int(entry.get("first_kept_index"), -1),
        entry.get("timestamp", ""),
    ))
    return metadata, messages, compactions


def _session_files(agent_dir: Path) -> list[Path]:
    sd = agent_dir / "sessions"
    if not sd.exists():
        return []
    return sorted(
        (path for path in sd.glob("*.jsonl") if _is_base_session_file(path)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _primary_session_path(name: str, profile: dict[str, Any]) -> Path | None:
    adir = _workspace() / "agents" / name
    if not adir.exists():
        return None
    session_dir = adir / "sessions"
    if not session_dir.exists():
        return None

    channel_ids = [
        str(channel_id)
        for channel_id in (_profile_value(profile, "discordChannels", "discord_channels") or [])
        if channel_id
    ]
    for channel_id in channel_ids:
        direct = session_dir / f"discord_{channel_id}.jsonl"
        if direct.exists():
            return direct

    for session_path in _session_files(adir):
        meta, _messages, _compactions = _parse_session_with_compactions(session_path)
        key = str(meta.get("key") or "")
        for channel_id in channel_ids:
            if channel_id in session_path.stem or channel_id in key:
                return session_path

    for session_path in _session_files(adir):
        meta, _messages, _compactions = _parse_session_with_compactions(session_path)
        key = str(meta.get("key") or "")
        if key.startswith("discord:") or session_path.stem.startswith("discord_"):
            return session_path

    sessions = _session_files(adir)
    return sessions[0] if sessions else None


def _session_public_key(path: Path, metadata: dict[str, Any]) -> str:
    return str(metadata.get("key") or path.stem)


def _session_channel_id(path: Path, metadata: dict[str, Any]) -> str | None:
    key = _session_public_key(path, metadata)
    if key.startswith("discord:"):
        return key.split(":", 1)[1]
    if path.stem.startswith("discord_"):
        return path.stem[len("discord_"):]
    return None


def _load_primary_session_bundle(name: str, profile: dict[str, Any]) -> dict[str, Any] | None:
    path = _primary_session_path(name, profile)
    if path is None:
        return None

    cfg = _load_config()
    defaults = cfg.get('agents', {}).get('defaults', {})
    session_store = str(
        _profile_value(profile, 'sessionStore', 'session_store')
        or defaults.get('sessionStore')
        or defaults.get('session_store')
        or ''
    ).lower()
    channel_id = _session_channel_id(path, {})
    key = f'discord:{channel_id}' if channel_id else None

    if session_store == 'sqlite' and key:
        workspace = _workspace() / 'agents' / name
        manager = SQLiteSessionManager(workspace)
        session = manager.get_or_create(key)
        messages = list(session.messages)
        metadata = {
            'key': session.key,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'last_consolidated': session.last_consolidated,
            **dict(session.metadata or {}),
        }
        compactions = list(session.compactions)
        usage_summary = get_session_summary(path)
        return {
            'path': path,
            'metadata': metadata,
            'messages': messages,
            'compactions': compactions,
            'usage_summary': usage_summary,
            'key': session.key,
            'channel_id': channel_id,
        }

    metadata, messages, compactions = _parse_session_with_compactions(path)
    usage_summary = get_session_summary(path)
    return {
        "path": path,
        "metadata": metadata,
        "messages": messages,
        "compactions": compactions,
        "usage_summary": usage_summary,
        "key": _session_public_key(path, metadata),
        "channel_id": _session_channel_id(path, metadata),
    }


def _compaction_state_for_message_index(
    compaction_events: list[dict[str, Any]],
    user_message_index: int,
) -> dict[str, Any] | None:
    candidates = []
    for event in compaction_events:
        post_context = event.get("post_context")
        if not isinstance(post_context, dict):
            continue
        first_kept = _coerce_int(post_context.get("first_kept_index"), -1)
        if 0 <= first_kept <= user_message_index:
            candidates.append(event)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda event: _coerce_int(
            (event.get("post_context") or {}).get("first_kept_index"),
            -1,
        ),
    )


def _build_context_turn_payload(
    turn: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    first_kept_index: int,
    messages_in_window: int,
    total_session_messages: int,
) -> dict[str, Any]:
    payload = dict(turn)
    payload["first_kept_index"] = first_kept_index
    payload["messages_in_window"] = messages_in_window
    payload["session_message_count"] = total_session_messages
    payload["system_prompt_changed_since_previous"] = (
        previous_turn is not None
        and previous_turn.get("system_prompt_hash") != turn.get("system_prompt_hash")
    )
    payload["previous_system_prompt_hash"] = (
        previous_turn.get("system_prompt_hash") if previous_turn else None
    )
    return payload


def _build_live_context_response(
    *,
    agent_name: str,
    profile: dict[str, Any],
    requested_turn_index: int | None = None,
) -> dict[str, Any]:
    session_bundle = _load_primary_session_bundle(agent_name, profile)
    if session_bundle is None:
        raise HTTPException(status_code=404, detail=f"No primary session found for '{agent_name}'")

    session_path = session_bundle["path"]
    metadata = session_bundle["metadata"]
    messages = session_bundle["messages"]
    usage_summary = session_bundle["usage_summary"] or {}
    turns = load_context_log(session_path)
    compaction_events = agent_session_compaction_sync(session_path)

    if requested_turn_index is None:
        selected_turn = turns[-1] if turns else None
    else:
        selected_turn = next(
            (entry for entry in turns if _coerce_int(entry.get("turn_index"), -1) == requested_turn_index),
            None,
        )
        if selected_turn is None:
            raise HTTPException(
                status_code=404,
                detail=f"Turn {requested_turn_index} not found for '{agent_name}'",
            )

    latest_turn = turns[-1] if turns else None
    total_session_messages = len(messages)
    model = _resolved_agent_model(profile, _load_config().get("agents", {}).get("defaults", {}))
    context_window = _context_window_for_model(model, usage_summary)
    current_tokens = _current_context_tokens(metadata, usage_summary, context_window)
    utilization_pct = round(current_tokens / context_window * 100, 1) if context_window else 0.0

    fallback_last_consolidated = _coerce_int(metadata.get("last_consolidated"), 0)
    first_kept_index = fallback_last_consolidated
    last_consolidated = fallback_last_consolidated
    visible_messages = messages
    turn_payload = None

    if selected_turn is not None:
        selected_turn_index = _coerce_int(selected_turn.get("turn_index"), 0)
        previous_turn = next(
            (
                entry for entry in turns
                if _coerce_int(entry.get("turn_index"), -1) == selected_turn_index - 1
            ),
            None,
        )
        user_message_index = _coerce_int(selected_turn.get("user_message_index"), len(messages) - 1)
        compaction_state = _compaction_state_for_message_index(compaction_events, user_message_index)
        if compaction_state:
            post_context = compaction_state.get("post_context") or {}
            first_kept_index = _coerce_int(post_context.get("first_kept_index"), first_kept_index)
            last_consolidated = _coerce_int(
                post_context.get("new_last_consolidated"),
                last_consolidated,
            )
        else:
            first_kept_index = 0 if requested_turn_index is not None else fallback_last_consolidated
            last_consolidated = min(user_message_index, fallback_last_consolidated)

        end_index = total_session_messages if requested_turn_index is None else min(total_session_messages, user_message_index + 1)
        start_index = max(0, min(first_kept_index, end_index))
        visible_messages = messages[start_index:end_index]
        turn_payload = _build_context_turn_payload(
            selected_turn,
            previous_turn,
            start_index,
            len(visible_messages),
            total_session_messages,
        )
    else:
        start_index = max(0, first_kept_index)
        visible_messages = messages[start_index:]

    response = {
        "agent": agent_name,
        "session_key": session_bundle["key"],
        "session_file": session_path.name,
        "context_window": context_window,
        "current_tokens": current_tokens,
        "utilization_pct": utilization_pct,
        "total_messages": total_session_messages,
        "first_kept_index": max(0, first_kept_index),
        "messages_in_window": len(visible_messages),
        "last_consolidated": max(0, last_consolidated),
        "turns_available": len(turns),
        "messages": visible_messages,
        "turn": turn_payload,
        "latest_turn": (
            turn_payload if requested_turn_index is None else (
                _build_context_turn_payload(
                    latest_turn,
                    turns[-2] if len(turns) > 1 else None,
                    max(0, first_kept_index),
                    len(messages[max(0, first_kept_index):]),
                    total_session_messages,
                ) if latest_turn else None
            )
        ),
    }
    return response


def _recent_activity_entry(agent: str, msg: dict[str, Any]) -> dict[str, Any] | None:
    ts = _message_timestamp(msg)
    if ts is None:
        return None
    role = str(msg.get("role") or msg.get("_type") or "message")
    content = msg.get("content")
    if isinstance(content, list):
        preview = " ".join(
            part.get("text", "")
            if isinstance(part, dict) and part.get("type") == "text"
            else "[image]"
            if isinstance(part, dict) and part.get("type") == "image_url"
            else str(part)
            for part in content
        )
    elif content is None:
        preview = ""
    else:
        preview = str(content)

    return {
        "agent": agent,
        "timestamp": ts.isoformat(),
        "role": role,
        "preview": preview[:240],
    }


def _build_agent_record(
    name: str,
    profile: dict[str, Any],
    defaults: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    agent_dir = workspace / "agents" / name
    sessions = _session_files(agent_dir)
    total_msgs = 0
    for session_path in sessions:
        try:
            _meta, session_messages = _parse_session(session_path)
            total_msgs += len(session_messages)
        except Exception:
            pass

    mem_file = agent_dir / "memory" / "MEMORY.md"
    mem_text = _read_text(mem_file) or ""
    hist_dir = agent_dir / "memory" / "history"
    hist_count = len(list(hist_dir.glob("*.md"))) if hist_dir.exists() else 0
    display_name = _resolved_agent_display_name(name, profile)
    model = _resolved_agent_model(profile, defaults)
    primary_bundle = _load_primary_session_bundle(name, profile)
    primary_key = None
    primary_channel_id = None
    last_active = None
    context_tokens = 0
    context_window = _context_window_for_model(model)
    context_pct = 0.0

    if primary_bundle is not None:
        primary_meta = primary_bundle["metadata"]
        primary_usage = primary_bundle["usage_summary"] or {}
        primary_key = primary_bundle["key"]
        primary_channel_id = primary_bundle["channel_id"]
        last_active = primary_meta.get("updated_at")
        context_window = _context_window_for_model(model, primary_usage)
        context_tokens = _current_context_tokens(primary_meta, primary_usage, context_window)
        context_pct = round(context_tokens / context_window * 100, 1) if context_window else 0.0

    return {
        "name": name,
        "displayName": display_name,
        "model": model,
        "configuredModel": _profile_value(profile, "model", "model"),
        "systemIdentity": (_profile_value(profile, "systemIdentity", "system_identity") or "")[:80],
        "sessionCount": len(sessions),
        "messageCount": total_msgs,
        "memoryChars": len(mem_text),
        "memoryTokens": _estimate_tokens(mem_text),
        "historyFileCount": hist_count,
        "discordChannels": _profile_value(profile, "discordChannels", "discord_channels") or [],
        "primaryChannelId": primary_channel_id,
        "primarySession": primary_key,
        "contextTokens": context_tokens,
        "contextWindow": context_window,
        "contextPct": context_pct,
        "lastActive": last_active,
    }


def agent_session_compaction_sync(session_path: Path) -> list[dict[str, Any]]:
    compaction_path = session_path.with_suffix(".compaction.jsonl")
    if not compaction_path.exists():
        return []

    events: list[dict[str, Any]] = []
    with open(compaction_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    events.sort(key=lambda event: event.get("timestamp", ""))
    return events


# ---------------------------------------------------------------------------
# LanceDB helpers (lazy init)
# ---------------------------------------------------------------------------

_lance_db = None
_lance_store = None


async def _get_lance_db():
    global _lance_db
    if _lance_db is not None:
        return _lance_db
    try:
        import lancedb
        cfg = _load_config()
        db_path = cfg.get("memoryGraph", {}).get("dbPath", "")
        if not db_path:
            return None
        db_path = os.path.expanduser(db_path)
        _lance_db = await lancedb.connect_async(db_path)
        return _lance_db
    except Exception:
        return None


async def _get_store():
    global _lance_store
    if _lance_store is not None:
        return _lance_store
    try:
        from agent_memory import LiteLLMEmbedding, MemoryGraphStore

        cfg = _load_config()
        mg = cfg.get("memoryGraph", {})
        db_path = os.path.expanduser(mg.get("dbPath", ""))
        if not db_path:
            return None
        emb_cfg = mg.get("embedding", {})
        api_key = emb_cfg.get("apiKey", "")
        model = emb_cfg.get("model", "openai/text-embedding-3-small")
        dims = emb_cfg.get("dimensions", 512)
        embedding = LiteLLMEmbedding(model=model, dimensions=dims, api_key=api_key)
        store = MemoryGraphStore(db_path=db_path, embedding=embedding)
        await store.initialize()
        _lance_store = store
        return store
    except Exception:
        return None


async def _lance_query(table_name: str, where: str | None = None,
                       limit: int = 50, offset: int = 0,
                       columns: list[str] | None = None) -> list[dict]:
    db = await _get_lance_db()
    if db is None:
        return []
    try:
        table = await db.open_table(table_name)
        q = table.query()
        if where:
            q = q.where(where)
        if columns:
            q = q.select(columns)
        q = q.offset(offset).limit(limit)
        rows = await q.to_list()
        # Convert pyarrow/lancedb rows to plain dicts
        result = []
        for r in rows:
            d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
            # Strip vector field for JSON serialization
            d.pop("vector", None)
            # Convert numpy/arrow types
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()
            result.append(d)
        return result
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "Dashboard UI not found")
    return HTMLResponse(index.read_text())


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@app.get("/api/agents")
async def list_agents():
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    ws = _workspace()
    defaults = cfg.get("agents", {}).get("defaults", {})
    agents = []
    for name, profile in profiles.items():
        agents.append(_build_agent_record(name, profile, defaults, ws))
    return {"agents": agents}


@app.get("/api/agents/{name}/context/live")
async def agent_context_live(name: str):
    cfg = _load_config()
    profile = cfg.get("agents", {}).get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return _build_live_context_response(agent_name=name, profile=profile)


@app.get("/api/agents/{name}/context/turn/{turn_index}")
async def agent_context_turn(name: str, turn_index: int):
    cfg = _load_config()
    profile = cfg.get("agents", {}).get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return _build_live_context_response(
        agent_name=name,
        profile=profile,
        requested_turn_index=turn_index,
    )


@app.get("/api/agents/{name}/extractions")
async def agent_extractions(name: str):
    cfg = _load_config()
    profile = cfg.get("agents", {}).get("profiles", {}).get(name)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    session_bundle = _load_primary_session_bundle(name, profile)
    if session_bundle is None:
        return {"events": [], "count": 0}
    events = load_extraction_log(session_bundle["path"])
    events.sort(key=lambda event: event.get("timestamp", ""), reverse=True)
    return {"events": events, "count": len(events)}


@app.get("/api/overview")
async def overview():
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    defaults = cfg.get("agents", {}).get("defaults", {})
    ws = _workspace()
    agents = [
        _build_agent_record(name, profile, defaults, ws)
        for name, profile in profiles.items()
    ]

    now = datetime.now(timezone.utc)
    recent_cutoff = now.timestamp() - 86400
    total_messages_today = 0
    recent_activity: list[dict[str, Any]] = []
    for name, profile in profiles.items():
        session_bundle = _load_primary_session_bundle(name, profile)
        if session_bundle is None:
            continue
        messages = session_bundle["messages"]
        total_messages_today += sum(
            1
            for msg in messages
            if (
                (ts := _message_timestamp(msg)) is not None
                and ts.timestamp() >= recent_cutoff
            )
        )
        for msg in messages[-20:]:
            activity = _recent_activity_entry(name, msg)
            if activity is not None:
                recent_activity.append(activity)

    recent_activity.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    disk = shutil.disk_usage(ws)
    graph = await graph_stats()
    return {
        "agent_count": len(agents),
        "total_messages_today": total_messages_today,
        "max_context_pct": round(max((agent.get("contextPct", 0) for agent in agents), default=0.0), 1),
        "uptime": _format_uptime(_uptime_seconds()),
        "uptime_seconds": _uptime_seconds(),
        "started_at": PROCESS_STARTED_AT.isoformat(),
        "nanobot_version": _nanobot_version(),
        "python_version": sys.version,
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        },
        "graph": graph,
        "recent_activity": recent_activity[:12],
    }


@app.get("/api/version")
async def api_version():
    cfg = _load_config()
    return {
        "version": nanobot.__version__,
        "api_version": "1",
        "schema_version": cfg.get("schemaVersion") or cfg.get("schema_version") or 1,
        "python": platform.python_version(),
    }


@app.get("/api/skills")
async def skills():
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    workspace = _workspace()
    global_skills_dir = workspace / "skills"
    builtin_skills_dir = _project_root() / "nanobot" / "skills"

    global_skills = [
        _read_skill_info(path, "global")
        for path in sorted(global_skills_dir.glob("*/SKILL.md"))
    ]
    builtin_skills = [
        _read_skill_info(path, "builtin")
        for path in sorted(builtin_skills_dir.glob("*/SKILL.md"))
    ]

    per_agent: dict[str, Any] = {}
    for agent_name in sorted(profiles.keys()):
        local_paths = sorted((workspace / "agents" / agent_name / "skills").glob("*/SKILL.md"))
        local_skills = [
            _read_skill_info(path, "local", agent=agent_name)
            for path in local_paths
        ]
        total = len({skill["name"] for skill in global_skills + builtin_skills + local_skills})
        per_agent[agent_name] = {
            "local": local_skills,
            "total": total,
        }

    return {
        "global": global_skills,
        "builtin": builtin_skills,
        "per_agent": per_agent,
    }


@app.get("/api/channels")
async def channels():
    cfg = _load_config()
    profiles = cfg.get("agents", {}).get("profiles", {})
    discord_cfg = cfg.get("channels", {}).get("discord", {})
    rows = []
    for agent_name, profile in profiles.items():
        display_name = _resolved_agent_display_name(agent_name, profile)
        model = _resolved_agent_model(profile, cfg.get("agents", {}).get("defaults", {}))
        for channel_id in _profile_value(profile, "discordChannels", "discord_channels") or []:
            rows.append({
                "id": str(channel_id),
                "name": None,
                "label": display_name,
                "agent": agent_name,
                "displayName": display_name,
                "model": model,
                "webhook": bool(_profile_value(profile, "discordWebhookUrl", "discord_webhook_url")),
                "status": "live" if discord_cfg.get("enabled") else "disabled",
            })

    return {
        "discord": {
            "enabled": bool(discord_cfg.get("enabled")),
            "guild_id": discord_cfg.get("guildId") or discord_cfg.get("guild_id"),
            "usage_dashboard": _sanitize_config_value(
                "usage_dashboard",
                discord_cfg.get("usageDashboard") or discord_cfg.get("usage_dashboard") or {},
            ),
            "codex_usage": _sanitize_config_value(
                "codex_usage",
                discord_cfg.get("codexUsage") or discord_cfg.get("codex_usage") or {},
            ),
            "system_status": _sanitize_config_value(
                "system_status",
                discord_cfg.get("systemStatus") or discord_cfg.get("system_status") or {},
            ),
            "channels": rows,
        }
    }


@app.get("/api/integrations")
async def integrations():
    cfg = _load_config()
    items = []

    mcp_servers = cfg.get("tools", {}).get("mcpServers", {})
    for server_id, server_cfg in mcp_servers.items():
        item = {
            "id": server_id,
            "name": _friendly_name(server_id),
            "type": "mcp",
            "status": "configured",
            "command": server_cfg.get("command", ""),
            "has_env": bool(server_cfg.get("env")),
        }
        if server_id == "google-drive":
            status, details = _google_drive_status(server_cfg)
            item["status"] = status
            item.update(details)
        items.append(item)

    channels_cfg = cfg.get("channels", {})

    telegram = channels_cfg.get("telegram", {})
    if telegram.get("enabled") or telegram.get("token"):
        items.append({
            "id": "telegram",
            "name": "Telegram",
            "type": "channel",
            "status": "connected" if telegram.get("enabled") else "disabled",
            "bot_token_set": bool(telegram.get("token")),
            "allowed_users": len(telegram.get("allowFrom", telegram.get("allow_from", [])) or []),
        })

    discord_cfg = channels_cfg.get("discord", {})
    if discord_cfg.get("enabled") or discord_cfg.get("token"):
        items.append({
            "id": "discord",
            "name": "Discord",
            "type": "channel",
            "status": "connected" if discord_cfg.get("enabled") else "disabled",
            "guild_id": discord_cfg.get("guildId") or discord_cfg.get("guild_id"),
        })

    providers = cfg.get("providers", {})
    for provider_id, provider_cfg in providers.items():
        # Skip providers without API keys configured (don't show unconfigured defaults)
        if not provider_cfg.get("apiKey"):
            continue
        items.append({
            "id": f"provider-{provider_id}",
            "name": _provider_display_name(provider_id),
            "type": "provider",
            "status": "configured",
            "has_api_key": True,
        })

    claude_creds = Path.home() / ".claude" / ".credentials.json"
    if claude_creds.exists():
        try:
            creds = json.loads(claude_creds.read_text())
            oauth = creds.get("claudeAiOauth", {})
            expires_at = oauth.get("expiresAt", 0)
            status = "connected"
            if expires_at and expires_at < time.time() * 1000:
                status = "expired"
            items.append({
                "id": "provider-anthropic-oauth",
                "name": "Anthropic (OAuth)",
                "type": "provider",
                "status": status,
                "subscription": oauth.get("subscriptionType", "unknown"),
                "scopes": oauth.get("scopes", []),
            })
        except Exception:
            pass

    return {"integrations": items}


@app.post("/api/integrations/{id}/check")
async def check_integration(id: str):
    cfg = _load_config()

    if id == "provider-anthropic-oauth":
        claude_creds = Path.home() / ".claude" / ".credentials.json"
        if not claude_creds.exists():
            return {"id": id, "status": "not_configured", "error": "No OAuth credentials found"}
        try:
            creds = json.loads(claude_creds.read_text())
            oauth = creds.get("claudeAiOauth", {})
            expires_at = oauth.get("expiresAt", 0)
            if expires_at and expires_at < time.time() * 1000:
                return {"id": id, "status": "expired", "error": "OAuth token expired (will auto-refresh)"}
            return {"id": id, "status": "connected", "subscription": oauth.get("subscriptionType")}
        except Exception as exc:
            return {"id": id, "status": "error", "error": str(exc)}

    mcp_servers = cfg.get("tools", {}).get("mcpServers", {})
    if id in mcp_servers:
        server = mcp_servers[id]
        env = server.get("env", {})
        if id == "google-drive":
            status, details = _google_drive_status(server)
            payload = {"id": id, "status": status}
            payload.update(details)
            if status == "connected":
                payload["checked_at"] = datetime.now(timezone.utc).isoformat()
            return payload

        cmd = str(server.get("command", "")).strip()
        missing_env = [key for key, value in env.items() if not value]
        if missing_env:
            return {"id": id, "status": "error", "error": f"Missing env vars: {', '.join(missing_env)}"}

        cmd_name = cmd.split()[0] if cmd else ""
        if cmd_name and not shutil.which(cmd_name):
            return {"id": id, "status": "error", "error": f"Command not found: {cmd_name}"}
        return {"id": id, "status": "configured", "checked_at": datetime.now(timezone.utc).isoformat()}

    return {"id": id, "status": "unknown", "error": "Integration not found"}


@app.get("/api/integrations/{id}/auth")
async def integration_auth(id: str, callback_base: str = ""):
    """Generate OAuth authorization URL for an integration."""
    import secrets
    import urllib.parse

    if not callback_base:
        callback_base = "http://localhost:18790/api"

    if id != "google-drive":
        raise HTTPException(400, f"OAuth not supported for {id}")

    cfg = _load_config()
    mcp = cfg.get("tools", {}).get("mcpServers", {}).get("google-drive", {})
    env = mcp.get("env", {})

    client_id = env.get("GOOGLE_CLIENT_ID", "")
    if not client_id:
        try:
            creds_data = json.loads(env.get("GOOGLE_CREDENTIALS_JSON", "{}"))
            client_id = creds_data.get("client_id", "")
        except Exception:
            pass

    if not client_id:
        raise HTTPException(400, "No Google client_id configured")

    callback_url = f"{callback_base}/integrations/{id}/callback"
    state = secrets.token_urlsafe(32)

    if not hasattr(app, "_oauth_states"):
        app._oauth_states = {}
    app._oauth_states[state] = {"id": id, "created": time.time()}

    cutoff = time.time() - 600
    app._oauth_states = {
        key: value for key, value in app._oauth_states.items() if value["created"] > cutoff
    }

    params = {
        "client_id": client_id,
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": (
            "https://www.googleapis.com/auth/drive "
            "https://www.googleapis.com/auth/spreadsheets "
            "https://www.googleapis.com/auth/documents.readonly"
        ),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }

    auth_url = "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)
    return {"auth_url": auth_url, "callback_url": callback_url, "state": state}


@app.get("/api/integrations/{id}/callback")
async def integration_callback(
    id: str,
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
):
    """Handle OAuth callback from provider."""
    from html import escape

    if error:
        return HTMLResponse(
            f"""<html><body style="font-family:sans-serif;padding:40px;background:#1a1a2e;color:#e0e0e0;">
            <h1>Authorization Failed</h1><p>{escape(error)}</p>
            <p>Close this tab and try again from the dashboard.</p>
        </body></html>"""
        )

    if not code:
        raise HTTPException(400, "No authorization code received")

    if hasattr(app, "_oauth_states") and state:
        if state not in app._oauth_states:
            return HTMLResponse(
                """<html><body style="font-family:sans-serif;padding:40px;background:#1a1a2e;color:#e0e0e0;">
                <h1>Invalid State</h1><p>The authorization request has expired. Please try again.</p>
            </body></html>"""
            )
        del app._oauth_states[state]

    if id != "google-drive":
        raise HTTPException(400, f"OAuth callback not supported for {id}")

    cfg = _load_config()
    mcp = cfg.get("tools", {}).get("mcpServers", {}).get("google-drive", {})
    env = mcp.get("env", {})

    client_id = env.get("GOOGLE_CLIENT_ID", "")
    client_secret = env.get("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        try:
            creds_data = json.loads(env.get("GOOGLE_CREDENTIALS_JSON", "{}"))
            client_id = client_id or creds_data.get("client_id", "")
            client_secret = client_secret or creds_data.get("client_secret", "")
        except Exception:
            pass

    import httpx

    callback_url = str(request.url).split("?", 1)[0]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": callback_url,
                    "grant_type": "authorization_code",
                },
            )
            if resp.status_code != 200:
                try:
                    error_detail = resp.json().get("error_description", resp.text)
                except Exception:
                    error_detail = resp.text
                return HTMLResponse(
                    f"""<html><body style="font-family:sans-serif;padding:40px;background:#1a1a2e;color:#e0e0e0;">
                    <h1>Token Exchange Failed</h1><p>{escape(error_detail)}</p>
                </body></html>"""
                )

            tokens = resp.json()
    except Exception as exc:
        return HTMLResponse(
            f"""<html><body style="font-family:sans-serif;padding:40px;background:#1a1a2e;color:#e0e0e0;">
            <h1>Error</h1><p>{escape(str(exc))}</p>
        </body></html>"""
        )

    new_creds = json.dumps({
        "token": tokens.get("access_token"),
        "refresh_token": tokens.get("refresh_token"),
        "client_id": client_id,
        "client_secret": client_secret,
    })

    creds_dir = _workspace().parent / "credentials"
    creds_dir.mkdir(parents=True, exist_ok=True)
    creds_file = creds_dir / "google-drive.json"
    creds_file.write_text(new_creds)

    return HTMLResponse(
        """<html><body style="font-family:sans-serif;padding:40px;background:#1a1a2e;color:#e0e0e0;text-align:center;">
        <h1 style="color:#4ade80;">Google Drive Connected</h1>
        <p style="color:#a0a0a0;">New credentials have been saved.</p>
        <p style="color:#a0a0a0;">You can close this tab and return to the dashboard.</p>
        <script>setTimeout(function(){ window.close(); }, 3000);</script>
    </body></html>"""
    )


@app.get("/api/logs")
async def get_logs(
    lines: int = Query(100, ge=1, le=500),
    level: str = Query("all"),
):
    log_path = Path.home() / ".nanobot" / "logs" / "gateway.log"
    if not log_path.exists():
        return {"entries": [], "count": 0, "path": str(log_path)}

    requested_level = level.lower()
    tail = deque(maxlen=lines * 5 if requested_level != "all" else lines)
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            tail.append(raw_line.rstrip("\n"))

    entries = []
    for raw_line in reversed(tail):
        match = _LOG_LINE_RE.match(raw_line)
        parsed = {
            "raw": raw_line,
            "timestamp": match.group("timestamp") if match else None,
            "level": (match.group("level") if match else "UNKNOWN"),
            "source": (match.group("source").strip() if match else None),
            "message": (match.group("message") if match else raw_line),
        }
        if requested_level != "all" and parsed["level"].lower() != requested_level:
            continue
        entries.append(parsed)
        if len(entries) >= lines:
            break

    return {
        "entries": list(reversed(entries)),
        "count": len(entries),
        "path": str(log_path),
    }


@app.get("/api/cron")
async def cron_jobs():
    cron = CronService(get_data_dir() / 'cron' / 'jobs')
    jobs = cron.list_jobs(include_disabled=True)
    rows = []
    for job in jobs:
        rows.append({
            'id': job.id,
            'name': job.name,
            'enabled': job.enabled,
            'schedule_type': job.schedule.kind,
            'cron_expr': job.schedule.expr,
            'every_seconds': (job.schedule.every_ms // 1000) if job.schedule.every_ms is not None else None,
            'at_ms': job.schedule.at_ms,
            'agent_id': job.payload.agent_id,
            'message': job.payload.message,
            'channel': job.payload.channel,
            'chat_id': job.payload.to,
            'deliver': job.payload.deliver,
            'origin_session_key': job.payload.origin_session_key,
            'next_run_at_ms': job.state.next_run_at_ms,
            'last_run_at_ms': job.state.last_run_at_ms,
            'last_status': job.state.last_status,
            'last_error': job.state.last_error,
            'runs': job.state.runs,
            'created_at_ms': job.created_at_ms,
            'updated_at_ms': job.updated_at_ms,
            'timeout_s': job.timeout_s,
            'max_runs': job.max_runs,
        })
    return {'jobs': rows, 'count': len(rows)}


@app.get("/api/config")
async def config():
    return {"config": _sanitize_config_value("config", _load_config())}


class AgentMutationRequest(BaseModel):
    mode: str
    agent_id: str
    source_agent_id: str | None = None
    model: str | None = None
    background_model: str | None = None
    system_identity: str | None = None
    skills: list[str] | None = None
    channel_name: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    copy_history: bool = False
    copy_sessions: bool = False


@app.post("/api/agents")
async def create_or_clone_agent(payload: AgentMutationRequest):
    mode = payload.mode.strip().lower()
    if mode not in {"create", "clone"}:
        raise HTTPException(status_code=400, detail="mode must be 'create' or 'clone'")

    agent_id = payload.agent_id.strip()
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")

    effective = _load_config()
    existing_profiles = effective.get("agents", {}).get("profiles", {})
    if agent_id in existing_profiles:
        raise HTTPException(status_code=409, detail=f"Agent '{agent_id}' already exists")

    base_config, state_store = _base_config_and_store()
    profile_manager = AgentProfileManager(base_config, state_store)
    workspace_root = Path(base_config.agents.defaults.workspace).expanduser()
    guild_id = base_config.channels.discord.guild_id
    if not base_config.channels.discord.enabled or not base_config.channels.discord.token:
        raise HTTPException(status_code=400, detail="Discord is not configured for agent provisioning")
    if not guild_id:
        raise HTTPException(status_code=400, detail="Discord guild_id is not configured")

    default_model = base_config.agents.defaults.model
    channel_name = (payload.channel_name or agent_id).strip()
    display_name = (payload.display_name or agent_id).strip()
    category_id = _discord_channel_category_id()
    discord = _build_dashboard_discord_channel(base_config)

    if mode == "create":
        model_topic = payload.model or default_model or ""
        channel_id = await discord.create_guild_channel(
            guild_id=guild_id,
            name=channel_name,
            topic=model_topic[:1024],
            category_id=category_id,
        )
        if not channel_id:
            raise HTTPException(status_code=502, detail="Failed to create Discord channel")

        webhook_url = await discord.create_channel_webhook(
            channel_id=channel_id,
            name=display_name,
            avatar_url=payload.avatar_url,
        )

        profile = profile_manager.create_profile(
            agent_id,
            model=payload.model,
            background_model=payload.background_model,
            skills=payload.skills,
            system_identity=payload.system_identity,
            discord_channels=[channel_id],
            display_name=display_name,
            avatar_url=payload.avatar_url,
            discord_webhook_url=webhook_url,
        )
        init_agent_workspace(workspace_root, agent_id, payload.system_identity)
        resolved = profile.resolve(base_config.agents.defaults)
        return {
            "status": "created",
            "restart_required": True,
            "agent": {
                "agent_id": agent_id,
                "model": resolved.model,
                "background_model": resolved.background_model,
                "display_name": resolved.display_name,
                "skills": resolved.skills,
                "reasoning_effort": resolved.reasoning_effort,
                "webhook_enabled": webhook_url is not None,
                "discord_channels": [channel_id],
                "discord_webhook_url": webhook_url,
            },
            "channel": {
                "id": channel_id,
                "name": channel_name,
                "topic": resolved.model,
            },
            "warnings": ["Gateway restart required for the new agent to become active."],
        }

    source_agent_id = (payload.source_agent_id or "").strip()
    if not source_agent_id:
        raise HTTPException(status_code=400, detail="source_agent_id is required for clone")
    source_profile = existing_profiles.get(source_agent_id)
    if not isinstance(source_profile, dict):
        raise HTTPException(status_code=404, detail=f"Source agent '{source_agent_id}' not found")

    clone_model = payload.model if payload.model is not None else _profile_value(source_profile, 'model', 'model') or default_model or ''
    channel_id = await discord.create_guild_channel(
        guild_id=guild_id,
        name=channel_name,
        topic=clone_model[:1024],
        category_id=category_id,
    )
    if not channel_id:
        raise HTTPException(status_code=502, detail="Failed to create Discord channel")

    webhook_url = await discord.create_channel_webhook(
        channel_id=channel_id,
        name=display_name,
        avatar_url=payload.avatar_url,
    )

    profile = profile_manager.create_profile(
        agent_id,
        model=clone_model,
        background_model=payload.background_model if payload.background_model is not None else _profile_value(source_profile, 'backgroundModel', 'background_model'),
        context_window=_profile_value(source_profile, 'contextWindow', 'context_window'),
        background_context_window=_profile_value(source_profile, 'backgroundContextWindow', 'background_context_window'),
        session_store=_profile_value(source_profile, 'sessionStore', 'session_store'),
        max_tokens=_profile_value(source_profile, 'maxTokens', 'max_tokens'),
        temperature=_profile_value(source_profile, 'temperature', 'temperature'),
        max_tool_iterations=_profile_value(source_profile, 'maxToolIterations', 'max_tool_iterations'),
        reasoning_effort=_profile_value(source_profile, 'reasoningEffort', 'reasoning_effort'),
        skills=payload.skills if payload.skills is not None else list(_profile_value(source_profile, 'skills', 'skills') or []),
        system_identity=payload.system_identity if payload.system_identity is not None else _profile_value(source_profile, 'systemIdentity', 'system_identity'),
        discord_channels=[channel_id],
        display_name=display_name,
        avatar_url=payload.avatar_url if payload.avatar_url is not None else _profile_value(source_profile, 'avatarUrl', 'avatar_url'),
        discord_webhook_url=webhook_url,
    )
    init_agent_workspace(workspace_root, agent_id, payload.system_identity or _profile_value(source_profile, 'systemIdentity', 'system_identity'))
    _copy_dashboard_workspace(
        workspace_root=workspace_root,
        source_agent_id=source_agent_id,
        target_agent_id=agent_id,
        copy_history=payload.copy_history,
        copy_sessions=payload.copy_sessions,
    )
    resolved = profile.resolve(base_config.agents.defaults)
    return {
        "status": "cloned",
        "restart_required": True,
        "agent": {
            "agent_id": agent_id,
            "source_agent_id": source_agent_id,
            "model": resolved.model,
            "background_model": resolved.background_model,
            "display_name": resolved.display_name,
            "skills": resolved.skills,
            "reasoning_effort": resolved.reasoning_effort,
            "webhook_enabled": webhook_url is not None,
            "discord_channels": [channel_id],
            "discord_webhook_url": webhook_url,
        },
        "channel": {
            "id": channel_id,
            "name": channel_name,
            "topic": resolved.model,
        },
        "copied": {
            "memory": True,
            "skills": True,
            "history": payload.copy_history,
            "sessions": payload.copy_sessions,
        },
        "warnings": ["Gateway restart required for the cloned agent to become active."],
    }


@app.post("/api/restart")
async def restart_gateway():
    """Gracefully restart the gateway worker process."""
    import signal as _signal

    try:
        from nanobot.daemon import GatewayDaemon

        if GatewayDaemon.send_signal(_signal.SIGUSR1):
            return {"status": "ok", "message": "Restart signal sent to gateway worker."}
        else:
            raise HTTPException(
                status_code=503,
                detail="Gateway process not running. Cannot send restart signal.",
            )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Daemon module not available.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/{name}/memory")
async def agent_memory(name: str):
    adir = _agent_dir(name)
    mem_file = adir / "memory" / "MEMORY.md"
    content = _read_text(mem_file) or ""
    chars = len(content)
    tokens = _estimate_tokens(content)
    budget_char_limit = 16000
    budget_token_limit = 4000
    return {
        "content": content,
        "chars": chars,
        "tokens": tokens,
        "budgetCharPercent": round(chars / budget_char_limit * 100, 1),
        "budgetTokenPercent": round(tokens / budget_token_limit * 100, 1),
    }


@app.get("/api/agents/{name}/history")
async def agent_history(name: str):
    adir = _agent_dir(name)
    hist_dir = adir / "memory" / "history"
    if not hist_dir.exists():
        return {"files": []}
    files = []
    for f in sorted(hist_dir.glob("*.md"), reverse=True):
        files.append({
            "date": f.stem,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"files": files}


@app.get("/api/agents/{name}/history/{date}")
async def agent_history_date(name: str, date: str):
    adir = _agent_dir(name)
    f = adir / "memory" / "history" / f"{date}.md"
    if not f.exists():
        raise HTTPException(404, f"History file for {date} not found")
    return {"date": date, "content": f.read_text(errors="replace")}


@app.get("/api/agents/{name}/identity")
async def agent_identity(name: str):
    adir = _agent_dir(name)
    ws = _workspace()
    return {
        "identity": _read_text(adir / "IDENTITY.md"),
        "soul": _read_text(ws / "SOUL.md") or _read_text(adir / "SOUL.md"),
        "user": _read_text(ws / "USER.md") or _read_text(adir / "USER.md"),
    }


@app.get("/api/agents/{name}/sessions")
async def agent_sessions(name: str):
    adir = _agent_dir(name)
    cfg = _load_config()
    profile = cfg.get("agents", {}).get("profiles", {}).get(name, {})
    primary_path = _primary_session_path(name, profile) if isinstance(profile, dict) else None
    sessions = []
    for sp in _session_files(adir):
        try:
            meta, msgs = _parse_session(sp)
            sessions.append({
                "key": meta.get("key", sp.stem),
                "file": sp.name,
                "messageCount": len(msgs),
                "lastConsolidated": meta.get("last_consolidated", 0),
                "createdAt": meta.get("created_at"),
                "updatedAt": meta.get("updated_at"),
                "size": sp.stat().st_size,
                "isPrimary": primary_path == sp,
                "isCron": sp.stem.startswith("cron_") or str(meta.get("key", "")).startswith("cron:"),
            })
        except Exception:
            sessions.append({
                "key": sp.stem,
                "file": sp.name,
                "messageCount": 0,
                "error": "Failed to parse",
                "isPrimary": primary_path == sp,
                "isCron": sp.stem.startswith("cron_"),
            })
    return {"sessions": sessions}


def _resolve_session_path(name: str, key: str) -> Path:
    """Resolve a session key to its JSONL file path, or raise 404."""
    adir = _agent_dir(name)
    sd = adir / "sessions"
    candidates = [
        sd / f"{key}.jsonl",
        sd / f"{_safe_filename(key)}.jsonl",
        sd / key,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try matching by key in metadata
    for sp in _session_files(adir):
        meta, _ = _parse_session(sp)
        if meta.get("key") == key:
            return sp
    raise HTTPException(404, f"Session '{key}' not found")


def _parse_session_paginated(
    path: Path,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> tuple[dict, list[dict], int, list[dict[str, Any]]]:
    """Parse a session JSONL file with optional tail-based pagination.

    When *offset* and *limit* are provided, only messages in the range
    ``[offset, offset+limit)`` are returned (0-indexed from the start of the
    file).  The caller typically computes *offset* from the end so that the
    most recent messages are fetched first.

    Compaction entries are excluded from the message stream and total count.
    They are collected separately and returned as a list so the frontend can
    render them at their logical ``first_kept_index`` position.

    Returns ``(metadata, messages_slice, total_message_count, compactions)``.
    """
    metadata: dict = {}
    total = 0
    messages: list[dict] = []
    compactions: list[dict[str, Any]] = []
    want_all = offset is None and limit is None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("_type") == "metadata":
                metadata = data
                continue
            if data.get("_type") == "compaction" and isinstance(data, dict):
                compactions.append(data)
                continue
            if want_all:
                messages.append(data)
            else:
                if offset <= total < offset + limit:  # type: ignore[operator]
                    messages.append(data)
            total += 1

    return metadata, messages, total, compactions


@app.get("/api/agents/{name}/sessions/{key:path}")
async def agent_session_detail(
    name: str,
    key: str,
    offset: int | None = Query(None, ge=0),
    limit: int | None = Query(None, ge=1, le=1000),
):
    # Intercept context log requests (key ends with /context)
    if key.endswith("/context"):
        actual_key = key[:-len("/context")]
        return await agent_session_context(name, actual_key)

    # Intercept compaction log requests (key ends with /compaction)
    if key.endswith("/compaction"):
        actual_key = key[:-len("/compaction")]
        return await agent_session_compaction(name, actual_key)

    path = _resolve_session_path(name, key)

    if offset is not None and limit is not None:
        meta, msgs, total, compactions = _parse_session_paginated(
            path,
            offset=offset,
            limit=limit,
        )
        return {
            "metadata": meta,
            "messages": msgs,
            "messageCount": total,
            "compactions": compactions,
            "offset": offset,
            "limit": limit,
            "hasMore": offset > 0,
        }

    # Legacy: return everything (used by other callers)
    meta, msgs = _parse_session(path)
    return {
        "metadata": meta,
        "messages": msgs,
        "messageCount": len(msgs),
    }


async def agent_session_compaction(name: str, key: str):
    """Return the compaction event log for a session."""
    path = _resolve_session_path(name, key)
    events = agent_session_compaction_sync(path)
    return {"events": events, "count": len(events)}


@app.get("/api/agents/{name}/compaction")
async def agent_compaction_all(name: str):
    """Return all compaction events across all sessions for an agent."""
    adir = _agent_dir(name)
    all_events: list[dict] = []

    for sp in _session_files(adir):
        session_key = sp.stem
        for event in agent_session_compaction_sync(sp):
            event = dict(event)
            event["_session_file"] = sp.name
            event["_session_key"] = session_key
            all_events.append(event)

    all_events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return {"events": all_events, "count": len(all_events)}


async def agent_session_context(name: str, key: str):
    """Return the per-turn context log for a session."""
    path = _resolve_session_path(name, key)
    entries = load_context_log(path)
    return {"entries": entries, "count": len(entries)}


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

@app.get("/api/graph/memories")
async def graph_memories(
    peer_key: str | None = Query(None),
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at_ms"),
    forgotten: int = Query(0),
):
    where_parts = [f"forgotten = {forgotten}"]
    if peer_key:
        where_parts.append(f"peer_key = '{peer_key}'")
    if type:
        where_parts.append(f"memory_type = '{type}'")
    where = " AND ".join(where_parts)

    cols = [
        "id", "content", "memory_type", "importance", "source",
        "source_session", "peer_key", "entities", "created_at_ms",
        "updated_at_ms", "access_count", "forgotten", "context_tag",
    ]
    rows = await _lance_query("memories", where=where, limit=limit, offset=offset, columns=cols)
    return {"memories": rows, "limit": limit, "offset": offset}


@app.get("/api/graph/memories/{memory_id}")
async def graph_memory_detail(memory_id: str):
    db = await _get_lance_db()
    if db is None:
        raise HTTPException(503, "Graph database not available")
    try:
        table = await db.open_table("memories")
        rows = await table.query().where(f"id = '{memory_id}'").limit(1).to_list()
        if not rows:
            raise HTTPException(404, "Memory not found")
        d = dict(rows[0]) if isinstance(rows[0], dict) else {k: rows[0][k] for k in rows[0]}
        d.pop("vector", None)
        for k, v in list(d.items()):
            if hasattr(v, "item"):
                d[k] = v.item()
            elif hasattr(v, "as_py"):
                d[k] = v.as_py()
        return {"memory": d}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    peer_key: str | None = None
    max_results: int = 10


@app.post("/api/graph/search")
async def graph_search(req: SearchRequest):
    store = await _get_store()
    if store is None:
        raise HTTPException(503, "Memory store not available (agent-memory not installed or not configured)")

    try:
        if req.mode == "vector":
            results = await store.vector_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        elif req.mode == "keyword":
            results = await store.keyword_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        else:
            results = await store.hybrid_search(
                query=req.query, max_results=req.max_results,
                peer_key=req.peer_key,
            )
        # Clean results for JSON
        clean = []
        for r in results:
            d = dict(r)
            d.pop("vector", None)
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()
            clean.append(d)
        return {"results": clean, "mode": req.mode, "query": req.query}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/graph/neighbors/{memory_id}")
async def graph_neighbors(memory_id: str, depth: int = Query(1, ge=1, le=3)):
    db = await _get_lance_db()
    if db is None:
        raise HTTPException(503, "Graph database not available")

    try:
        assoc_table = await db.open_table("associations")
        mem_table = await db.open_table("memories")

        visited_ids: set[str] = {memory_id}
        edges: list[dict] = []
        frontier = [memory_id]

        for _ in range(depth):
            if not frontier:
                break
            next_frontier = []
            for mid in frontier:
                # Get edges where this memory is source or target
                for direction, field in [("outgoing", "source_id"), ("incoming", "target_id")]:
                    rows = await assoc_table.query().where(f"{field} = '{mid}'").limit(100).to_list()
                    for r in rows:
                        d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
                        for k, v in list(d.items()):
                            if hasattr(v, "item"):
                                d[k] = v.item()
                            elif hasattr(v, "as_py"):
                                d[k] = v.as_py()
                        edges.append(d)
                        other = d.get("target_id" if field == "source_id" else "source_id")
                        if other and other not in visited_ids:
                            visited_ids.add(other)
                            next_frontier.append(other)
            frontier = next_frontier

        # Fetch memory content for all visited nodes
        nodes = []
        for nid in visited_ids:
            rows = await mem_table.query().where(f"id = '{nid}'").select(
                ["id", "content", "memory_type", "importance", "peer_key", "entities"]
            ).limit(1).to_list()
            if rows:
                d = dict(rows[0]) if isinstance(rows[0], dict) else {k: rows[0][k] for k in rows[0]}
                d.pop("vector", None)
                for k, v in list(d.items()):
                    if hasattr(v, "item"):
                        d[k] = v.item()
                    elif hasattr(v, "as_py"):
                        d[k] = v.as_py()
                nodes.append(d)

        return {"nodes": nodes, "edges": edges, "rootId": memory_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/graph/stats")
async def graph_stats():
    db = await _get_lance_db()
    if db is None:
        return {"available": False}

    try:
        mem_table = await db.open_table("memories")
        assoc_table = await db.open_table("associations")

        # Get all memories for stats computation
        all_mems = await mem_table.query().select(
            ["id", "memory_type", "importance", "peer_key", "forgotten", "created_at_ms", "access_count"]
        ).limit(10000).to_list()

        total = len(all_mems)
        forgotten_count = 0
        type_dist: dict[str, int] = {}
        peer_dist: dict[str, int] = {}
        importance_sum = 0.0
        creation_dates: dict[str, int] = {}

        for r in all_mems:
            d = dict(r) if isinstance(r, dict) else {k: r[k] for k in r}
            for k, v in list(d.items()):
                if hasattr(v, "item"):
                    d[k] = v.item()
                elif hasattr(v, "as_py"):
                    d[k] = v.as_py()

            if d.get("forgotten", 0):
                forgotten_count += 1

            mt = d.get("memory_type", "unknown")
            type_dist[mt] = type_dist.get(mt, 0) + 1

            pk = d.get("peer_key") or "none"
            peer_dist[pk] = peer_dist.get(pk, 0) + 1

            imp = d.get("importance", 0)
            if isinstance(imp, (int, float)):
                importance_sum += imp

            ts = d.get("created_at_ms", 0)
            if ts:
                date_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                creation_dates[date_str] = creation_dates.get(date_str, 0) + 1

        # Edge count
        all_edges = await assoc_table.query().select(["id"]).limit(10000).to_list()
        edge_count = len(all_edges)

        return {
            "available": True,
            "totalMemories": total,
            "totalEdges": edge_count,
            "forgottenCount": forgotten_count,
            "avgImportance": round(importance_sum / max(total, 1), 3),
            "typeDistribution": dict(sorted(type_dist.items(), key=lambda x: -x[1])),
            "peerKeyDistribution": dict(sorted(peer_dist.items(), key=lambda x: -x[1])),
            "creationTimeline": dict(sorted(creation_dates.items())),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}
