"""Configuration loading utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from nanobot.config.schema import Config, SCHEMA_VERSION


def get_config_dir() -> Path:
    """Get the nanobot config directory."""
    return Path.home() / ".nanobot"


def get_config_path() -> Path:
    """Get the preferred base configuration file path."""
    config_dir = get_config_dir()
    base_path = config_dir / "config.base.json"
    if base_path.exists():
        return base_path
    return config_dir / "config.json"


def get_state_path(config_path: Path | None = None) -> Path:
    """Get the runtime state file path."""
    base_path = config_path or get_config_path()
    return base_path.parent / "state.json"


def get_data_dir() -> Path:
    """Get the nanobot data directory."""
    from nanobot.utils.helpers import get_data_path
    return get_data_path()


def load_base_config_data(config_path: Path | None = None) -> dict[str, Any]:
    """Load raw base config JSON with migrations applied."""
    path = config_path or get_config_path()
    data = _load_json_file(path)
    if data is None:
        return {"schemaVersion": SCHEMA_VERSION}
    return _migrate_config(data)


def load_base_config(config_path: Path | None = None) -> Config:
    """Load only the base configuration file."""
    return Config.model_validate(load_base_config_data(config_path))


def load_config_data(
    config_path: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Load the effective config by overlaying runtime state onto base config."""
    from nanobot.config.state import StateStore

    base_path = config_path or get_config_path()
    base_data = load_base_config_data(base_path)
    store = StateStore(state_path or get_state_path(base_path))
    return store.overlay_config(base_data)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    try:
        data = load_config_data(path)
        return Config.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to load config from {path}: {e}")
        print("Using default configuration.")
        return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)
    data["schemaVersion"] = SCHEMA_VERSION
    _atomic_write_json(path, data)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    data = dict(data)
    data.setdefault("schemaVersion", SCHEMA_VERSION)

    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data


def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Load JSON from disk, returning None when the file is absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via temp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
