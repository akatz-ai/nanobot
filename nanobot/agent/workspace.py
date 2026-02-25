"""Workspace utilities for multi-agent directory isolation."""

from pathlib import Path

from loguru import logger


def get_agent_workspace(base_workspace: Path, agent_id: str) -> Path:
    """Get the workspace directory for an agent.

    For the "default" agent, returns the base workspace (backward compat).
    For named agents, returns base/agents/{agent_id}/.
    """
    if agent_id == "default":
        return base_workspace
    return base_workspace / "agents" / agent_id


def get_base_workspace(workspace: Path) -> Path:
    """Resolve the root workspace path from any agent workspace path.

    Named agent workspaces are expected at base/agents/{agent_id}. For those
    paths, this returns base. For all other paths (including default agent),
    this returns workspace unchanged.
    """
    if workspace.parent.name == "agents":
        return workspace.parent.parent
    return workspace


def get_global_skills_dir(workspace: Path) -> Path:
    """Get the shared/global skills directory for an agent workspace."""
    return get_base_workspace(workspace) / "skills"


def init_agent_workspace(
    base_workspace: Path,
    agent_id: str,
    system_identity: str | None = None,
) -> Path:
    """Initialize an agent's workspace directory structure.

    Creates: memory/, sessions/, skills/ directories and optionally IDENTITY.md.

    Returns the workspace path.
    """
    workspace = get_agent_workspace(base_workspace, agent_id)

    for subdir in ("memory", "sessions", "skills"):
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    identity_file = workspace / "IDENTITY.md"
    if system_identity and not identity_file.exists():
        identity_file.write_text(system_identity, encoding="utf-8")

    logger.info("Initialized workspace for agent '{}' at {}", agent_id, workspace)
    return workspace
