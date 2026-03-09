"""Shared provider construction helpers."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from nanobot.config.schema import Config

console = Console()


def _setup_secondary_env_vars(config: Config) -> None:
    """Set env vars for secondary providers so auxiliary model calls can resolve."""
    mapping = [
        ("openai", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ]
    for name, env_key in mapping:
        provider = getattr(config.providers, name, None)
        if provider and hasattr(provider, "api_key") and provider.api_key:
            os.environ.setdefault(env_key, provider.api_key)


def _is_claude_model_name(model_name: str) -> bool:
    model_lower = model_name.lower()
    if model_lower.startswith(("claude-code/", "claude_code/")):
        return True
    if "/" in model_lower:
        prefix = model_lower.split("/", 1)[0]
        if prefix in {"anthropic", "claude", "anthropic-direct", "anthropic_direct"}:
            return True
    return "claude" in model_lower


def build_provider(
    config: Config,
    *,
    model: str | None = None,
    workspace: Path | None = None,
):
    """Create a provider instance for the requested model."""
    from nanobot.providers.anthropic_auth import get_oauth_token, is_oauth_token
    from nanobot.providers.claude_code_provider import ClaudeCodeProvider
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.registry import find_by_name

    model = model or config.agents.defaults.model
    workspace = workspace or config.workspace_path
    provider_name = config.get_provider_name(model)
    provider_cfg = config.get_provider(model)
    api_key = provider_cfg.api_key if provider_cfg and hasattr(provider_cfg, "api_key") else None

    if provider_name == "claude_code" or model.startswith(("claude-code/", "claude_code/")):
        return ClaudeCodeProvider(default_model=model, workspace=workspace)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    if provider_name == "custom":
        return CustomProvider(
            api_key=provider_cfg.api_key if provider_cfg else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    if not model.startswith("bedrock/") and api_key:
        return LiteLLMProvider(
            api_key=api_key,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=provider_cfg.extra_headers if provider_cfg and hasattr(provider_cfg, "extra_headers") else None,
            provider_name=provider_name,
        )

    direct_cfg = config.providers.anthropic_direct
    wants_anthropic_direct = provider_name == "anthropic_direct" or _is_claude_model_name(model)
    if direct_cfg.enabled and wants_anthropic_direct:
        oauth_token = get_oauth_token()
        if oauth_token and is_oauth_token(oauth_token):
            from nanobot.providers.anthropic_direct_provider import AnthropicDirectProvider

            _setup_secondary_env_vars(config)
            return AnthropicDirectProvider(oauth_token=oauth_token, default_model=model)
        if provider_name == "anthropic_direct" or _is_claude_model_name(model):
            console.print("[red]Error: No Anthropic OAuth token configured.[/red]")
            console.print("Run [cyan]claude login[/cyan] or set [cyan]CLAUDE_CODE_OAUTH_TOKEN[/cyan]")
            raise typer.Exit(1)

    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not api_key and not (spec and spec.is_oauth):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.nanobot/config.json under providers section")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=api_key,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=provider_cfg.extra_headers if provider_cfg and hasattr(provider_cfg, "extra_headers") else None,
        provider_name=provider_name,
    )


class ProviderFactory:
    """Cache provider instances by resolved model and workspace."""

    def __init__(self, config: Config):
        self.config = config
        self._cache: dict[tuple[str, str], object] = {}

    def for_model(self, model: str | None = None, *, workspace: Path | None = None):
        resolved_model = model or self.config.agents.defaults.model
        resolved_workspace = (workspace or self.config.workspace_path).expanduser().resolve()
        key = (resolved_model, str(resolved_workspace))
        provider = self._cache.get(key)
        if provider is None:
            provider = build_provider(
                self.config,
                model=resolved_model,
                workspace=resolved_workspace,
            )
            self._cache[key] = provider
        return provider
