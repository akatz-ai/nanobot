"""CLI commands for nanobot."""

import asyncio
import os
import signal
from pathlib import Path
import select
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from nanobot import __version__, __logo__
from nanobot.config.schema import Config

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".nanobot" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard():
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config
    from nanobot.config.schema import Config
    from nanobot.utils.helpers import get_workspace_path
    
    config_path = get_config_path()
    
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")
    
    # Create workspace
    workspace = get_workspace_path()
    
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")
    
    # Create default bootstrap files
    _create_workspace_templates(workspace)
    
    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.nanobot/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")




def _create_workspace_templates(workspace: Path):
    """Create default workspace template files from bundled templates."""
    from importlib.resources import files as pkg_files

    templates_dir = pkg_files("nanobot") / "templates"

    for item in templates_dir.iterdir():
        if not item.name.endswith(".md"):
            continue
        dest = workspace / item.name
        if not dest.exists():
            dest.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(f"  [dim]Created {item.name}[/dim]")

    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)

    memory_template = templates_dir / "memory" / "MEMORY.md"
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text(memory_template.read_text(encoding="utf-8"), encoding="utf-8")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")

    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        history_file.write_text("", encoding="utf-8")
        console.print("  [dim]Created memory/HISTORY.md[/dim]")

    history_daily_dir = memory_dir / "history"
    history_daily_dir.mkdir(exist_ok=True)

    (workspace / "skills").mkdir(exist_ok=True)


def _setup_secondary_env_vars(config: Config) -> None:
    """Set env vars for secondary providers (OpenAI, Groq, etc.) so litellm can find them."""
    import os
    mapping = [
        ("openai", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ]
    for name, env_key in mapping:
        p = getattr(config.providers, name, None)
        if p and hasattr(p, "api_key") and p.api_key:
            os.environ.setdefault(env_key, p.api_key)


# ============================================================================
# Phase 1 Provisioning
# ============================================================================


@app.command()
def provision(
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate existing Phase 1 provisioning setup.",
    ),
    bot_token: str | None = typer.Option(
        None,
        "--bot-token",
        help="Discord bot token (required for provision/check).",
    ),
    guild_id: str | None = typer.Option(
        None,
        "--guild-id",
        help="Discord guild/server ID (required for provision/check).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Anthropic model name (e.g. claude-sonnet-4-20250514).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Overwrite existing config without prompt.",
    ),
):
    """Provision a Phase 1 nanobot instance (basic Discord layout only)."""
    from nanobot.config.loader import get_config_path, save_config
    from nanobot.discord.server_setup import setup_basic_server
    from nanobot.provisioning.phase1 import (
        DEFAULT_ANTHROPIC_MODEL,
        build_basic_config,
        run_phase1_checks,
        summarize_expected_layout,
        validate_config_file,
    )

    config_path = get_config_path()

    if check:
        try:
            config, _ = validate_config_file(config_path)
        except RuntimeError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)

        check_bot_token = bot_token or config.channels.discord.token
        check_guild_id = guild_id or config.channels.discord.guild_id
        if not check_bot_token or not check_guild_id:
            console.print(
                "[red]✗[/red] Missing Discord token/guild. "
                "Provide --bot-token/--guild-id or ensure config has channels.discord configured."
            )
            raise typer.Exit(1)

        checks = asyncio.run(
            run_phase1_checks(
                bot_token=check_bot_token,
                guild_id=check_guild_id,
                config_path=config_path,
            )
        )
        table = Table(title="Provision Check")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Details", style="dim")
        has_failure = False
        for item in checks:
            ok = bool(item.ok)
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(item.name, status, item.detail)
            has_failure = has_failure or (not ok)
        console.print(table)
        if has_failure:
            raise typer.Exit(1)
        return

    # Interactive provisioning flow
    provision_bot_token = bot_token or typer.prompt("Discord bot token", hide_input=True).strip()
    provision_guild_id = guild_id or typer.prompt("Discord guild ID").strip()
    provision_model = model or typer.prompt(
        "Anthropic model preference",
        default=DEFAULT_ANTHROPIC_MODEL,
    ).strip()

    if not provision_bot_token:
        console.print("[red]✗[/red] Discord bot token is required.")
        raise typer.Exit(1)
    if not provision_guild_id:
        console.print("[red]✗[/red] Discord guild ID is required.")
        raise typer.Exit(1)

    if config_path.exists() and not yes:
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite existing config with Phase 1 provisioned config?"):
            console.print("[yellow]Provisioning cancelled.[/yellow]")
            raise typer.Exit(1)

    setup_result = asyncio.run(
        setup_basic_server(
            guild_id=provision_guild_id,
            bot_token=provision_bot_token,
        )
    )

    config = build_basic_config(
        bot_token=provision_bot_token,
        guild_id=provision_guild_id,
        model=provision_model,
        channel_ids=setup_result.channel_ids,
        webhook_urls=setup_result.webhook_urls,
    )
    save_config(config, config_path)

    # Post-write validation focused on provision success criteria.
    checks = asyncio.run(
        run_phase1_checks(
            bot_token=provision_bot_token,
            guild_id=provision_guild_id,
            config_path=config_path,
        )
    )
    required_checks = {"config", "discord-access", "discord-layout", "general-profile", "usage-dashboard"}
    has_required_failure = any((not c.ok) for c in checks if c.name in required_checks)

    summary = Table(title="Provisioning Summary")
    summary.add_column("Item", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Guild", provision_guild_id)
    summary.add_row("Model", config.agents.defaults.model)
    summary.add_row("Config", str(config_path))
    summary.add_row("Layout", summarize_expected_layout())
    summary.add_row("Channel #general", setup_result.channel_ids["general"])
    summary.add_row("Channel #system-status", setup_result.channel_ids["system-status"])
    summary.add_row("Channel #claude-usage", setup_result.channel_ids["claude-usage"])
    summary.add_row(
        "General webhook",
        "created" if setup_result.webhook_urls.get("general") else "missing",
    )
    console.print(summary)

    checks_table = Table(title="Provision Validation")
    checks_table.add_column("Check", style="cyan")
    checks_table.add_column("Status")
    checks_table.add_column("Details", style="dim")
    for item in checks:
        status = "[green]PASS[/green]" if item.ok else "[yellow]WARN[/yellow]"
        if item.name in required_checks and not item.ok:
            status = "[red]FAIL[/red]"
        checks_table.add_row(item.name, status, item.detail)
    console.print(checks_table)

    if has_required_failure:
        raise typer.Exit(1)

    console.print("\n[green]✓ Provisioning complete.[/green]")
    console.print("[dim]If Claude auth is missing, run `claude auth login` on this instance, then re-run `nanobot provision --check`.[/dim]")


def _make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.claude_code_provider import ClaudeCodeProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.anthropic_auth import get_oauth_token, is_oauth_token

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    api_key = p.api_key if p and hasattr(p, "api_key") else None

    def _is_claude_model_name(model_name: str) -> bool:
        model_lower = model_name.lower()
        if model_lower.startswith(("claude-code/", "claude_code/")):
            return True
        if "/" in model_lower:
            prefix = model_lower.split("/", 1)[0]
            if prefix in {"anthropic", "claude", "anthropic-direct", "anthropic_direct"}:
                return True
        return "claude" in model_lower

    # Claude Code (OAuth via local CLI)
    if provider_name == "claude_code" or model.startswith("claude-code/") or model.startswith("claude_code/"):
        return ClaudeCodeProvider(
            default_model=model,
            workspace=config.workspace_path,
        )

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    # Explicit API key always wins and routes through LiteLLM.
    if not model.startswith("bedrock/") and api_key:
        return LiteLLMProvider(
            api_key=api_key,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p and hasattr(p, "extra_headers") else None,
            provider_name=provider_name,
        )

    # Fallback: Claude OAuth token (direct Anthropic API).
    direct_cfg = config.providers.anthropic_direct
    wants_anthropic_direct = provider_name == "anthropic_direct" or _is_claude_model_name(model)
    if direct_cfg.enabled and wants_anthropic_direct:
        oauth_token = get_oauth_token()
        if oauth_token and is_oauth_token(oauth_token):
            from nanobot.providers.anthropic_direct_provider import AnthropicDirectProvider

            direct_model = direct_cfg.model or model
            # Ensure secondary provider env vars are set for memory/consolidation ops
            _setup_secondary_env_vars(config)
            return AnthropicDirectProvider(oauth_token=oauth_token, default_model=direct_model)
        if provider_name == "anthropic_direct" or _is_claude_model_name(model):
            console.print("[red]Error: No Anthropic OAuth token configured.[/red]")
            console.print("Run [cyan]claude login[/cyan] or set [cyan]CLAUDE_CODE_OAUTH_TOKEN[/cyan]")
            raise typer.Exit(1)

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not api_key and not (spec and spec.is_oauth):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.nanobot/config.json under providers section")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=api_key,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p and hasattr(p, "extra_headers") else None,
        provider_name=provider_name,
    )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command(name="gateway-worker", hidden=True)
def gateway_worker(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Internal gateway worker process."""
    from nanobot.config.loader import load_config, get_config_path, get_data_dir
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.router import AgentRouter
    from nanobot.agent.profile_manager import AgentProfileManager
    from nanobot.agent.tools.agent_manager import AgentManagerTool
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.discord.usage_dashboard import UsageDashboard
    from nanobot.discord.system_status import SystemStatusDashboard

    # Enable file logging for the worker process
    from nanobot.daemon import setup_daemon_logging
    setup_daemon_logging()

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")

    config = load_config()
    bus = MessageBus()
    provider = _make_provider(config)

    # Create cron service first (callback set after router creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # Create channel manager (shares the front-door bus)
    channels = ChannelManager(config, bus)

    # Create the multi-agent router
    router = AgentRouter(
        front_bus=bus,
        config=config,
        provider=provider,
        cron_service=cron,
    )

    async def _init_router():
        """Initialize router and wire up tools (must be async for create_agent)."""
        await router.initialize_from_config()

        # Wire up agent manager tool on the default agent (only if Discord guild_id is set)
        guild_id = config.channels.discord.guild_id
        default = router.default_agent
        if default and guild_id:
            profile_manager = AgentProfileManager(config, get_config_path())
            tool = AgentManagerTool(
                router=router,
                profile_manager=profile_manager,
                channel_manager=channels,
                guild_id=guild_id,
            )
            default.loop.tools.register(tool)
            console.print(f"[green]✓[/green] Agent manager tool registered on '{router._default_agent_id}'")

        # Register webhooks from saved agent profiles
        discord = channels.get_discord_channel()
        if discord:
            for aid, prof in config.agents.profiles.items():
                if prof.discord_webhook_url and prof.discord_channels:
                    for ch_id in prof.discord_channels:
                        discord.register_webhook(
                            channel_id=ch_id,
                            webhook_url=prof.discord_webhook_url,
                            display_name=prof.display_name,
                            avatar_url=prof.avatar_url,
                        )
                    console.print(f"[green]✓[/green] Webhook registered for agent '{aid}' (name={prof.display_name})")

        # Log agent info
        for aid, inst in router.agents.items():
            tag = " (default)" if aid == router._default_agent_id else ""
            ch_info = f", channels={inst.profile.discord_channels}" if inst.profile.discord_channels else ""
            console.print(f"[green]✓[/green] Agent '{aid}'{tag}: model={inst.profile.model}{ch_info}")

    # Set cron callback (routes to the agent that owns the job)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the owning agent (falls back to default)."""
        target = None
        if job.payload.agent_id:
            target = router.get_agent(job.payload.agent_id)
        if not target:
            target = router.default_agent
        if not target:
            return None
        response = await target.loop.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        if job.payload.deliver and job.payload.to:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response or ""
            ))
        return response
    cron.on_job = on_cron_job

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        default = router.default_agent
        if default:
            for item in default.loop.sessions.list_sessions():
                key = item.get("key") or ""
                if ":" not in key:
                    continue
                channel, chat_id = key.split(":", 1)
                if channel in {"cli", "system"}:
                    continue
                if channel in enabled and chat_id:
                    return channel, chat_id
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        default = router.default_agent
        if not default:
            return ""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            pass

        return await default.loop.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from nanobot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    default_agent = router.default_agent
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=default_agent.loop.model if default_agent else config.agents.defaults.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    # Create usage dashboard if configured
    usage_dashboard: UsageDashboard | None = None
    dc = config.channels.discord
    dash_cfg = dc.usage_dashboard
    if dc.enabled and dash_cfg.enabled and dash_cfg.channel_id and dc.token:
        from nanobot.providers.anthropic_auth import get_oauth_token
        oauth_token = get_oauth_token()
        if oauth_token:
            usage_dashboard = UsageDashboard(
                anthropic_token=oauth_token,
                discord_token=dc.token,
                channel_id=dash_cfg.channel_id,
                poll_interval_s=dash_cfg.poll_interval_s,
                message_id=dash_cfg.message_id or None,
                config_path=str(get_config_path()),
            )
        else:
            console.print("[yellow]⚠ Usage dashboard: no Anthropic OAuth token[/yellow]")

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    if usage_dashboard:
        console.print(f"[green]✓[/green] Usage dashboard: channel={dash_cfg.channel_id}, interval={dash_cfg.poll_interval_s}s")

    # Create system status dashboard if configured
    system_status_dashboard: SystemStatusDashboard | None = None
    status_cfg = dc.system_status
    if dc.enabled and status_cfg.enabled and status_cfg.channel_id and dc.token:
        system_status_dashboard = SystemStatusDashboard(
            router=router,
            discord_token=dc.token,
            channel_id=status_cfg.channel_id,
            poll_interval_s=status_cfg.poll_interval_s,
            message_id=status_cfg.message_id or None,
            config_path=str(get_config_path()),
        )
        console.print(f"[green]✓[/green] System status: channel={status_cfg.channel_id}, interval={status_cfg.poll_interval_s}s")


    async def _send_restart_notifications() -> None:
        """Send a system message to all active agent Discord channels after startup."""
        from nanobot.bus.events import OutboundMessage

        # Give Discord a moment to connect
        await asyncio.sleep(3)

        notified: set[str] = set()
        for aid, inst in router.agents.items():
            # Find Discord session keys for this agent
            for item in inst.loop.sessions.list_sessions():
                key = item.get("key") or ""
                if ":" not in key:
                    continue
                channel, chat_id = key.split(":", 1)
                if channel != "discord" or chat_id in notified:
                    continue
                # Skip cron/system sessions
                if chat_id.startswith("cron:") or chat_id == "direct":
                    continue
                notified.add(chat_id)
                await bus.publish_outbound(OutboundMessage(
                    channel="discord",
                    chat_id=chat_id,
                    content="🔄 *Agent process restarted.*",
                ))
        if notified:
            console.print(f"[green]✓[/green] Restart notifications sent to {len(notified)} channel(s)")

    async def run():
        try:
            await _init_router()
            await cron.start()
            await heartbeat.start()
            if usage_dashboard:
                await usage_dashboard.start()
            if system_status_dashboard:
                await system_status_dashboard.start()
            # Fire restart notifications in the background
            asyncio.create_task(_send_restart_notifications())
            await asyncio.gather(
                router.start(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            if system_status_dashboard:
                await system_status_dashboard.close()
            if usage_dashboard:
                await usage_dashboard.close()
            heartbeat.stop()
            cron.stop()
            await router.stop()
            await channels.stop_all()

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_as_keyboard_interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run in background"),
):
    """Start the nanobot gateway supervisor."""
    from nanobot.daemon import GatewayDaemon

    d = GatewayDaemon(port=port, verbose=verbose)
    d.start(daemonize=daemon)


@app.command()
def restart():
    """Gracefully restart the gateway worker."""
    from nanobot.daemon import GatewayDaemon

    if GatewayDaemon.send_signal(signal.SIGUSR1):
        console.print("[green]Restart signal sent.[/green]")
    else:
        console.print("[red]Gateway not running.[/red]")


@app.command()
def stop():
    """Stop the gateway."""
    from nanobot.daemon import GatewayDaemon

    if GatewayDaemon.send_signal(signal.SIGTERM):
        console.print("[green]Stop signal sent.[/green]")
    else:
        console.print("[red]Gateway not running.[/red]")




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from nanobot.config.loader import load_config, get_data_dir
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    from nanobot.cron.service import CronService
    from loguru import logger
    
    config = load_config()
    
    bus = MessageBus()
    provider = _make_provider(config)

    # Create cron service for tool usage (no callback needed for CLI unless running)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")
    
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        memory_graph_config=config.memory_graph,
    )

    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]nanobot is thinking...[/dim]", spinner="dots")

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        console.print(f"  [dim]↳ {content}[/dim]")

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id, on_progress=_cli_progress)
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — route through bus like other channels
        from nanobot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            elif ch and not is_tool_hint and not ch.send_progress:
                                pass
                            else:
                                console.print(f"  [dim]↳ {msg.content}[/dim]")
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            console.print()
                            _print_agent_response(msg.content, render_markdown=markdown)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        with _thinking_ctx():
                            await turn_done.wait()

                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    dc = config.channels.discord
    table.add_row(
        "Discord",
        "✓" if dc.enabled else "✗",
        dc.gateway_url
    )

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "Feishu",
        "✓" if fs.enabled else "✗",
        fs_config
    )

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row(
        "Mochat",
        "✓" if mc.enabled else "✗",
        mc_base
    )
    
    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row(
        "Slack",
        "✓" if slack.enabled else "✗",
        slack_config
    )

    # DingTalk
    dt = config.channels.dingtalk
    dt_config = f"client_id: {dt.client_id[:10]}..." if dt.client_id else "[dim]not configured[/dim]"
    table.add_row(
        "DingTalk",
        "✓" if dt.enabled else "✗",
        dt_config
    )

    # QQ
    qq = config.channels.qq
    qq_config = f"app_id: {qq.app_id[:10]}..." if qq.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "QQ",
        "✓" if qq.enabled else "✗",
        qq_config
    )

    # Email
    em = config.channels.email
    em_config = em.imap_host if em.imap_host else "[dim]not configured[/dim]"
    table.add_row(
        "Email",
        "✓" if em.enabled else "✗",
        em_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess
    
    # User's bridge location
    user_bridge = Path.home() / ".nanobot" / "bridge"
    
    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge
    
    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)
    
    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)
    
    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge
    
    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)
    
    console.print(f"{__logo__} Setting up bridge...")
    
    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))
    
    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)
    
    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess
    from nanobot.config.loader import load_config
    
    config = load_config()
    bridge_dir = _get_bridge_dir()
    
    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")
    
    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token
    
    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    jobs = service.list_jobs(include_disabled=all)
    
    if not jobs:
        console.print("No scheduled jobs.")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")
    
    import time
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = f"{job.schedule.expr or ''} ({job.schedule.tz})" if job.schedule.tz else (job.schedule.expr or "")
        else:
            sched = "one-time"
        
        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            ts = job.state.next_run_at_ms / 1000
            try:
                tz = ZoneInfo(job.schedule.tz) if job.schedule.tz else None
                next_run = _dt.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M")
            except Exception:
                next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        
        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        
        table.add_row(job.id, job.name, sched, status, next_run)
    
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    tz: str | None = typer.Option(None, "--tz", help="IANA timezone for cron (e.g. 'America/Vancouver')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule
    
    if tz and not cron_expr:
        console.print("[red]Error: --tz can only be used with --cron[/red]")
        raise typer.Exit(1)

    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    try:
        job = service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            to=to,
            channel=channel,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from loguru import logger
    from nanobot.config.loader import load_config, get_data_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.bus.queue import MessageBus
    from nanobot.agent.loop import AgentLoop
    logger.disable("nanobot")

    config = load_config()
    provider = _make_provider(config)
    bus = MessageBus()
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    result_holder = []

    async def on_job(job: CronJob) -> str | None:
        response = await agent_loop.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        result_holder.append(response)
        return response

    service.on_job = on_job

    async def run():
        return await service.run_job(job_id, force=force)

    if asyncio.run(run()):
        console.print("[green]✓[/green] Job executed")
        if result_holder:
            _print_agent_response(result_holder[0], render_markdown=True)
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


def _read_proc_uptime_seconds(pid: int) -> float | None:
    """Return process uptime using /proc/<pid>/stat when available."""
    proc_stat = Path(f"/proc/{pid}/stat")
    proc_uptime = Path("/proc/uptime")
    if not proc_stat.exists() or not proc_uptime.exists():
        return None
    try:
        stat_fields = proc_stat.read_text(encoding="utf-8").split()
        start_ticks = int(stat_fields[21])
        uptime_total = float(proc_uptime.read_text(encoding="utf-8").split()[0])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return max(0.0, uptime_total - (start_ticks / hz))
    except Exception:
        return None


def _format_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    s = int(seconds)
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@app.command()
def status():
    """Show nanobot and gateway status."""
    from nanobot.config.loader import load_config, get_config_path
    from nanobot.daemon import GatewayDaemon

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")
    supervisor_pid = GatewayDaemon.read_pid()
    worker_pid = GatewayDaemon.read_worker_pid()
    if supervisor_pid:
        worker_uptime = _format_uptime(_read_proc_uptime_seconds(worker_pid)) if worker_pid else "unknown"
        worker_part = f", worker PID: {worker_pid}" if worker_pid else ", worker PID: unknown"
        console.print(
            f"Gateway: [green]running[/green] (supervisor PID: {supervisor_pid}{worker_part}, uptime: {worker_uptime})"
        )
    else:
        console.print("Gateway: [yellow]not running[/yellow]")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        if config.providers.anthropic_direct.enabled:
            from nanobot.providers.anthropic_auth import get_oauth_token

            token = get_oauth_token()
            oauth_status = "[green]✓ token found[/green]" if token else "[yellow]no token[/yellow]"
            console.print(f"Anthropic Direct (OAuth): {oauth_status}")
        
        # Check API keys from registry
        for spec in PROVIDERS:
            if spec.name == "anthropic_direct":
                continue
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
