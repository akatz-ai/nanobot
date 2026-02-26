"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.context_log import TurnContextLogger
from nanobot.session.usage_log import TokenUsageLogger
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {_camel_to_snake(str(k)): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    _RESUME_SYSTEM_MESSAGE = (
        "Note: You were interrupted mid-turn by a system restart. "
        "Your previous tool calls and results are preserved above. "
        "Continue where you left off — do not re-execute tools that already have results."
    )

    # Known context window sizes for common models (in tokens).
    # Used to determine when to trigger compaction (at 70% of window).
    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
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
    _DEFAULT_CONTEXT_WINDOW = 200_000
    _COMPACTION_THRESHOLD_RATIO = 0.70  # Compact at 70% of context window
    _HISTORY_MAX_MESSAGES = 1000
    _CONSOLIDATION_KEEP_COUNT = 25
    _CONTINUITY_TTL_MESSAGES = 5

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        background_model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        memory_graph_config: dict | None = None,
        skill_names: list[str] | None = None,
        agent_id: str = "default",
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.agent_id = agent_id
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.background_model = background_model or self.model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._memory_graph_config = (
            _normalize_keys(memory_graph_config) if isinstance(memory_graph_config, dict) else None
        )
        self.skill_names = skill_names

        self.context = ContextBuilder(
            workspace,
            memory_graph_config=self._memory_graph_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._memory_file_lock = asyncio.Lock()  # Serialize global memory file writes across sessions.
        self._context_loggers: dict[str, TurnContextLogger] = {}
        self._usage_loggers: dict[str, TokenUsageLogger] = {}
        self._memory_module = None
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        # Token-based compaction: track last known input_tokens per session
        self._last_input_tokens: dict[str, int] = {}
        self._compaction_token_threshold = self._resolve_compaction_threshold()
        self._register_default_tools()
        self._register_memory_graph_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service, agent_id=self.agent_id))

    def _register_memory_graph_tools(self) -> None:
        """Register optional memory graph tools from adapter package."""
        if not self._memory_graph_config or not self._memory_graph_config.get("enabled"):
            return
        try:
            from agent_memory_nanobot import NanobotMemoryModule

            module_config = dict(self._memory_graph_config)
            module_config.setdefault("background_model", self.background_model)
            self._memory_module = NanobotMemoryModule(
                provider=self.provider,
                workspace=self.workspace,
                config=module_config,
            )
            for tool in self._memory_module.get_tools():
                self.tools.register(tool)
            logger.info("Memory graph tools registered")
        except ImportError:
            logger.warning("agent-memory-nanobot not installed, memory graph disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize memory graph: {e}")

    def _resolve_compaction_threshold(self) -> int:
        """Determine the token threshold at which compaction should trigger."""
        context_window = self._get_context_window_size()
        threshold = int(context_window * self._COMPACTION_THRESHOLD_RATIO)
        logger.info(
            "Compaction threshold: {} tokens ({}% of {} context window for model '{}')",
            threshold,
            int(self._COMPACTION_THRESHOLD_RATIO * 100),
            context_window,
            self.model,
        )
        return threshold

    def _get_context_window_size(self) -> int:
        """Look up the context window size for the current model."""
        model_lower = (self.model or "").lower()
        # Try exact match first, then substring match
        for key, size in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_lower:
                return size
        return self._DEFAULT_CONTEXT_WINDOW

    def _should_compact_by_tokens(self, session_key: str) -> bool:
        """Check if the last known input token count exceeds the compaction threshold."""
        tokens = self._last_input_tokens.get(session_key, 0)
        if tokens >= self._compaction_token_threshold:
            logger.info(
                "Token-based compaction triggered for {}: {} tokens >= {} threshold",
                session_key, tokens, self._compaction_token_threshold,
            )
            return True
        return False

    def _compaction_would_execute(self, session: Session, *, archive_all: bool = False) -> bool:
        """Mirror no-op checks to tell whether consolidation work would execute."""
        if archive_all:
            return bool(session.messages)
        keep_count = max(0, self._CONSOLIDATION_KEEP_COUNT)
        total_messages = len(session.messages)
        if total_messages <= keep_count:
            return False
        if total_messages - session.last_consolidated <= 0:
            return False
        end_index = total_messages - keep_count
        return end_index > session.last_consolidated

    def _get_context_logger(self, session: Session) -> TurnContextLogger:
        """Get or create a TurnContextLogger for a session."""
        key = session.key
        if key not in self._context_loggers:
            session_path = self.sessions._get_session_path(key)
            self._context_loggers[key] = TurnContextLogger(session_path)
        return self._context_loggers[key]

    def _get_usage_logger(self, session: Session) -> TokenUsageLogger:
        """Get or create a TokenUsageLogger for a session."""
        key = session.key
        if key not in self._usage_loggers:
            session_path = self.sessions._get_session_path(key)
            self._usage_loggers[key] = TokenUsageLogger(session_path)
        return self._usage_loggers[key]

    async def _retrieve_memory_context(
        self,
        session: Session,
        user_message: str,
    ) -> str | None:
        """Retrieve compressed memory context for the current turn."""
        if not self._memory_module or not self._memory_module.retriever:
            return None
        try:
            if not self._memory_module.initialized:
                await self._memory_module.initialize()
            retrieval_cfg = {}
            if isinstance(self._memory_graph_config, dict):
                retrieval_cfg = self._memory_graph_config.get("retrieval") or {}
            peer_key = retrieval_cfg["peer_key"] if "peer_key" in retrieval_cfg else session.key
            recent_turns = session.get_history(max_messages=6)
            prompt_headroom_words = self._estimate_retrieval_headroom_words(
                recent_turns=recent_turns,
                user_message=user_message,
            )
            return await self._memory_module.retriever.retrieve_context(
                current_message=user_message,
                recent_turns=recent_turns,
                peer_key=peer_key,
                prompt_headroom_words=prompt_headroom_words,
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return None

    @classmethod
    def _word_count_from_content(cls, content: Any) -> int:
        if isinstance(content, str):
            return len(content.split())
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return len(" ".join(parts).split())
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return len(text.split())
        return len(str(content or "").split())

    def _estimate_retrieval_headroom_words(
        self,
        *,
        recent_turns: list[dict[str, Any]],
        user_message: str,
    ) -> int:
        # Prompt budget should be derived from context window minus completion headroom.
        context_window = max(int(self._get_context_window_size()), 1)
        completion_headroom = max(int(self.max_tokens), 1)
        prompt_token_budget = max(256, context_window - completion_headroom)
        prompt_word_budget = max(120, int(prompt_token_budget * 0.75))

        history_words = sum(
            self._word_count_from_content(turn.get("content"))
            for turn in recent_turns
        )
        message_words = len((user_message or "").split())
        remaining = prompt_word_budget - history_words - message_words
        return max(80, remaining)

    def _clear_expired_continuity_context(self, session: Session) -> None:
        expiry_raw = session.metadata.get("continuity_expires_at_message_count")
        if expiry_raw is None:
            return
        try:
            expiry = int(expiry_raw)
        except (TypeError, ValueError):
            session.metadata.pop("continuity_context", None)
            session.metadata.pop("continuity_expires_at_message_count", None)
            return
        if len(session.messages) > expiry:
            session.metadata.pop("continuity_context", None)
            session.metadata.pop("continuity_expires_at_message_count", None)

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _error_signature(tool_name: str, args: dict, error: str) -> str:
        """Create a hashable signature for a tool error to detect repetition.

        Uses full argument values (not just keys) so that different arguments
        with the same tool name don't falsely trigger the circuit breaker.
        """
        # Include full args JSON so only truly identical calls match
        args_json = json.dumps(args, sort_keys=True, ensure_ascii=False) if args else ""
        return f"{tool_name}:{args_json}:{error[:200]}"

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        session: Session | None = None,
        checkpoint_start: int | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        checkpoint_cursor = len(messages)

        # Circuit breaker: track consecutive identical tool calls that produce errors.
        # Only triggers when the model retries the EXACT same tool+args+error combo.
        _last_error_sig: str | None = None
        _consecutive_errors: int = 0
        _MAX_CONSECUTIVE_ERRORS = 5  # Raised from 3 — less aggressive

        if session is not None and checkpoint_start is not None:
            start_index = max(0, checkpoint_start)
            if start_index < len(messages):
                session.checkpoint(messages[start_index:])
            checkpoint_cursor = len(messages)

        while iteration < self.max_iterations:
            iteration += 1

            _api_start = time.monotonic()
            logger.debug("Calling provider.chat() — iteration {}/{}, {} messages", iteration, self.max_iterations, len(messages))
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            _api_elapsed = time.monotonic() - _api_start
            logger.debug("provider.chat() returned in {:.1f}s — finish_reason={}, tool_calls={}", _api_elapsed, response.finish_reason, len(response.tool_calls) if response.tool_calls else 0)

            # Track token usage for compaction decisions + sidecar log
            if response.usage and session is not None:
                input_tokens = response.usage.get("prompt_tokens", 0)
                output_tokens = response.usage.get("completion_tokens", 0)
                cache_read = response.usage.get("cache_read_input_tokens", 0)
                cache_creation = response.usage.get("cache_creation_input_tokens", 0)
                # Anthropic's input_tokens only reports non-cached tokens.
                # Total context = input_tokens + cache_read + cache_creation.
                total_input = input_tokens + cache_read + cache_creation
                if total_input:
                    self._last_input_tokens[session.key] = total_input
                    context_window = self._get_context_window_size()
                    utilization = round(total_input / context_window * 100, 1) if context_window else 0
                    logger.info(
                        "Token usage: total_in={} (raw={} cache_read={} cache_create={}) out={} | {:.1f}% of {}k window",
                        total_input,
                        input_tokens,
                        cache_read,
                        cache_creation,
                        output_tokens,
                        utilization,
                        context_window // 1000,
                    )
                    # Write to sidecar usage log
                    usage_logger = self._get_usage_logger(session)
                    usage_logger.log_usage(
                        usage=response.usage,
                        iteration=iteration,
                        context_window=context_window,
                        model=self.model,
                        finish_reason=response.finish_reason,
                    )

            # --- Layer 1: Truncation guard ---
            # When output hits max_tokens, tool call JSON may be truncated,
            # producing malformed calls with missing required parameters.
            # Detect this and tell the model to break its output into smaller pieces.
            if response.finish_reason == "length" and response.has_tool_calls:
                logger.warning(
                    "Output truncated (max_tokens={}) with {} tool call(s) — "
                    "discarding likely-malformed calls",
                    self.max_tokens,
                    len(response.tool_calls),
                )
                truncation_notice = (
                    "Your response was truncated because it exceeded the output token limit "
                    f"({self.max_tokens} tokens). Your tool call was NOT executed because "
                    "its parameters were likely incomplete.\n\n"
                    "Break your work into smaller pieces:\n"
                    "- For write_file: write the file in sections using multiple calls, "
                    "or use exec with heredoc (cat << 'EOF' > file)\n"
                    "- For long outputs: summarize or split across multiple responses"
                )
                # Add the truncated assistant message (so context is preserved)
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                # Add synthetic tool results with the truncation notice
                for tc in response.tool_calls:
                    messages = self.context.add_tool_result(
                        messages, tc.id, tc.name, truncation_notice
                    )
                if session is not None and checkpoint_cursor < len(messages):
                    session.checkpoint(messages[checkpoint_cursor:])
                    checkpoint_cursor = len(messages)
                continue

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                if session is not None and checkpoint_cursor < len(messages):
                    session.checkpoint(messages[checkpoint_cursor:])
                    checkpoint_cursor = len(messages)

                # --- Layer 2: Repetitive error circuit breaker ---
                _turn_had_error = False
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)

                    # Track consecutive identical errors
                    if isinstance(result, str) and "Error" in result:
                        sig = self._error_signature(tool_call.name, tool_call.arguments, result)
                        if sig == _last_error_sig:
                            _consecutive_errors += 1
                        else:
                            _last_error_sig = sig
                            _consecutive_errors = 1
                        _turn_had_error = True
                    else:
                        # Successful tool call resets the counter
                        _last_error_sig = None
                        _consecutive_errors = 0

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if session is not None and checkpoint_cursor < len(messages):
                        session.checkpoint(messages[checkpoint_cursor:])
                        checkpoint_cursor = len(messages)

                logger.debug("All {} tool call(s) processed, continuing loop", len(response.tool_calls))

                # Check circuit breaker after processing all tool calls in this iteration
                if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "Circuit breaker: {} consecutive identical errors for {}",
                        _consecutive_errors,
                        _last_error_sig,
                    )
                    final_content = (
                        f"I'm stuck in a loop — the same tool call has failed "
                        f"{_consecutive_errors} times with the same error. "
                        "I'll stop here to avoid wasting resources. "
                        "Try rephrasing your request or breaking it into smaller steps."
                    )
                    break
            else:
                final_content = self._strip_think(response.content)
                messages = self.context.add_assistant_message(
                    messages,
                    final_content,
                    reasoning_content=response.reasoning_content,
                )
                if session is not None and checkpoint_cursor < len(messages):
                    session.checkpoint(messages[checkpoint_cursor:])
                    checkpoint_cursor = len(messages)
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    @classmethod
    def _resume_notice_for_state(cls, state: str) -> str | None:
        if state in {"mid_tool", "mid_loop"}:
            return cls._RESUME_SYSTEM_MESSAGE
        return None

    @staticmethod
    def _split_session_key(key: str) -> tuple[str, str]:
        if ":" in key:
            return key.split(":", 1)
        return "cli", key

    async def _resume_session(self, session: Session, channel: str, chat_id: str) -> OutboundMessage | None:
        """Resume an interrupted turn without a new inbound user message."""
        self._set_tool_context(channel, chat_id)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self._HISTORY_MAX_MESSAGES)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=None,
            skill_names=self.skill_names,
            channel=channel,
            chat_id=chat_id,
            resume_notice=self._RESUME_SYSTEM_MESSAGE,
        )
        self._get_context_logger(session).log_turn(
            built_messages=initial_messages,
            memory_context=None,
            resume_notice=self._RESUME_SYSTEM_MESSAGE,
            user_message_index=len(session.messages),
        )
        self._get_usage_logger(session).new_turn()
        self.subagents.set_skill_index(self.context.get_last_skills_summary())
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            session=session,
            checkpoint_start=None,
        )

        self._save_turn(session, all_msgs, len(initial_messages))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=final_content or "Background task completed.",
            metadata={},
        )

    async def resume_inflight_sessions(self) -> int:
        """Auto-resume interrupted sessions when no inbound message is waiting."""
        if self.bus.inbound_size > 0:
            logger.info("Skipping auto-resume: inbound queue already has pending messages")
            return 0

        resumed = 0
        for item in self.sessions.list_sessions():
            key = item.get("key")
            if not key:
                continue
            if self.bus.inbound_size > 0:
                break

            session = self.sessions.get_or_create(key)
            if session.detect_resume_state() not in {"mid_tool", "mid_loop"}:
                continue

            channel, chat_id = self._split_session_key(key)
            logger.info("Auto-resuming interrupted session {}", key)
            response = await self._resume_session(session, channel, chat_id)
            if response is not None:
                await self.bus.publish_outbound(response)
            resumed += 1

        return resumed

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        resumed = await self.resume_inflight_sessions()
        if resumed:
            logger.info("Auto-resumed {} interrupted session(s)", resumed)
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop and flush all cached sessions to disk."""
        self._running = False
        saved = self.sessions.save_all()
        if saved:
            logger.info("Flushed {} session(s) to disk", saved)
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            metadata = msg.metadata or {}
            resume_notice = self._resume_notice_for_state(session.detect_resume_state())
            self._set_tool_context(channel, chat_id, metadata.get("message_id"))
            memory_context = await self._retrieve_memory_context(session, msg.content)
            history = session.get_history(max_messages=self._HISTORY_MAX_MESSAGES)
            initial_messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                skill_names=self.skill_names,
                channel=channel,
                chat_id=chat_id,
                memory_context=memory_context,
                resume_notice=resume_notice,
            )
            self._get_context_logger(session).log_turn(
                built_messages=initial_messages,
                memory_context=memory_context,
                resume_notice=resume_notice,
                user_message_index=len(session.messages),
            )
            self._get_usage_logger(session).new_turn()
            self.subagents.set_skill_index(self.context.get_last_skills_summary())
            turn_start = max(len(initial_messages) - 1, 0)
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages,
                session=session,
                checkpoint_start=turn_start,
            )
            self._save_turn(session, all_msgs, turn_start)
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        metadata = msg.metadata or {}
        self._clear_expired_continuity_context(session)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            session.metadata.pop("continuity_context", None)
            session.metadata.pop("continuity_expires_at_message_count", None)
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        # Token-based compaction: check if the session's last known input token count
        # exceeds 70% of the model's context window.
        _needs_compaction = False
        _compaction_reason = "none"
        if session.key not in self._consolidating:
            _token_count = self._last_input_tokens.get(session.key, 0)
            logger.info(
                "Compaction check for {}: tokens={}/{} ({}%), last_consolidated={}, total={}",
                session.key,
                _token_count,
                self._compaction_token_threshold,
                round(_token_count / self._compaction_token_threshold * 100, 1) if self._compaction_token_threshold else 0,
                session.last_consolidated,
                len(session.messages),
            )
            if self._should_compact_by_tokens(session.key):
                _needs_compaction = True
                _compaction_reason = f"tokens ({_token_count} >= {self._compaction_token_threshold})"
        else:
            logger.debug(
                "Skipping compaction check for {}: consolidation already in progress",
                session.key,
            )

        continuity_context = None
        if _needs_compaction:
            logger.info(
                "Compaction TRIGGERED for {} — reason: {}",
                session.key, _compaction_reason,
            )
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    # Avoid repeated no-op compaction loops from stale token readings.
                    if not self._compaction_would_execute(session):
                        logger.info(
                            "Compaction preflight no-op for {} (last_consolidated={}, total={})",
                            session.key,
                            session.last_consolidated,
                            len(session.messages),
                        )
                        self._last_input_tokens.pop(session.key, None)
                    else:
                        try:
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="⏳ *Compacting session* — summarizing older context to free up space...",
                                metadata={"_system_notice": True},
                            ))
                        except Exception:
                            logger.warning("Failed to send compaction-start notice")

                        # Snapshot continuity so this same turn can keep recent conversation detail.
                        continuity = self._snapshot_continuity_context(session)
                        if continuity:
                            logger.info(
                                "Saved continuity context ({} chars) for post-compaction injection",
                                len(continuity),
                            )

                        prev_consolidated = session.last_consolidated
                        if await self._consolidate_memory(session):
                            # _consolidate_memory returns True on no-ops too; clear stale token
                            # reading either way to avoid immediate retrigger loops.
                            self._last_input_tokens.pop(session.key, None)
                            if session.last_consolidated > prev_consolidated:
                                # Persist continuity for bounded follow-up turns.
                                if continuity:
                                    session.metadata["continuity_context"] = continuity
                                    session.metadata["continuity_expires_at_message_count"] = (
                                        len(session.messages) + self._CONTINUITY_TTL_MESSAGES
                                    )
                                continuity_context = continuity
                                self.sessions.save(session)
                                try:
                                    await self.bus.publish_outbound(OutboundMessage(
                                        channel=msg.channel,
                                        chat_id=msg.chat_id,
                                        content=(
                                            "⚙️ *Session compacted* — older context has been "
                                            "summarized. Continuing with your message..."
                                        ),
                                        metadata={"_system_notice": True},
                                    ))
                                except Exception:
                                    logger.warning("Failed to send compaction-complete notice")
                            else:
                                logger.debug(
                                    "Compaction was a no-op for {} (checkpoint unchanged at {})",
                                    session.key, prev_consolidated,
                                )
                        else:
                            logger.warning("Compaction failed for {}", session.key)
            except Exception:
                logger.exception("Inline compaction failed for {}", session.key)
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

        resume_notice = self._resume_notice_for_state(session.detect_resume_state())
        self._set_tool_context(msg.channel, msg.chat_id, metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        memory_context = await self._retrieve_memory_context(session, msg.content)
        history = session.get_history(max_messages=self._HISTORY_MAX_MESSAGES)

        # Load persisted continuity context if no inline compaction produced one this turn
        if not continuity_context:
            continuity_context = session.metadata.get("continuity_context")
        if continuity_context:
            logger.info("Injecting continuity context ({} chars)", len(continuity_context))

        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            skill_names=self.skill_names,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            memory_context=memory_context,
            resume_notice=resume_notice,
            continuity_context=continuity_context,
        )
        self._get_context_logger(session).log_turn(
            built_messages=initial_messages,
            memory_context=memory_context,
            resume_notice=resume_notice,
            user_message_index=len(session.messages),
        )
        self._get_usage_logger(session).new_turn()
        self.subagents.set_skill_index(self.context.get_last_skills_summary())
        turn_start = max(len(initial_messages) - 1, 0)

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            session=session,
            checkpoint_start=turn_start,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, turn_start)
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=metadata,
        )

    def _save_turn(self, session: Session, messages: list[dict], start_index: int) -> None:
        """End-of-turn hook; messages are checkpointed during loop execution."""
        _ = (session, messages, start_index)

    @staticmethod
    def _snapshot_continuity_context(
        session: Session,
        *,
        max_exchanges: int = 4,
        max_chars: int = 3000,
    ) -> str | None:
        """Extract the last N user/assistant conversational turns for post-compaction continuity.

        Returns a formatted string of recent conversation, or None if nothing useful.

        Only captures actual conversational turns — each "exchange" is a user message
        paired with the final assistant response that follows it. Intermediate assistant
        messages during tool loops (e.g. "Let me check that...") are skipped so the
        continuity context reflects the real conversation, not tool-call narration.
        """
        # Phase 1: Walk backwards to find user messages and their final assistant replies.
        # An "exchange" = (user_text, assistant_text | None).
        # The "final assistant reply" for a user message is the last assistant message
        # with text content that appears before the *next* user message (or end of list).
        exchanges: list[tuple[str, str | None]] = []  # (user_text, assistant_text)
        last_assistant_text: str | None = None

        for msg in reversed(session.messages):
            role = msg.get("role")
            if role == "assistant":
                content = msg.get("content")
                text = str(content).strip() if content else ""
                # Keep the first (i.e. latest) non-empty assistant text we find
                # as we walk backwards — that's the final reply for the preceding user msg.
                if text and last_assistant_text is None:
                    last_assistant_text = text
            elif role == "user":
                text = str(msg.get("content", "")).strip()
                if text:
                    exchanges.append((text, last_assistant_text))
                    last_assistant_text = None  # reset for the next (earlier) exchange
                    if len(exchanges) >= max_exchanges:
                        break
            # tool results, system messages, etc. — skip silently

        if not exchanges:
            return None

        exchanges.reverse()

        # Phase 2: Build formatted output, respecting the char budget.
        lines: list[str] = []
        total_chars = 0
        for user_text, assistant_text in exchanges:
            # Truncate individual messages that are very long
            if len(user_text) > 600:
                user_text = user_text[:597] + "..."
            user_line = f"**User:** {user_text}"

            entry_chars = len(user_line) + 1
            assistant_line = None
            if assistant_text:
                if len(assistant_text) > 600:
                    assistant_text = assistant_text[:597] + "..."
                assistant_line = f"**Assistant:** {assistant_text}"
                entry_chars += len(assistant_line) + 1

            if total_chars + entry_chars > max_chars and lines:
                break
            lines.append(user_line)
            if assistant_line:
                lines.append(assistant_line)
            total_chars += entry_chars

        return "\n\n".join(lines) if lines else None

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Consolidate file memory and optionally graph memory. Returns file-memory success."""
        async with self._memory_file_lock:
            consolidation_cfg = (
                self._memory_graph_config.get("consolidation") or {}
                if isinstance(self._memory_graph_config, dict)
                else {}
            )
            engine = str(consolidation_cfg.get("engine") or "legacy").lower()

            if engine == "hybrid" and self._memory_module and self._memory_module.hybrid:
                try:
                    if not self._memory_module.initialized:
                        await self._memory_module.initialize()

                    keep_count = 0 if archive_all else self._CONSOLIDATION_KEEP_COUNT
                    logger.info(
                        "Hybrid consolidation for {}: archive_all={}, keep_count={}, "
                        "total_messages={}, last_consolidated={}",
                        session.key, archive_all, keep_count,
                        len(session.messages), session.last_consolidated,
                    )
                    if not archive_all:
                        if len(session.messages) <= keep_count:
                            logger.info(
                                "Hybrid consolidation no-op: total_messages ({}) <= keep_count ({})",
                                len(session.messages), keep_count,
                            )
                            return True
                        if len(session.messages) - session.last_consolidated <= 0:
                            logger.info(
                                "Hybrid consolidation no-op: nothing unconsolidated",
                            )
                            return True
                        start_index = session.last_consolidated
                        end_index = len(session.messages) - keep_count
                        if end_index <= start_index:
                            logger.info(
                                "Hybrid consolidation no-op: end_index ({}) <= start_index ({})",
                                end_index, start_index,
                            )
                            return True
                        target_last_consolidated = end_index
                    else:
                        start_index = 0
                        end_index = len(session.messages)
                        target_last_consolidated = 0

                    logger.info(
                        "Hybrid consolidation executing: messages[{}:{}] -> last_consolidated={}",
                        start_index, end_index, target_last_consolidated,
                    )
                    await self._memory_module.hybrid.compact(
                        session_key=session.key,
                        messages=session.messages,
                        start_index=start_index,
                        end_index=end_index,
                    )
                    session.last_consolidated = target_last_consolidated
                    logger.info(
                        "Hybrid memory consolidation done: {} messages, last_consolidated={}",
                        len(session.messages),
                        session.last_consolidated,
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Hybrid memory consolidation failed: {e}")
                    return False

            graph_window_messages: list[dict[str, Any]] = []
            if self._memory_module and self._memory_module.consolidator:
                if archive_all:
                    graph_window_messages = list(session.messages)
                else:
                    keep_count = self._CONSOLIDATION_KEEP_COUNT
                    if len(session.messages) > keep_count and len(session.messages) > session.last_consolidated:
                        start_index = session.last_consolidated
                        end_index = len(session.messages) - keep_count
                        if end_index > start_index:
                            graph_window_messages = list(session.messages[start_index:end_index])

            success = await MemoryStore(self.workspace).consolidate(
                session, self.provider, self.background_model,
                archive_all=archive_all, keep_count=self._CONSOLIDATION_KEEP_COUNT,
            )
            if self._memory_module and self._memory_module.consolidator and graph_window_messages:
                try:
                    if not self._memory_module.initialized:
                        await self._memory_module.initialize()
                    await self._memory_module.consolidator.consolidate_session(
                        messages=graph_window_messages,
                        peer_key=session.key,
                        source_session=session.key,
                    )
                    logger.info("Memory graph consolidation complete")
                except Exception as e:
                    logger.warning(f"Memory graph consolidation failed: {e}")
            return success

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
