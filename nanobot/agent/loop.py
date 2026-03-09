"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter
from contextlib import AsyncExitStack
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.batch import BatchTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.session_search import SessionSearchTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.session.evidence import (
    EvidenceIndex,
    build_evidence_content,
    _evidence_threshold_content,
    should_index_tool,
)
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.compaction import (
    _decision_tokens_with_pruning_ceiling,
    _usage_snapshot_tokens,
    compact_session,
    estimate_message_tokens,
    extract_file_ops,
    generate_compaction_summary,
    should_compact,
)
from nanobot.session.compaction_log import CompactionEvent, CompactionLogger
from nanobot.session.context_log import TurnContextLogger
from nanobot.session.extraction_log import ExtractionEvent, ExtractionLogger
from nanobot.session.inbox import InboxEvent
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
    # Used to determine when to trigger compaction (at 75% of window).
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
    _COMPACTION_THRESHOLD_RATIO = 0.75  # Compact at 75% of context window
    _PROMPT_BUDGET_RATIO = 0.90  # Allow append-only growth between compactions
    _CONSOLIDATION_KEEP_COUNT = 25
    _COMPACTION_RESERVE_TOKENS = 16_384
    _COMPACTION_KEEP_RECENT_TOKENS = 20_000
    _EMERGENCY_TRIM_SAFETY_TOKENS = 4_096
    _PRUNE_PROTECTED_TOOLS = frozenset(
        {"memory_recall", "memory_save", "memory_graph", "memory_ingest"}
    )

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
        memory_window: int | None = None,
        reasoning_effort: str | None = None,
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
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
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
            reasoning_effort=reasoning_effort,
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
        self._compaction_loggers: dict[str, CompactionLogger] = {}
        self._extraction_loggers: dict[str, ExtractionLogger] = {}
        self._memory_module = None
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_locks: dict[str, asyncio.Lock] = {}
        # Token-based compaction: track last known input_tokens per session
        self._last_input_tokens: dict[str, int] = {}
        self._compaction_token_threshold = self._resolve_compaction_threshold()
        self._evidence_indexes: dict[str, EvidenceIndex] = {}
        self._current_session_key_ctx: ContextVar[str | None] = ContextVar(
            "agent_loop_current_session_key",
            default=None,
        )
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
        self.tools.register(BatchTool(registry=self.tools))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service, agent_id=self.agent_id))
        self.tools.register(SessionSearchTool(get_evidence_index=self._get_current_evidence_index))

    def _register_memory_graph_tools(self) -> None:
        """Register optional memory graph tools from adapter package."""
        if not self._memory_graph_config or not self._memory_graph_config.get("enabled"):
            return
        try:
            from agent_memory_nanobot import NanobotMemoryModule

            module_config = dict(self._memory_graph_config)
            module_config.setdefault("background_model", self.background_model)
            module_config.setdefault("agent_id", self.agent_id)
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
        threshold = self._get_compaction_threshold()
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

    def _get_context_logger(self, session: Session) -> TurnContextLogger:
        """Get or create a TurnContextLogger for a session."""
        key = session.key
        if key not in self._context_loggers:
            session_path = self.sessions.get_session_path(key)
            self._context_loggers[key] = TurnContextLogger(session_path)
        return self._context_loggers[key]

    def _get_usage_logger(self, session: Session) -> TokenUsageLogger:
        """Get or create a TokenUsageLogger for a session."""
        key = session.key
        if key not in self._usage_loggers:
            session_path = self.sessions.get_session_path(key)
            self._usage_loggers[key] = TokenUsageLogger(session_path)
        return self._usage_loggers[key]

    def _get_compaction_logger(self, session: Session) -> CompactionLogger:
        """Get or create a CompactionLogger for a session."""
        key = session.key
        if key not in self._compaction_loggers:
            session_path = self.sessions.get_session_path(key)
            self._compaction_loggers[key] = CompactionLogger(session_path)
        return self._compaction_loggers[key]

    def _get_extraction_logger(self, session: Session) -> ExtractionLogger:
        """Get or create an ExtractionLogger for a session."""
        key = session.key
        if key not in self._extraction_loggers:
            session_path = self.sessions.get_session_path(key)
            self._extraction_loggers[key] = ExtractionLogger(session_path)
        return self._extraction_loggers[key]

    @staticmethod
    def _usage_value(usage: dict[str, Any] | None, key: str) -> int | None:
        if not isinstance(usage, dict):
            return None
        value = usage.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _summarize_extracted_items(items: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        actions = Counter()
        types = Counter()
        scopes = Counter()
        for item in items:
            action = str(item.get("action", "")).strip().lower()
            if action:
                actions[action] += 1
            memory_type = str(
                item.get("type", item.get("memory_type", item.get("entry_type", "")))
            ).strip().lower()
            if memory_type:
                types[memory_type] += 1
            scope = str(item.get("scope", "")).strip().lower()
            if scope:
                scopes[scope] += 1
        return dict(actions), dict(types), dict(scopes)

    def _history_file_for_log(self, history_file: Any) -> str | None:
        if not history_file:
            return None
        try:
            path = Path(history_file)
        except TypeError:
            return None
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return str(path)

    def _record_extraction_event(
        self,
        *,
        session: Session,
        batch_start: int,
        batch_end: int,
        last_consolidated_before: int,
        batch_duration_ms: int,
        result: Any,
        reason: str,
        compaction_event_id: str | None,
    ) -> None:
        extracted_items = getattr(result, "extracted_items", None)
        if not isinstance(extracted_items, list):
            extracted_items = []
        indexed_items = getattr(result, "indexed_items", None)
        if not isinstance(indexed_items, list):
            indexed_items = []

        item_actions, item_types, item_scopes = self._summarize_extracted_items(extracted_items)
        usage = getattr(result, "llm_usage", None)
        retry_usage = getattr(result, "retry_llm_usage", None)
        finish_reason = str(getattr(result, "finish_reason", "unknown") or "unknown")
        error = getattr(result, "error", None)

        event = ExtractionEvent(
            session_key=session.key,
            agent_id=self.agent_id or "",
        )
        event.set_trigger(
            reason=reason,
            compaction_event_id=compaction_event_id,
            total_session_messages=len(session.messages),
            last_consolidated_before=last_consolidated_before,
        )
        context_overlap_count = int(getattr(result, "context_overlap_count", 0) or 0)
        event.set_message_window(
            start_index=batch_start,
            end_index=batch_end,
            user_turns=int(getattr(result, "user_turns_in_window", 0) or 0),
            context_overlap_start=max(0, batch_start - context_overlap_count),
            context_overlap_count=context_overlap_count,
        )
        existing_memories = getattr(result, "existing_memories_recalled", None)
        event.set_existing_memories(
            query_text_chars=int(getattr(result, "existing_memories_query_text_chars", 0) or 0),
            results_requested=int(getattr(result, "existing_memories_results_requested", 0) or 0),
            results_returned=len(existing_memories) if isinstance(existing_memories, list) else 0,
            recall_duration_ms=int(
                getattr(result, "existing_memories_recall_duration_ms", 0) or 0
            ),
            memories=existing_memories if isinstance(existing_memories, list) else [],
        )
        event.set_prompt_stats(
            system_prompt_chars=int(getattr(result, "prompt_system_chars", 0) or 0),
            transcript_chars=int(getattr(result, "transcript_chars", 0) or 0),
            context_section_chars=int(getattr(result, "prompt_context_section_chars", 0) or 0),
            extraction_section_chars=int(
                getattr(result, "prompt_extraction_section_chars", 0) or 0
            ),
            existing_memories_section_chars=int(
                getattr(result, "prompt_existing_memories_chars", 0) or 0
            ),
            max_tokens_budget=int(getattr(result, "max_tokens_budget", 0) or 0),
            full_prompt_hash=str(getattr(result, "prompt_full_hash", "") or ""),
        )
        event.set_llm_call(
            model=str(
                getattr(self._memory_module.consolidator, "model", "")
                if self._memory_module and getattr(self._memory_module, "consolidator", None)
                else ""
            ),
            temperature=0.0,
            duration_ms=int(getattr(result, "llm_duration_ms", 0) or 0),
            input_tokens=self._usage_value(usage, "prompt_tokens"),
            output_tokens=self._usage_value(usage, "completion_tokens"),
            cache_read_tokens=self._usage_value(usage, "cache_read_input_tokens"),
            cache_creation_tokens=self._usage_value(usage, "cache_creation_input_tokens"),
            retry_needed=bool(getattr(result, "retry_needed", False)),
            retry_input_tokens=self._usage_value(retry_usage, "prompt_tokens"),
            retry_output_tokens=self._usage_value(retry_usage, "completion_tokens"),
            finish_reason=finish_reason,
            error=str(error) if finish_reason == "error" and error else None,
        )
        event.set_extraction_raw(
            response_chars=int(getattr(result, "llm_response_chars", 0) or 0),
            response_preview=str(getattr(result, "llm_response_preview", "") or ""),
            parse_ok=bool(getattr(result, "success", False)),
            parse_error=None if bool(getattr(result, "success", False)) else (str(error) if error else None),
            items_before_cap=int(getattr(result, "items_before_cap", 0) or 0),
            items_after_cap=len(extracted_items),
            items_by_action=item_actions,
            items_by_type=item_types,
            items_by_scope=item_scopes,
        )
        event.set_extracted_items(extracted_items)
        event.set_graph_indexing(
            duration_ms=int(getattr(result, "indexing_duration_ms", 0) or 0),
            memories_added=int(getattr(result, "memories_added", 0) or 0),
            memories_updated=int(getattr(result, "memories_updated", 0) or 0),
            memories_superseded=int(getattr(result, "memories_superseded", 0) or 0),
            edges_created=int(getattr(result, "edges_created", 0) or 0),
            items_skipped=int(getattr(result, "items_skipped", 0) or 0),
            indexed_items=indexed_items,
        )
        event.set_file_ops(
            history_file=self._history_file_for_log(getattr(result, "history_file", None)),
            history_entries_written=len(getattr(result, "entries", []) or []),
            memory_md_rewrite_triggered=bool(
                getattr(result, "memory_md_rewrite_triggered", False)
            ),
            memory_md_before_chars=int(getattr(result, "memory_md_before_chars", 0) or 0),
            memory_md_after_chars=int(getattr(result, "memory_md_after_chars", 0) or 0),
            memory_md_duration_ms=int(getattr(result, "memory_md_duration_ms", 0) or 0),
        )
        event.finalize(
            success=bool(getattr(result, "success", False)),
            error=str(error) if error else None,
            total_duration_ms=batch_duration_ms,
            total_items_extracted=len(extracted_items),
            total_items_indexed=len(indexed_items),
            new_last_consolidated=batch_end,
        )
        self._get_extraction_logger(session).write(event)

    def _get_evidence_index(self, session: Session) -> EvidenceIndex:
        """Get or create an EvidenceIndex for a session (lazy)."""
        key = session.key
        if key not in self._evidence_indexes:
            session_path = self.sessions.get_session_path(key)
            sqlite_path = session_path.with_suffix(".evidence.sqlite")
            self._evidence_indexes[key] = EvidenceIndex(sqlite_path)
        return self._evidence_indexes[key]

    def _close_evidence_index(self, session_key: str, *, delete_files: bool = False) -> None:
        """Close a cached evidence index and optionally remove its sidecar files."""
        index = self._evidence_indexes.pop(session_key, None)
        if index is not None:
            index.close()

        if not delete_files:
            return

        sqlite_path = self.sessions.get_session_path(session_key).with_suffix(".evidence.sqlite")
        for candidate in (
            sqlite_path,
            sqlite_path.with_name(f"{sqlite_path.name}-wal"),
            sqlite_path.with_name(f"{sqlite_path.name}-shm"),
        ):
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove evidence sidecar {}", candidate)

    def _get_current_evidence_index(self) -> EvidenceIndex | None:
        """Return the evidence index for the currently active session (for tool callback)."""
        session_key = self._current_session_key_ctx.get()
        if session_key and session_key in self._evidence_indexes:
            return self._evidence_indexes[session_key]
        return None

    def _index_tool_result(
        self,
        session: Session,
        tool_name: str,
        result: str,
        message_index: int,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Index a tool result in the session evidence index if applicable."""
        content, metadata = build_evidence_content(tool_name, result, arguments)
        threshold_content = _evidence_threshold_content(tool_name, result, arguments)
        if not should_index_tool(tool_name, threshold_content):
            return
        try:
            index = self._get_evidence_index(session)
            ts = None
            message = session.get_message_at(message_index)
            if message is not None:
                ts = message.get("timestamp")
            index.add(
                message_index=message_index,
                tool_name=tool_name,
                content=content,
                timestamp=ts,
                metadata=metadata,
            )
        except Exception:
            logger.warning("Evidence indexing failed for tool={} msg_idx={}", tool_name, message_index)

    async def _retrieve_memory_context(
        self,
        session: Session,
        user_message: str,
    ) -> str | None:
        """Retrieve compressed memory context for the current turn."""
        if not self._memory_module or not self._memory_module.retriever:
            return None
        try:
            import time as _time
            _t0 = _time.monotonic()
            if not self._memory_module.initialized:
                await self._memory_module.initialize()
                logger.info("Memory module initialized in {:.2f}s", _time.monotonic() - _t0)
            retrieval_cfg = {}
            if isinstance(self._memory_graph_config, dict):
                retrieval_cfg = self._memory_graph_config.get("retrieval") or {}
            peer_key = retrieval_cfg["peer_key"] if "peer_key" in retrieval_cfg else session.key
            context_window = self._get_context_window_size()
            recent_turns, _ = session.get_history(
                max_messages=6,
                context_window=context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            prompt_headroom_words = self._estimate_retrieval_headroom_words(
                recent_turns=recent_turns,
                user_message=user_message,
            )
            _t1 = _time.monotonic()
            result = await self._memory_module.retriever.retrieve_context(
                current_message=user_message,
                recent_turns=recent_turns,
                peer_key=peer_key,
                agent_id=self.agent_id,
                prompt_headroom_words=prompt_headroom_words,
            )
            logger.info(
                "Memory retrieval completed in {:.2f}s (total prep+retrieve: {:.2f}s)",
                _time.monotonic() - _t1,
                _time.monotonic() - _t0,
            )
            return result
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

    def _get_compaction_threshold(self) -> int:
        """Return the synchronous compaction trigger threshold."""
        context_window = max(int(self._get_context_window_size()), 1)
        return max(
            512,
            int((context_window - self._COMPACTION_RESERVE_TOKENS) * self._COMPACTION_THRESHOLD_RATIO),
        )

    def _get_emergency_prompt_threshold(self) -> int:
        """Return the last-resort prompt threshold after output reservation."""
        context_window = max(int(self._get_context_window_size()), 1)
        output_reserve = max(int(self.max_tokens), 1)
        return max(
            512,
            context_window - output_reserve - self._EMERGENCY_TRIM_SAFETY_TOKENS,
        )

    def _get_prompt_input_budget(self) -> int:
        """Return the soft prompt-input budget used between compactions.

        Compaction still triggers at ``_COMPACTION_THRESHOLD_RATIO``, but the
        preflight path should allow append-only growth after compaction until we
        approach the real provider limit. The hard emergency limit remains
        ``context_window - output_reserve - safety_margin``.
        """
        context_window = max(int(self._get_context_window_size()), 1)
        return max(
            512,
            min(
                int(context_window * self._PROMPT_BUDGET_RATIO),
                self._get_emergency_prompt_threshold(),
            ),
        )

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate prompt tokens from message content and tool metadata."""
        return sum(estimate_message_tokens(msg) for msg in messages)

    @staticmethod
    def _drop_oldest_history_message(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop one oldest history entry."""
        if not history:
            return history
        return history[1:]

    @staticmethod
    def _latest_compaction_system_messages(session: Session) -> list[str]:
        """Return the latest compaction summary as explicit system input."""
        entry = session.get_last_compaction()
        if entry and entry.summary:
            return [entry.summary]
        return []

    def _build_messages_with_prompt_budget(
        self,
        *,
        history: list[dict[str, Any]],
        current_message: str | None,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        memory_context: str | None = None,
        resume_notice: str | None = None,
        extra_system_messages: list[str] | None = None,
        trim_if_needed: bool = True,
    ) -> list[dict[str, Any]]:
        """Build initial messages and only trim history as an emergency fallback."""
        history_window = list(history)
        emergency_threshold = self._get_emergency_prompt_threshold()
        context_label = (
            f"{channel}:{chat_id}" if channel is not None and chat_id is not None else "unknown"
        )

        def _build(hw: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return self.context.build_messages(
                history=hw,
                current_message=current_message,
                skill_names=skill_names,
                media=media,
                channel=channel,
                chat_id=chat_id,
                memory_context=memory_context,
                resume_notice=resume_notice,
                extra_system_messages=extra_system_messages,
            )

        messages = _build(history_window)
        initial_estimate = self._estimate_prompt_tokens(messages)
        estimated = initial_estimate
        trimmed = 0

        if trim_if_needed and estimated > emergency_threshold and history_window:
            # Estimate tokens per message for batch trimming
            avg_tokens_per_msg = max(estimated / max(len(history_window), 1), 1)
            overshoot = estimated - emergency_threshold
            # Drop estimated number of messages in one batch (with 10% safety margin)
            batch_drop = min(
                int(overshoot / avg_tokens_per_msg * 1.1) + 1,
                len(history_window) - 1,
            )
            if batch_drop > 0:
                for _ in range(batch_drop):
                    history_window = self._drop_oldest_history_message(history_window)
                    trimmed += 1
                messages = _build(history_window)
                estimated = self._estimate_prompt_tokens(messages)

            # Fine-tune: drop one at a time if still over (should be few iterations)
            while estimated > emergency_threshold and history_window:
                history_window = self._drop_oldest_history_message(history_window)
                trimmed += 1
                messages = _build(history_window)
                estimated = self._estimate_prompt_tokens(messages)

        if trimmed:
            logger.error(
                "Emergency prompt trim dropped {} history message(s) for {} "
                "(estimate {} -> {}, emergency_threshold={})",
                trimmed,
                context_label,
                initial_estimate,
                estimated,
                emergency_threshold,
            )
        if trim_if_needed and estimated > emergency_threshold:
            logger.error(
                "Emergency prompt trim still over threshold for {}: estimate={} threshold={} "
                "(history_messages={})",
                context_label,
                estimated,
                emergency_threshold,
                len(history_window),
            )
        return messages

    @staticmethod
    def _is_provider_context_overflow_error(error: BaseException | str | None) -> bool:
        """Detect provider-specific context overflow failures."""
        if error is None:
            return False
        if isinstance(error, BaseException):
            text = str(error)
        else:
            text = str(error)
        if not text:
            return False
        normalized = text.lower()
        patterns = (
            "prompt is too long",
            "context window",
            "context length",
            "maximum context length",
            "too many input tokens",
            "input is too long",
            "exceeds the context limit",
        )
        return any(pattern in normalized for pattern in patterns)

    async def _run_structured_compaction(
        self,
        *,
        session: Session,
        channel: str | None,
        chat_id: str | None,
        input_tokens: int,
        context_window: int | None = None,
        reason: str,
        send_notices: bool = True,
    ) -> None:
        """Run the existing structured compaction flow synchronously."""
        if session.key in self._consolidating:
            logger.debug(
                "Skipping structured compaction for {}: consolidation already in progress",
                session.key,
            )
            return

        resolved_context_window = context_window or self._get_context_window_size()
        threshold = self._get_compaction_threshold()
        logger.info(
            "Structured compaction TRIGGERED for {} — reason: {}",
            session.key,
            reason,
        )
        compaction_started_at = time.monotonic()
        compaction_event = CompactionEvent(session.key)
        compaction_event.set_trigger(
            input_tokens=input_tokens,
            threshold=threshold,
            context_window=resolved_context_window,
            utilization_pct=(input_tokens / resolved_context_window * 100 if resolved_context_window else 0),
            total_messages=len(session.messages),
            last_consolidated=session.last_consolidated,
        )

        lock = self._get_consolidation_lock(session.key)
        self._consolidating.add(session.key)
        try:
            async with lock:
                if send_notices and channel is not None and chat_id is not None:
                    try:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=channel,
                            chat_id=chat_id,
                            content="⏳ *Compacting session* — summarizing older context to free up space...",
                            metadata={"_system_notice": True},
                        ))
                    except Exception:
                        logger.warning("Failed to send compaction-start notice")

                previous_entry = session.get_last_compaction()
                previous_boundary = previous_entry.first_kept_index if previous_entry else 0

                memory_md_chars = len(self.context.memory.read_long_term() or "")
                visible_start, visible_end = session.get_visible_bounds()
                compaction_event.set_pre_compaction_context(
                    system_prompt_chars=len(self.context.build_system_prompt() or ""),
                    memory_md_chars=memory_md_chars,
                    history_chars=0,
                    conversation_messages=max(0, visible_end - visible_start),
                    conversation_start_index=visible_start,
                )

                if input_tokens > 0:
                    self.sessions.set_usage_snapshot(
                        session,
                        total_input_tokens=int(input_tokens),
                        source="compaction_trigger",
                    )

                entry = await compact_session(
                    session=session,
                    provider=self.provider,
                    model=self.background_model,
                    session_manager=self.sessions,
                    context_window=resolved_context_window,
                    reserve_tokens=self._COMPACTION_RESERVE_TOKENS,
                    keep_recent_tokens=self._COMPACTION_KEEP_RECENT_TOKENS,
                    threshold_ratio=self._COMPACTION_THRESHOLD_RATIO,
                )

                if entry is None:
                    self._refresh_estimated_token_snapshot(session)
                    self.sessions.pop_compaction_plan(session)
                    compaction_event.finalize(
                        success=True,
                        error="structured no-op or summary failure",
                        total_duration_ms=int((time.monotonic() - compaction_started_at) * 1000),
                    )
                else:
                    plan_meta = self.sessions.pop_compaction_plan(session)
                    extract_start_raw = plan_meta.get("extract_start", previous_boundary)
                    extract_end_raw = plan_meta.get("extract_end", entry.first_kept_index)
                    try:
                        extract_start = int(extract_start_raw)
                    except (TypeError, ValueError):
                        extract_start = previous_boundary
                    try:
                        extract_end = int(extract_end_raw)
                    except (TypeError, ValueError):
                        extract_end = entry.first_kept_index
                    total_messages = session.get_message_count()
                    extract_start = max(0, min(extract_start, total_messages))
                    extract_end = max(extract_start, min(extract_end, total_messages))

                    extraction_ok = True
                    self._active_compaction_event = compaction_event
                    try:
                        extraction_ok = await self._consolidate_memory(
                            session,
                            extraction_range=(extract_start, extract_end),
                        )
                    finally:
                        self._active_compaction_event = None

                    if not extraction_ok:
                        logger.warning(
                            "Structured compaction memory extraction failed for {}",
                            session.key,
                        )

                    memory_compact_result = None
                    try:
                        memory_compact_result = await self.context.memory.compact_memory_md(
                            provider=self.provider,
                            model=self.background_model,
                        )
                        if memory_compact_result and memory_compact_result.get("success"):
                            logger.info(
                                "MEMORY.md compacted: {} -> {} tokens",
                                memory_compact_result["before_tokens"],
                                memory_compact_result["after_tokens"],
                            )
                        elif memory_compact_result:
                            logger.warning("MEMORY.md compaction attempted but did not produce an update")
                    except Exception:
                        logger.exception("MEMORY.md compaction failed (non-fatal)")

                    self._refresh_estimated_token_snapshot(session)
                    self.sessions.save_state(session)

                    compaction_event.set_post_compaction_context(
                        first_kept_index=entry.first_kept_index,
                        new_last_consolidated=session.last_consolidated,
                        keep_count=self._CONSOLIDATION_KEEP_COUNT,
                        visible_messages=session.get_visible_message_count(),
                        total_items_extracted=compaction_event.data["result"].get("total_items", 0)
                        if compaction_event.data.get("result")
                        else 0,
                        summary_length=len(entry.summary),
                        file_ops_read_count=len(entry.file_ops.get("read_files", [])),
                        file_ops_modified_count=len(entry.file_ops.get("modified_files", [])),
                        is_iterative_update=bool(entry.previous_summary),
                        cut_point_type=str(plan_meta.get("cut_point_type", "clean")),
                    )
                    if memory_compact_result:
                        compaction_event.data["memory_md_compaction"] = memory_compact_result
                    compaction_event.finalize(
                        success=True,
                        error=None if extraction_ok else "memory extraction failed",
                        total_duration_ms=int((time.monotonic() - compaction_started_at) * 1000),
                    )
                    if extraction_ok and send_notices and channel is not None and chat_id is not None:
                        try:
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=channel,
                                chat_id=chat_id,
                                content=(
                                    "⚙️ *Session compacted* — older context has been "
                                    "summarized. Continuing with your message..."
                                ),
                                metadata={"_system_notice": True},
                            ))
                        except Exception:
                            logger.warning("Failed to send compaction-complete notice")

                self._get_compaction_logger(session).write(compaction_event)
        except Exception:
            logger.exception("Structured inline compaction failed for {}", session.key)
            try:
                compaction_event.finalize(
                    success=False,
                    error="exception in structured inline compaction",
                    total_duration_ms=int((time.monotonic() - compaction_started_at) * 1000),
                )
                self._get_compaction_logger(session).write(compaction_event)
            except Exception:
                pass
        finally:
            self._consolidating.discard(session.key)
            self._prune_consolidation_lock(session.key, lock)

    async def _finalize_prompt_for_send(
        self,
        *,
        session: Session,
        channel: str | None,
        chat_id: str | None,
        build_messages: Callable[[bool], list[dict[str, Any]]],
        context_label: str,
    ) -> list[dict[str, Any]]:
        """Build, compact if needed, then apply emergency trim as a last resort."""
        messages = build_messages(False)
        estimated_tokens = self._estimate_prompt_tokens(messages)
        compaction_threshold = self._get_compaction_threshold()
        emergency_threshold = self._get_emergency_prompt_threshold()
        logger.info(
            "Prompt estimate for {}: tokens={} compaction_threshold={} emergency_threshold={}",
            context_label,
            estimated_tokens,
            compaction_threshold,
            emergency_threshold,
        )

        if estimated_tokens > compaction_threshold and session.key not in self._consolidating:
            await self._run_structured_compaction(
                session=session,
                channel=channel,
                chat_id=chat_id,
                input_tokens=estimated_tokens,
                context_window=self._get_context_window_size(),
                reason=f"current prompt estimate ({estimated_tokens} > {compaction_threshold})",
            )
            messages = build_messages(False)
            estimated_tokens = self._estimate_prompt_tokens(messages)
            logger.info(
                "Prompt estimate after compaction for {}: tokens={} emergency_threshold={}",
                context_label,
                estimated_tokens,
                emergency_threshold,
            )

        if estimated_tokens > emergency_threshold:
            messages = build_messages(True)
            estimated_tokens = self._estimate_prompt_tokens(messages)
            logger.info(
                "Prompt estimate after emergency trim for {}: tokens={} emergency_threshold={}",
                context_label,
                estimated_tokens,
                emergency_threshold,
            )

        return messages

    def _refresh_estimated_token_snapshot(self, session: Session) -> None:
        """Refresh token snapshot from visible prompt window after compaction."""
        context_window = self._get_context_window_size()
        visible_history, _ = session.get_history(
            max_messages=max(1, session.get_visible_message_count()),
            context_window=context_window,
            protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
        )
        estimated_tokens = int(self._estimate_prompt_tokens(visible_history))
        self._last_input_tokens[session.key] = estimated_tokens
        self.sessions.set_usage_snapshot(
            session,
            total_input_tokens=estimated_tokens,
            source="estimated_visible_history",
        )

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
    def _build_inbox_delivery(events: list[InboxEvent]) -> str:
        """Render pending inbox events as a persisted assistant history message.

        NOTE: Content is clipped to 8KB per event here even though inbox stores
        up to 16KB. Phase 2 should add configurable delivery limits and optional
        full-content retrieval via a tool call.
        """
        lines = ["[External Events]"]
        for evt in events:
            header = f"- [{evt.source} {evt.event_id}] {evt.summary}"
            if evt.content:
                content = evt.content[:8000]
                if len(evt.content) > 8000:
                    content += "\n... (truncated)"
                lines.append(f"{header}\n{content}")
            else:
                lines.append(header)
        return "\n\n".join(lines)

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
        rebuild_messages: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        session_key_token = None
        if session is not None:
            session_key_token = self._current_session_key_ctx.set(session.key)
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        checkpoint_cursor = len(messages)
        _length_continuations = 0
        _accumulated_content: list[str] = []

        # Circuit breaker: track consecutive identical tool calls that produce errors.
        # Only triggers when the model retries the EXACT same tool+args+error combo.
        _last_error_sig: str | None = None
        _consecutive_errors: int = 0
        _MAX_CONSECUTIVE_ERRORS = 5  # Raised from 3 — less aggressive
        _overflow_retry_used = False

        try:
            if session is not None and checkpoint_start is not None:
                start_index = max(0, checkpoint_start)
                if start_index < len(messages):
                    session.checkpoint(messages[start_index:])
                checkpoint_cursor = len(messages)

            while iteration < self.max_iterations:
                iteration += 1

                _api_start = time.monotonic()
                logger.debug("Calling provider.chat() — iteration {}/{}, {} messages", iteration, self.max_iterations, len(messages))
                chat_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": self.tools.get_definitions(),
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.reasoning_effort:
                    chat_kwargs["reasoning_effort"] = self.reasoning_effort
                try:
                    response = await self.provider.chat(**chat_kwargs)
                except Exception as exc:
                    if (
                        not _overflow_retry_used
                        and session is not None
                        and rebuild_messages is not None
                        and self._is_provider_context_overflow_error(exc)
                    ):
                        _overflow_retry_used = True
                        visible_history, _ = session.get_history(
                            max_messages=max(1, session.get_visible_message_count()),
                            context_window=self._get_context_window_size(),
                            protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                        )
                        overflow_tokens = max(
                            self._estimate_prompt_tokens(messages),
                            sum(estimate_message_tokens(msg) for msg in visible_history),
                            self._get_compaction_threshold() + 1,
                        )
                        logger.error(
                            "Provider rejected prompt for {} due to context overflow; compacting and retrying once "
                            "(estimate={})",
                            session.key,
                            overflow_tokens,
                        )
                        await self._run_structured_compaction(
                            session=session,
                            channel=None,
                            chat_id=None,
                            input_tokens=overflow_tokens,
                            reason="provider context overflow",
                            send_notices=False,
                        )
                        messages = rebuild_messages()
                        checkpoint_cursor = len(messages)
                        continue
                    raise
                _api_elapsed = time.monotonic() - _api_start
                logger.debug("provider.chat() returned in {:.1f}s — finish_reason={}, tool_calls={}", _api_elapsed, response.finish_reason, len(response.tool_calls) if response.tool_calls else 0)

                if (
                    response.finish_reason == "error"
                    and not _overflow_retry_used
                    and session is not None
                    and rebuild_messages is not None
                    and self._is_provider_context_overflow_error(response.content)
                ):
                    _overflow_retry_used = True
                    visible_history, _ = session.get_history(
                        max_messages=max(1, session.get_visible_message_count()),
                        context_window=self._get_context_window_size(),
                        protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                    )
                    overflow_tokens = max(
                        self._estimate_prompt_tokens(messages),
                        sum(estimate_message_tokens(msg) for msg in visible_history),
                        self._get_compaction_threshold() + 1,
                    )
                    logger.error(
                        "Provider reported prompt overflow for {} as an error response; compacting and retrying once "
                        "(estimate={})",
                        session.key,
                        overflow_tokens,
                    )
                    await self._run_structured_compaction(
                        session=session,
                        channel=None,
                        chat_id=None,
                        input_tokens=overflow_tokens,
                        reason="provider context overflow",
                        send_notices=False,
                    )
                    messages = rebuild_messages()
                    checkpoint_cursor = len(messages)
                    continue

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
                        self.sessions.set_usage_snapshot(
                            session,
                            total_input_tokens=int(total_input),
                        )
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
                        thinking_blocks=response.thinking_blocks,
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
                        thinking_blocks=response.thinking_blocks,
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

                        # Index tool result in evidence index
                        if session is not None and isinstance(result, str):
                            self._index_tool_result(
                                session,
                                tool_call.name,
                                result,
                                message_index=len(session.messages) - 1,
                                arguments=tool_call.arguments,
                            )

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
                    clean = self._strip_think(response.content)
                    # Don't persist error responses to session history — they can
                    # poison the context and cause permanent 400 loops (#1303).
                    if response.finish_reason == "error":
                        logger.error("LLM returned error: {}", (clean or "")[:200])
                        final_content = clean or "Sorry, I encountered an error calling the AI model."
                        break
                    if response.finish_reason == "length":
                        if clean:
                            _accumulated_content.append(clean)
                        if _length_continuations < 2:
                            _length_continuations += 1
                            logger.warning(
                                "Response truncated (finish_reason=length, continuation {}/2)",
                                _length_continuations,
                            )
                            messages = self.context.add_assistant_message(
                                messages,
                                clean,
                                reasoning_content=response.reasoning_content,
                                thinking_blocks=response.thinking_blocks,
                            )
                            messages.append({"role": "user", "content": "Continue from where you left off."})
                            if session is not None and checkpoint_cursor < len(messages):
                                session.checkpoint(messages[checkpoint_cursor:])
                                checkpoint_cursor = len(messages)
                            continue
                        logger.warning(
                            "Response truncated after {} continuation(s); returning accumulated partial output",
                            _length_continuations,
                        )

                    if _accumulated_content:
                        if response.finish_reason != "length" and clean:
                            _accumulated_content.append(clean)
                        merged = "\n".join(part for part in _accumulated_content if part)
                        final_content = merged or clean
                    else:
                        final_content = clean
                    messages = self.context.add_assistant_message(
                        messages,
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    if session is not None and checkpoint_cursor < len(messages):
                        session.checkpoint(messages[checkpoint_cursor:])
                        checkpoint_cursor = len(messages)
                        prune_method = getattr(self.sessions, "prune_old_tool_results", None)
                        if callable(prune_method):
                            prune_method(
                                session.key,
                                context_window=self._get_context_window_size(),
                                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                            )
                    break

            if final_content is None and iteration >= self.max_iterations:
                logger.warning("Max iterations ({}) reached", self.max_iterations)
                final_content = (
                    f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                    "without completing the task. You can try breaking the task into smaller steps."
                )

            return final_content, tools_used, messages
        finally:
            if session_key_token is not None:
                self._current_session_key_ctx.reset(session_key_token)

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

    def _get_processing_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._processing_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._processing_locks[session_key] = lock
        return lock

    @staticmethod
    def _lock_has_waiters(lock: asyncio.Lock) -> bool:
        # asyncio.Lock doesn't expose waiters publicly; this prevents pruning a
        # lock while another same-session task is queued.
        waiters = getattr(lock, "_waiters", None)
        return bool(waiters)

    def _prune_processing_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        if not lock.locked() and not self._lock_has_waiters(lock):
            self._processing_locks.pop(session_key, None)

    def _track_task(self, task: asyncio.Task, session_keys: set[str]) -> None:
        keys = tuple(dict.fromkeys(session_keys))
        for key in keys:
            self._active_tasks.setdefault(key, []).append(task)

        def _cleanup(done_task: asyncio.Task) -> None:
            self._untrack_task(done_task, keys)

        task.add_done_callback(_cleanup)

    def _untrack_task(self, task: asyncio.Task, session_keys: tuple[str, ...]) -> None:
        for key in session_keys:
            tasks = self._active_tasks.get(key)
            if not tasks:
                continue
            while task in tasks:
                tasks.remove(task)
            if not tasks:
                self._active_tasks.pop(key, None)

    @staticmethod
    def _resume_task_keys(session_key: str, channel: str, chat_id: str) -> set[str]:
        routed_key = f"{channel}:{chat_id}"
        return {session_key, routed_key}

    def _collect_resumable_sessions(self) -> list[tuple[str, str, str]]:
        sessions: list[tuple[str, str, str]] = []
        for item in self.sessions.list_sessions():
            key = item.get("key")
            if not key:
                continue
            if self.bus.inbound_size > 0:
                break

            session = self.sessions.get_or_create(key)
            state = session.detect_resume_state()
            if state == "mid_tool":
                self._normalize_interrupted_mid_tool_session(session)
                state = session.detect_resume_state()
            if state != "mid_loop":
                continue

            channel, chat_id = self._split_session_key(key)
            if key.startswith("cron:"):
                origin_channel = session.metadata.get("origin_channel")
                origin_chat_id = session.metadata.get("origin_chat_id")
                if isinstance(origin_channel, str) and origin_channel:
                    channel = origin_channel
                if isinstance(origin_chat_id, str) and origin_chat_id:
                    chat_id = origin_chat_id
            sessions.append((key, channel, chat_id))
        return sessions

    def _normalize_interrupted_mid_tool_session(self, session: Session) -> bool:
        """Convert a dangling assistant tool_call tail into explicit interruption records."""
        if session.detect_resume_state() != "mid_tool":
            return False

        tail = session.messages[-1] if session.messages else {}
        tool_calls = tail.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            return False

        timestamp = datetime.now().isoformat()
        entries: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id") or (tool_call.get("function") or {}).get("id")
            tool_name = (tool_call.get("function") or {}).get("name") or "unknown_tool"
            if not tool_call_id:
                continue
            entries.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id),
                    "name": str(tool_name),
                    "content": "[Tool execution interrupted during gateway restart]",
                    "timestamp": timestamp,
                }
            )

        if not entries:
            return False

        entries.append(
            {
                "role": "assistant",
                "content": (
                    "A previous tool execution was interrupted by a gateway restart. "
                    "The interrupted call was marked failed; waiting for a new user message."
                ),
                "timestamp": timestamp,
            }
        )
        session.checkpoint(entries)
        self.sessions.save_state(session)
        logger.warning(
            "Normalized interrupted mid_tool session {} into explicit interrupted tool results",
            session.key,
        )
        return True

    async def _resume_and_publish(self, session_key: str, channel: str, chat_id: str) -> bool:
        session = self.sessions.get_or_create(session_key)
        lock = self._get_processing_lock(session_key)
        try:
            async with lock:
                response = await self._resume_session(session, channel, chat_id)
            if response is not None:
                await self.bus.publish_outbound(response)
            return True
        except asyncio.CancelledError:
            logger.info("Auto-resume task cancelled for session {}", session_key)
            raise
        except Exception:
            logger.exception("Error auto-resuming interrupted session {}", session_key)
            return False
        finally:
            self._prune_processing_lock(session_key, lock)

    async def _resume_session(self, session: Session, channel: str, chat_id: str) -> OutboundMessage | None:
        """Resume an interrupted turn without a new inbound user message."""
        self._set_tool_context(channel, chat_id)
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        context_window = self._get_context_window_size()
        def _build_resume_prompt(trim_if_needed: bool) -> list[dict[str, Any]]:
            history, _ = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            return self._build_messages_with_prompt_budget(
                history=history,
                current_message=None,
                skill_names=self.skill_names,
                channel=channel,
                chat_id=chat_id,
                resume_notice=self._RESUME_SYSTEM_MESSAGE,
                extra_system_messages=self._latest_compaction_system_messages(session),
                trim_if_needed=trim_if_needed,
            )

        initial_messages = await self._finalize_prompt_for_send(
            session=session,
            channel=channel,
            chat_id=chat_id,
            context_label=f"resume:{session.key}",
            build_messages=_build_resume_prompt,
        )
        _, prune_result = session.get_history(
            max_messages=max(1, session.get_visible_message_count()),
            context_window=context_window,
            protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
        )
        self._get_context_logger(session).log_turn(
            built_messages=initial_messages,
            memory_context=None,
            resume_notice=self._RESUME_SYSTEM_MESSAGE,
            user_message_index=session.get_message_count(),
            prune_result=prune_result,
        )
        self._get_usage_logger(session).new_turn()
        self.subagents.set_skill_index(self.context.get_last_skills_summary())
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            session=session,
            checkpoint_start=None,
        )

        self._save_turn(session, all_msgs, len(initial_messages))
        self.sessions.save_state(session)

        suppress_final_reply = False
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                sent_targets = set(message_tool.get_turn_sends())
                suppress_final_reply = (channel, chat_id) in sent_targets

        if suppress_final_reply:
            logger.info(
                "Skipping resumed auto-reply because message tool already sent to {}:{} in this turn",
                channel,
                chat_id,
            )
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
        for key, channel, chat_id in self._collect_resumable_sessions():
            logger.info("Auto-resuming interrupted session {}", key)
            if await self._resume_and_publish(key, channel, chat_id):
                resumed += 1
        return resumed

    def _schedule_inflight_resumes(self) -> int:
        """Schedule auto-resume work in background tasks and return task count."""
        if self.bus.inbound_size > 0:
            logger.info("Skipping auto-resume: inbound queue already has pending messages")
            return 0

        scheduled = 0
        for key, channel, chat_id in self._collect_resumable_sessions():
            logger.info("Scheduling auto-resume for interrupted session {}", key)
            task = asyncio.create_task(
                self._resume_and_publish(key, channel, chat_id),
                name=f"resume-{key}",
            )
            self._track_task(task, self._resume_task_keys(key, channel, chat_id))
            scheduled += 1
        return scheduled

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        scheduled = self._schedule_inflight_resumes()
        if scheduled:
            logger.info("Auto-resume scheduled for {} interrupted session(s)", scheduled)
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
                self._track_task(task, {msg.session_key})

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = list(dict.fromkeys(self._active_tasks.pop(msg.session_key, [])))
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
        """Process a message under a per-session lock."""
        lock = self._get_processing_lock(msg.session_key)
        try:
            async with lock:
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
        finally:
            self._prune_processing_lock(msg.session_key, lock)

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
        for session_key in list(self._evidence_indexes):
            self._close_evidence_index(session_key)
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked() and not self._lock_has_waiters(lock):
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
            context_window = self._get_context_window_size()
            history, prune_result = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            def _build_system_prompt(trim_if_needed: bool) -> list[dict[str, Any]]:
                history, _ = session.get_history(
                    max_messages=max(1, session.get_visible_message_count()),
                    context_window=context_window,
                    protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                )
                return self._build_messages_with_prompt_budget(
                    history=history,
                    current_message=msg.content,
                    skill_names=self.skill_names,
                    channel=channel,
                    chat_id=chat_id,
                    memory_context=memory_context,
                    resume_notice=resume_notice,
                    extra_system_messages=self._latest_compaction_system_messages(session),
                    trim_if_needed=trim_if_needed,
                )

            initial_messages = await self._finalize_prompt_for_send(
                session=session,
                channel=channel,
                chat_id=chat_id,
                context_label=f"{channel}:{chat_id}",
                build_messages=_build_system_prompt,
            )
            history, prune_result = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            self._get_context_logger(session).log_turn(
                built_messages=initial_messages,
                memory_context=memory_context,
                resume_notice=resume_notice,
                user_message_index=session.get_message_count(),
                prune_result=prune_result,
            )
            self._get_usage_logger(session).new_turn()
            self.subagents.set_skill_index(self.context.get_last_skills_summary())
            turn_start = max(len(initial_messages) - 1, 0)
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages,
                session=session,
                checkpoint_start=turn_start,
                rebuild_messages=lambda: self._build_messages_with_prompt_budget(
                    history=session.get_history(
                        max_messages=max(1, session.get_visible_message_count()),
                        context_window=context_window,
                        protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                    )[0],
                    current_message=None,
                    skill_names=self.skill_names,
                    channel=channel,
                    chat_id=chat_id,
                    extra_system_messages=self._latest_compaction_system_messages(session),
                    trim_if_needed=True,
                ),
            )
            self._save_turn(session, all_msgs, turn_start)
            self.sessions.save_state(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        self._normalize_interrupted_mid_tool_session(session)
        metadata = msg.metadata or {}
        if key.startswith("cron:"):
            session.metadata.setdefault("origin_channel", msg.channel)
            session.metadata.setdefault("origin_chat_id", msg.chat_id)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.get_messages_slice()
                    if snapshot:
                        previous = session.get_last_compaction()
                        previous_summary = previous.summary if previous else None
                        summary = await generate_compaction_summary(
                            snapshot,
                            self.provider,
                            self.background_model,
                            previous_summary=previous_summary,
                        )
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        temp.append_compaction(
                            summary=summary,
                            first_kept_index=len(snapshot),
                            tokens_before=sum(estimate_message_tokens(item) for item in snapshot),
                            file_ops=extract_file_ops(snapshot),
                            previous_summary=previous_summary,
                        )
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

            self.sessions.get_inbox(session.key).clear()
            session.clear()
            self._last_input_tokens.pop(session.key, None)
            self.sessions.clear_usage_snapshot(session)
            self._close_evidence_index(session.key, delete_files=False)
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        # Token-based compaction: check if the session's last known input token count
        # exceeds 75% of the model's context window.
        _needs_compaction = False
        _compaction_reason = "none"
        _token_count = 0
        _context_window = self._get_context_window_size()
        if session.key not in self._consolidating:
            fresh_snapshot_tokens = _usage_snapshot_tokens(session)
            if fresh_snapshot_tokens is not None:
                _token_count = int(fresh_snapshot_tokens)
            else:
                _token_count = int(self._last_input_tokens.get(session.key, 0) or 0)

            baseline_messages, _ = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=_context_window,
                prune_tool_results=False,
            )
            pressure_messages, _ = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=_context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            baseline_tokens = sum(estimate_message_tokens(msg) for msg in baseline_messages)
            post_prune_tokens = sum(estimate_message_tokens(msg) for msg in pressure_messages)
            decision_tokens = _decision_tokens_with_pruning_ceiling(
                baseline_tokens=baseline_tokens,
                post_prune_tokens=post_prune_tokens,
                api_tokens_ceiling=fresh_snapshot_tokens,
                fallback_tokens=_token_count if _token_count > 0 else None,
            )
            if decision_tokens is not None:
                _token_count = int(decision_tokens)
            _needs_compaction = should_compact(
                pressure_messages,
                context_window=_context_window,
                reserve_tokens=self._COMPACTION_RESERVE_TOKENS,
                last_input_tokens=decision_tokens,
                threshold_ratio=self._COMPACTION_THRESHOLD_RATIO,
            )
            threshold = int((_context_window - self._COMPACTION_RESERVE_TOKENS) * self._COMPACTION_THRESHOLD_RATIO)
            logger.info(
                "Structured compaction check for {}: tokens={} threshold={} total_messages={} visible_messages={}",
                session.key,
                _token_count,
                threshold,
                session.get_message_count(),
                len(pressure_messages),
            )
            if _needs_compaction:
                if _token_count > 0:
                    _compaction_reason = f"tokens ({_token_count} > {threshold})"
                else:
                    _compaction_reason = "estimated token pressure"
        else:
            logger.debug(
                "Skipping compaction check for {}: consolidation already in progress",
                session.key,
            )

        if _needs_compaction:
            await self._run_structured_compaction(
                session=session,
                channel=msg.channel,
                chat_id=msg.chat_id,
                input_tokens=_token_count,
                context_window=_context_window,
                reason=_compaction_reason,
            )

        resume_notice = self._resume_notice_for_state(session.detect_resume_state())
        self._set_tool_context(msg.channel, msg.chat_id, metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        inbox = self.sessions.get_inbox(session.key)
        pending_events = inbox.drain()

        legacy_present = "recent_cron_actions" in session.metadata
        removed_state = (
            self.sessions.apply_state(
                session,
                metadata_remove=["recent_cron_actions"],
            )
            if legacy_present
            else {}
        )
        legacy_cron = removed_state.get("recent_cron_actions")
        if isinstance(legacy_cron, list) and legacy_cron:
            for idx, action in enumerate(legacy_cron):
                if not isinstance(action, dict):
                    continue
                raw_job_id = action.get("job_id", "unknown")
                job_id = str(raw_job_id) if raw_job_id is not None else "unknown"
                pending_events.append(
                    InboxEvent(
                        event_id=f"evt_legacy_{job_id}_{idx}",
                        source="legacy_cron",
                        summary=f"Cron job {job_id}",
                        content=str(action.get("response_preview") or ""),
                        occurred_at=str(action.get("timestamp") or ""),
                        source_meta=action,
                    )
                )

        if pending_events:
            delivered_text = self._build_inbox_delivery(pending_events)
            session.checkpoint(
                [
                    {
                        "role": "assistant",
                        "content": delivered_text,
                        "timestamp": datetime.now().isoformat(),
                    }
                ]
            )

        memory_context = await self._retrieve_memory_context(session, msg.content)
        history, prune_result = session.get_history(
            max_messages=max(1, session.get_visible_message_count()),
            context_window=_context_window,
            protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
        )
        def _build_live_prompt(trim_if_needed: bool) -> list[dict[str, Any]]:
            history, _ = session.get_history(
                max_messages=max(1, session.get_visible_message_count()),
                context_window=_context_window,
                protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
            )
            return self._build_messages_with_prompt_budget(
                history=history,
                current_message=msg.content,
                skill_names=self.skill_names,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_context=memory_context,
                resume_notice=resume_notice,
                extra_system_messages=self._latest_compaction_system_messages(session),
                trim_if_needed=trim_if_needed,
            )

        initial_messages = await self._finalize_prompt_for_send(
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            context_label=session.key,
            build_messages=_build_live_prompt,
        )
        history, prune_result = session.get_history(
            max_messages=max(1, session.get_visible_message_count()),
            context_window=_context_window,
            protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
        )

        # Tag the current user message with the platform message_id so it
        # persists in the session JSONL for future lookup / reply threading.
        _msg_id = metadata.get("message_id") if metadata else None
        if _msg_id:
            for _m in reversed(initial_messages):
                if _m.get("role") == "user":
                    _m["message_id"] = str(_msg_id)
                    break

        self._get_context_logger(session).log_turn(
            built_messages=initial_messages,
            memory_context=memory_context,
            resume_notice=resume_notice,
            user_message_index=session.get_message_count(),
            prune_result=prune_result,
        )
        self._get_usage_logger(session).new_turn()
        self.subagents.set_skill_index(self.context.get_last_skills_summary())
        turn_start = max(len(initial_messages) - 1, 0)

        streamed_progress_content = False

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        async def _tracked_progress(content: str, *, tool_hint: bool = False) -> None:
            nonlocal streamed_progress_content
            if content and not tool_hint:
                streamed_progress_content = True
            progress_callback = on_progress or _bus_progress
            await progress_callback(content, tool_hint=tool_hint)

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=_tracked_progress,
            session=session,
            checkpoint_start=turn_start,
            rebuild_messages=lambda: self._build_messages_with_prompt_budget(
                history=session.get_history(
                    max_messages=max(1, session.get_visible_message_count()),
                    context_window=_context_window,
                    protected_tools=set(self._PRUNE_PROTECTED_TOOLS),
                )[0],
                current_message=None,
                skill_names=self.skill_names,
                channel=msg.channel,
                chat_id=msg.chat_id,
                extra_system_messages=self._latest_compaction_system_messages(session),
                trim_if_needed=True,
            ),
        )

        self._save_turn(session, all_msgs, turn_start)
        self.sessions.save_state(session)

        suppress_reason: str | None = None
        if final_content is None and streamed_progress_content:
            suppress_reason = "streamed progress already delivered content for this turn"

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                sent_targets = set(message_tool.get_turn_sends())
                if (msg.channel, msg.chat_id) in sent_targets:
                    suppress_reason = (
                        f"message tool already sent to {msg.channel}:{msg.chat_id} in this turn"
                    )

        if suppress_reason is not None:
            logger.info("Skipping final auto-reply because {}", suppress_reason)
            return None

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=metadata,
        )

    def _save_turn(self, session: Session, messages: list[dict], start_index: int) -> None:
        """End-of-turn hook; messages are checkpointed during loop execution."""
        _ = (session, messages, start_index)

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
            return " ".join(parts).strip()
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        return str(content or "")

    async def _plan_hybrid_batches(
        self,
        *,
        start_index: int,
        end_index: int,
        all_messages: list[dict[str, Any]],
        consolidator: Any,
        llm_adapter: Any,
    ) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        """Plan token-aware consolidation batches for hybrid compaction."""
        window = list(all_messages[start_index:end_index])
        total_messages = len(window)
        if total_messages <= 0:
            return [], {
                "token_source": "none",
                "total_input_tokens": 0,
                "input_budget": 0,
                "context_window": 0,
                "output_budget": 0,
                "safety_margin": 0,
            }

        build_messages_fn = getattr(consolidator, "build_extraction_messages", None)
        if callable(build_messages_fn):
            extraction_messages = build_messages_fn(window)
        else:
            transcript = "\n".join(
                f"[{msg.get('role', 'unknown')}]: {self._message_content_to_text(msg.get('content'))}"
                for msg in window
            )
            extraction_messages = [{"role": "user", "content": transcript}]

        resolved_model = str(
            getattr(consolidator, "model", None)
            or self.background_model
            or self.model
            or ""
        )

        context_window = self._get_context_window_size()
        get_window_fn = getattr(llm_adapter, "get_context_window", None)
        if callable(get_window_fn):
            try:
                context_window = int(get_window_fn(resolved_model))
            except Exception:
                context_window = self._get_context_window_size()
        context_window = max(1, context_window)

        get_output_budget_fn = getattr(consolidator, "get_extraction_max_tokens", None)
        if callable(get_output_budget_fn):
            try:
                output_budget = int(get_output_budget_fn(window))
            except Exception:
                output_budget = int(getattr(consolidator, "extraction_max_tokens", 4096))
        else:
            output_budget = int(getattr(consolidator, "extraction_max_tokens", 4096))
        output_budget = max(256, output_budget)

        safety_margin = max(256, int(context_window * 0.10))
        input_budget = max(512, context_window - output_budget - safety_margin)

        total_input_tokens: int | None = None
        token_source = "count_tokens"
        count_fn = getattr(llm_adapter, "count_tokens", None)
        if callable(count_fn):
            try:
                total_input_tokens = await count_fn(
                    messages=extraction_messages,
                    model=resolved_model,
                    system=None,
                )
            except TypeError:
                total_input_tokens = await count_fn(extraction_messages, resolved_model, None)
            except Exception:
                total_input_tokens = None

        if total_input_tokens is None:
            token_source = "estimate_tokens"
            estimate_fn = getattr(llm_adapter, "estimate_tokens", None)
            text_payload = "\n".join(
                self._message_content_to_text(msg.get("content"))
                for msg in extraction_messages
            )
            if callable(estimate_fn):
                try:
                    total_input_tokens = int(estimate_fn(text_payload))
                except Exception:
                    total_input_tokens = max(1, len(text_payload) // 4)
            else:
                total_input_tokens = max(1, len(text_payload) // 4)

        total_input_tokens = max(0, int(total_input_tokens))
        if total_input_tokens <= input_budget:
            return [(start_index, end_index)], {
                "token_source": token_source,
                "total_input_tokens": total_input_tokens,
                "input_budget": input_budget,
                "context_window": context_window,
                "output_budget": output_budget,
                "safety_margin": safety_margin,
            }

        batch_count = max(2, math.ceil(total_input_tokens / input_budget))
        messages_per_batch = max(1, math.ceil(total_messages / batch_count))
        planned: list[tuple[int, int]] = []
        cursor = start_index
        while cursor < end_index:
            batch_end = min(cursor + messages_per_batch, end_index)
            planned.append((cursor, batch_end))
            cursor = batch_end

        return planned, {
            "token_source": token_source,
            "total_input_tokens": total_input_tokens,
            "input_budget": input_budget,
            "context_window": context_window,
            "output_budget": output_budget,
            "safety_margin": safety_margin,
        }

    async def _consolidate_memory(
        self,
        session,
        archive_all: bool = False,
        extraction_range: tuple[int, int] | None = None,
    ) -> bool:
        """Consolidate file memory and optionally graph memory. Returns file-memory success."""
        async with self._memory_file_lock:
            consolidation_cfg = (
                self._memory_graph_config.get("consolidation") or {}
                if isinstance(self._memory_graph_config, dict)
                else {}
            )
            engine = str(consolidation_cfg.get("engine") or "legacy").lower()
            explicit_range: tuple[int, int] | None = None
            if extraction_range is not None and not archive_all:
                try:
                    raw_start, raw_end = extraction_range
                    start = int(raw_start)
                    end = int(raw_end)
                except (TypeError, ValueError):
                    explicit_range = None
                else:
                    total_messages = session.get_message_count()
                    start = max(0, min(start, total_messages))
                    end = max(start, min(end, total_messages))
                    if end > start:
                        explicit_range = (start, end)

            if engine == "hybrid" and self._memory_module and self._memory_module.hybrid:
                try:
                    if not self._memory_module.initialized:
                        await self._memory_module.initialize()

                    keep_count = 0 if archive_all else self._CONSOLIDATION_KEEP_COUNT
                    logger.info(
                        "Hybrid consolidation for {}: archive_all={}, keep_count={}, "
                        "total_messages={}, last_consolidated={}",
                        session.key, archive_all, keep_count,
                        session.get_message_count(), session.last_consolidated,
                    )
                    if not archive_all:
                        if explicit_range is not None:
                            start_index, end_index = explicit_range
                            logger.info(
                                "Hybrid consolidation explicit range for {}: [{}:{}]",
                                session.key,
                                start_index,
                                end_index,
                            )
                        else:
                            total_messages = session.get_message_count()
                            if total_messages <= keep_count:
                                logger.info(
                                    "Hybrid consolidation no-op: total_messages ({}) <= keep_count ({})",
                                    total_messages, keep_count,
                                )
                                return True
                            if total_messages - session.last_consolidated <= 0:
                                logger.info(
                                    "Hybrid consolidation no-op: nothing unconsolidated",
                                )
                                return True
                            start_index = session.last_consolidated
                            end_index = total_messages - keep_count
                        if end_index <= start_index:
                            logger.info(
                                "Hybrid consolidation no-op: end_index ({}) <= start_index ({})",
                                end_index, start_index,
                            )
                            return True

                        planned_batches, plan_meta = await self._plan_hybrid_batches(
                            start_index=start_index,
                            end_index=end_index,
                            all_messages=session.messages,
                            consolidator=self._memory_module.consolidator,
                            llm_adapter=self._memory_module.consolidator.llm,
                        )
                        if not planned_batches:
                            logger.info("Hybrid consolidation no-op: no planned batches")
                            return True

                        logger.info(
                            "Hybrid consolidation plan for {}: {} batches, source={}, "
                            "input_tokens={} budget={} (context={} output={} safety={})",
                            session.key,
                            len(planned_batches),
                            plan_meta.get("token_source"),
                            plan_meta.get("total_input_tokens"),
                            plan_meta.get("input_budget"),
                            plan_meta.get("context_window"),
                            plan_meta.get("output_budget"),
                            plan_meta.get("safety_margin"),
                        )

                        completed_batches = 0
                        skipped_batches = 0
                        accumulated_entries: list[Any] = []
                        aborted_on_failure = False
                        _ce = getattr(self, "_active_compaction_event", None)
                        for batch_index, (batch_start, batch_end) in enumerate(planned_batches):
                            last_consolidated_before = session.last_consolidated
                            extraction_reason = (
                                "compaction"
                                if _ce is not None or explicit_range is None
                                else "manual"
                            )
                            compaction_event_id = None
                            logger.info(
                                "Hybrid consolidation batch {}/{} for {}: messages[{}:{}]",
                                batch_index + 1,
                                len(planned_batches),
                                session.key,
                                batch_start,
                                batch_end,
                            )
                            self.sessions.mark_extraction_batch_pending(
                                session,
                                batch_start=batch_start,
                                batch_end=batch_end,
                            )

                            result = None
                            last_candidate = None
                            last_error: str | None = None
                            _batch_ms = 0
                            _batch_total_t0 = time.monotonic()
                            for attempt in range(2):
                                _batch_t0 = time.monotonic()
                                try:
                                    candidate = await self._memory_module.hybrid.compact(
                                        session_key=session.key,
                                        messages=session.messages,
                                        start_index=batch_start,
                                        end_index=batch_end,
                                        agent_id=self.agent_id,
                                        skip_memory_rewrite=True,
                                    )
                                    last_candidate = candidate
                                    _batch_ms = int((time.monotonic() - _batch_t0) * 1000)
                                    if hasattr(candidate, "success") and not bool(
                                        getattr(candidate, "success")
                                    ):
                                        last_error = str(getattr(candidate, "error", "unknown error"))
                                        logger.warning(
                                            "Hybrid consolidation batch failed for {} at messages[{}:{}] "
                                            "(attempt {}/2): {}",
                                            session.key,
                                            batch_start,
                                            batch_end,
                                            attempt + 1,
                                            last_error,
                                        )
                                    else:
                                        result = candidate
                                        last_error = None
                                        break
                                except Exception as e:
                                    _batch_ms = int((time.monotonic() - _batch_t0) * 1000)
                                    last_error = str(e)
                                    logger.warning(
                                        "Hybrid consolidation batch exception for {} at messages[{}:{}] "
                                        "(attempt {}/2): {}",
                                        session.key,
                                        batch_start,
                                        batch_end,
                                        attempt + 1,
                                        e,
                                    )

                                if attempt == 0:
                                    logger.info(
                                        "Retrying hybrid batch for {} at messages[{}:{}]",
                                        session.key,
                                        batch_start,
                                        batch_end,
                                    )
                            _batch_total_ms = int((time.monotonic() - _batch_total_t0) * 1000)

                            if result is None:
                                skipped_batches += 1
                                skip_error = f"skipped after retry: {last_error or 'unknown error'}"
                                logger.warning(
                                    "Skipping hybrid batch for {} at messages[{}:{}] after retry failure",
                                    session.key,
                                    batch_start,
                                    batch_end,
                                )
                                if _ce:
                                    _ce.add_batch(
                                        batch_index=batch_index,
                                        msg_start=batch_start,
                                        msg_end=batch_end,
                                        transcript_chars=0,
                                        llm_response_chars=0,
                                        llm_response_preview=skip_error,
                                        items_extracted=0,
                                        success=False,
                                        error=skip_error,
                                        duration_ms=_batch_total_ms,
                                    )
                                failed_result = last_candidate
                                if failed_result is not None and not hasattr(failed_result, "error"):
                                    setattr(failed_result, "error", skip_error)
                                elif failed_result is None:
                                    failed_result = SimpleNamespace(success=False, error=skip_error)
                                self._record_extraction_event(
                                    session=session,
                                    batch_start=batch_start,
                                    batch_end=batch_end,
                                    last_consolidated_before=last_consolidated_before,
                                    batch_duration_ms=_batch_total_ms,
                                    result=failed_result,
                                    reason=extraction_reason,
                                    compaction_event_id=compaction_event_id,
                                )
                                self.sessions.mark_extraction_batch_failed(
                                    session,
                                    batch_start=batch_start,
                                    batch_end=batch_end,
                                    error=skip_error,
                                )
                                aborted_on_failure = True
                                break

                            batch_entries = getattr(result, "entries", None)
                            _batch_items = len(batch_entries) if isinstance(batch_entries, list) else 0
                            if isinstance(batch_entries, list) and batch_entries:
                                accumulated_entries.extend(batch_entries)

                            if _ce:
                                _ce.add_batch(
                                    batch_index=batch_index,
                                    msg_start=batch_start,
                                    msg_end=batch_end,
                                    transcript_chars=getattr(result, "transcript_chars", 0),
                                    llm_response_chars=getattr(result, "llm_response_chars", 0),
                                    llm_response_preview=(
                                        getattr(result, "llm_response_preview", "")
                                        or getattr(result, "error", "")
                                        or f"{_batch_items} items"
                                    ),
                                    items_extracted=_batch_items,
                                    success=bool(getattr(result, "success", True)),
                                    error=getattr(result, "error", None),
                                    duration_ms=_batch_total_ms,
                                )
                            self._record_extraction_event(
                                session=session,
                                batch_start=batch_start,
                                batch_end=batch_end,
                                last_consolidated_before=last_consolidated_before,
                                batch_duration_ms=_batch_total_ms,
                                result=result,
                                reason=extraction_reason,
                                compaction_event_id=compaction_event_id,
                            )

                            # Persist per-batch checkpoint so failures are resumable.
                            self.sessions.advance_last_consolidated(session, batch_end)
                            self.sessions.mark_extraction_batch_succeeded(
                                session,
                                batch_end=batch_end,
                            )
                            completed_batches += 1

                        if accumulated_entries and hasattr(self._memory_module.hybrid, "rewrite_memory_md"):
                            try:
                                await self._memory_module.hybrid.rewrite_memory_md(
                                    entries=accumulated_entries,
                                    session_key=session.key,
                                )
                            except Exception as e:
                                logger.warning("Hybrid run-level MEMORY.md rewrite failed: {}", e)

                        logger.info(
                            "Hybrid memory consolidation done: {} messages, batches={}, skipped={}, "
                            "last_consolidated={}{}",
                            session.get_message_count(),
                            completed_batches,
                            skipped_batches,
                            session.last_consolidated,
                            " (aborted on first failed batch)" if aborted_on_failure else "",
                        )
                        return True
                    else:
                        start_index = 0
                        end_index = session.get_message_count()
                        target_last_consolidated = 0

                    logger.info(
                        "Hybrid consolidation executing: messages[{}:{}] -> last_consolidated={}",
                        start_index, end_index, target_last_consolidated,
                    )
                    result = None
                    archive_error: str | None = None
                    for attempt in range(2):
                        try:
                            candidate = await self._memory_module.hybrid.compact(
                                session_key=session.key,
                                messages=session.messages,
                                start_index=start_index,
                                end_index=end_index,
                                agent_id=self.agent_id,
                            )
                            if hasattr(candidate, "success") and not bool(getattr(candidate, "success")):
                                archive_error = str(getattr(candidate, "error", "unknown error"))
                                logger.warning(
                                    "Hybrid archive-all consolidation failed for {} (attempt {}/2): {}",
                                    session.key,
                                    attempt + 1,
                                    archive_error,
                                )
                            else:
                                result = candidate
                                archive_error = None
                                break
                        except Exception as e:
                            archive_error = str(e)
                            logger.warning(
                                "Hybrid archive-all consolidation exception for {} (attempt {}/2): {}",
                                session.key,
                                attempt + 1,
                                e,
                            )
                        if attempt == 0:
                            logger.info("Retrying archive-all consolidation for {}", session.key)

                    if result is None:
                        logger.warning(
                            "Hybrid memory consolidation failed for {}: {}",
                            session.key,
                            archive_error or "unknown error",
                        )
                        return False
                    self.sessions.advance_last_consolidated(
                        session,
                        target_last_consolidated,
                    )
                    logger.info(
                        "Hybrid memory consolidation done: {} messages, last_consolidated={}",
                        session.get_message_count(),
                        session.last_consolidated,
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Hybrid memory consolidation failed: {e}")
                    return False

            graph_window_messages: list[dict[str, Any]] = []
            if self._memory_module and self._memory_module.consolidator:
                if archive_all:
                    graph_window_messages = session.get_messages_slice()
                elif explicit_range is not None:
                    start_index, end_index = explicit_range
                    graph_window_messages = session.get_messages_slice(start_index, end_index)
                else:
                    keep_count = self._CONSOLIDATION_KEEP_COUNT
                    total_messages = session.get_message_count()
                    if total_messages > keep_count and total_messages > session.last_consolidated:
                        start_index = session.last_consolidated
                        end_index = total_messages - keep_count
                        if end_index > start_index:
                            graph_window_messages = session.get_messages_slice(start_index, end_index)

            if explicit_range is not None and not archive_all:
                start_index, end_index = explicit_range
                if end_index <= start_index:
                    return True
                temp_session = Session(key=session.key)
                temp_session.messages = session.get_messages_slice(start_index, end_index)
                success = await MemoryStore(self.workspace).consolidate(
                    temp_session,
                    self.provider,
                    self.background_model,
                    archive_all=True,
                    keep_count=0,
                )
                if success:
                    self.sessions.advance_last_consolidated(
                        session,
                        max(session.last_consolidated, end_index),
                    )
            else:
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
                        agent_id=self.agent_id,
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
