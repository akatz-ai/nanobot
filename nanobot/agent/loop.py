"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
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

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        memory_graph_config: dict | None = None,
        skill_names: list[str] | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
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
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._memory_file_lock = asyncio.Lock()  # Serialize global memory file writes across sessions.
        self._memory_module = None
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
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    def _register_memory_graph_tools(self) -> None:
        """Register optional memory graph tools from adapter package."""
        if not self._memory_graph_config or not self._memory_graph_config.get("enabled"):
            return
        try:
            from agent_memory_nanobot import NanobotMemoryModule

            self._memory_module = NanobotMemoryModule(
                provider=self.provider,
                workspace=self.workspace,
                config=self._memory_graph_config,
            )
            for tool in self._memory_module.get_tools():
                self.tools.register(tool)
            logger.info("Memory graph tools registered")
        except ImportError:
            logger.warning("agent-memory-nanobot not installed, memory graph disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize memory graph: {e}")

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
        # Approximate prompt budget from runtime token cap and currently loaded turn history.
        max_output_tokens = max(int(self.max_tokens), 1)
        prompt_token_budget = max(256, int(max_output_tokens * 0.45))
        prompt_word_budget = max(120, int(prompt_token_budget * 0.75))

        history_words = sum(
            self._word_count_from_content(turn.get("content"))
            for turn in recent_turns
        )
        message_words = len((user_message or "").split())
        remaining = prompt_word_budget - history_words - message_words
        return max(80, remaining)

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

        if session is not None and checkpoint_start is not None:
            start_index = max(0, checkpoint_start)
            if start_index < len(messages):
                session.checkpoint(messages[start_index:])
            checkpoint_cursor = len(messages)

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

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

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if session is not None and checkpoint_cursor < len(messages):
                        session.checkpoint(messages[checkpoint_cursor:])
                        checkpoint_cursor = len(messages)
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

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=None,
            skill_names=self.skill_names,
            channel=channel,
            chat_id=chat_id,
            resume_notice=self._RESUME_SYSTEM_MESSAGE,
        )
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
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        resumed = await self.resume_inflight_sessions()
        if resumed:
            logger.info("Auto-resumed {} interrupted session(s)", resumed)
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
                except Exception as e:
                    logger.error("Error processing message: {}", e)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

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
            history = session.get_history(max_messages=self.memory_window)
            initial_messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                skill_names=self.skill_names,
                channel=channel,
                chat_id=chat_id,
                memory_context=memory_context,
                resume_notice=resume_notice,
            )
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
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        if await self._consolidate_memory(session):
                            # Persist consolidation checkpoint immediately to survive restarts.
                            self.sessions.save(session)
                except Exception:
                    logger.exception("Background consolidation failed for {}", session.key)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        resume_notice = self._resume_notice_for_state(session.detect_resume_state())
        self._set_tool_context(msg.channel, msg.chat_id, metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        memory_context = await self._retrieve_memory_context(session, msg.content)
        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            skill_names=self.skill_names,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            memory_context=memory_context,
            resume_notice=resume_notice,
        )
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

                    keep_count = 0 if archive_all else self.memory_window // 2
                    if not archive_all:
                        if len(session.messages) <= keep_count:
                            return True
                        if len(session.messages) - session.last_consolidated <= 0:
                            return True
                        start_index = session.last_consolidated
                        end_index = len(session.messages) - keep_count
                        if end_index <= start_index:
                            return True
                        target_last_consolidated = end_index
                    else:
                        start_index = 0
                        end_index = len(session.messages)
                        target_last_consolidated = 0

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
                    keep_count = self.memory_window // 2
                    if len(session.messages) > keep_count and len(session.messages) > session.last_consolidated:
                        start_index = session.last_consolidated
                        end_index = len(session.messages) - keep_count
                        if end_index > start_index:
                            graph_window_messages = list(session.messages[start_index:end_index])

            success = await MemoryStore(self.workspace).consolidate(
                session, self.provider, self.model,
                archive_all=archive_all, memory_window=self.memory_window,
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
