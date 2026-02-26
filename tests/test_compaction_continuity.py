"""Tests for post-compaction continuity context behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


class TestSnapshotContinuityContext:
    """Test _snapshot_continuity_context static method."""

    def test_extracts_last_exchanges(self):
        """Should extract the last N user/assistant text exchanges."""
        session = Session(key="test:snap")
        for i in range(10):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=3)
        assert result is not None
        assert "Question 7" in result
        assert "Answer 7" in result
        assert "Question 9" in result
        assert "Answer 9" in result
        # Should NOT include very old messages
        assert "Question 0" not in result

    def test_skips_tool_only_assistant_messages(self):
        """Assistant messages with no text content (tool-call-only) should be skipped."""
        session = Session(key="test:tool_only")
        session.add_message("user", "Do something")
        session.add_message("assistant", "")  # tool-call-only, no text
        session.add_message("tool", "tool result")
        session.add_message("assistant", "Here's what I found")
        session.add_message("user", "Thanks")
        session.add_message("assistant", "You're welcome")

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=4)
        assert result is not None
        assert "Here's what I found" in result
        assert "Thanks" in result
        assert "You're welcome" in result

    def test_skips_tool_messages(self):
        """Tool role messages should not appear in the snapshot."""
        session = Session(key="test:no_tools")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")
        session.add_message("tool", "some tool output")
        session.add_message("user", "Next question")
        session.add_message("assistant", "Next answer")

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=4)
        assert result is not None
        assert "tool output" not in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_returns_none_for_empty_session(self):
        """Empty session should return None."""
        session = Session(key="test:empty")
        result = AgentLoop._snapshot_continuity_context(session)
        assert result is None

    def test_returns_none_for_tool_only_session(self):
        """Session with only tool messages should return None."""
        session = Session(key="test:tools_only")
        session.add_message("tool", "result 1")
        session.add_message("tool", "result 2")
        result = AgentLoop._snapshot_continuity_context(session)
        assert result is None

    def test_starts_with_user_message(self):
        """The snapshot should start with a user message."""
        session = Session(key="test:start_user")
        session.add_message("assistant", "Previous context")
        session.add_message("user", "My question")
        session.add_message("assistant", "My answer")

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=4)
        assert result is not None
        # Should start with user, not the orphaned assistant
        assert result.startswith("**User:**")

    def test_truncates_long_messages(self):
        """Individual messages longer than 600 chars should be truncated."""
        session = Session(key="test:long")
        session.add_message("user", "Short question")
        session.add_message("assistant", "A" * 1000)

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=4)
        assert result is not None
        assert "..." in result
        # Should be truncated to ~600 chars per message
        assert len(result) < 1000

    def test_respects_max_chars_budget(self):
        """Total output should respect the max_chars budget."""
        session = Session(key="test:budget")
        for i in range(20):
            session.add_message("user", f"Question {'X' * 200} {i}")
            session.add_message("assistant", f"Answer {'Y' * 200} {i}")

        result = AgentLoop._snapshot_continuity_context(
            session, max_exchanges=10, max_chars=500
        )
        assert result is not None
        assert len(result) <= 600  # Some slack for formatting

    def test_max_exchanges_limits_user_messages(self):
        """max_exchanges should limit the number of user messages captured."""
        session = Session(key="test:max_ex")
        for i in range(10):
            session.add_message("user", f"Q{i}")
            session.add_message("assistant", f"A{i}")

        result = AgentLoop._snapshot_continuity_context(session, max_exchanges=2)
        assert result is not None
        # Should only have the last 2 user messages
        assert "Q8" in result
        assert "Q9" in result
        assert "Q7" not in result


class TestContinuityContextInjection:
    """Test that continuity context flows through build_messages correctly."""

    def test_continuity_context_injected_as_system_message(self, tmp_path):
        """Continuity context should appear as a system message in built messages."""
        from nanobot.agent.context import ContextBuilder

        # Create minimal workspace
        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "MEMORY.md").write_text("")

        builder = ContextBuilder(tmp_path)
        messages = builder.build_messages(
            history=[],
            current_message="Hello",
            continuity_context="**User:** What were we doing?\n\n**Assistant:** Working on tests.",
        )

        # Find the continuity system message
        continuity_msgs = [
            m for m in messages
            if m["role"] == "system" and "Session Continuity" in m.get("content", "")
        ]
        assert len(continuity_msgs) == 1
        assert "What were we doing?" in continuity_msgs[0]["content"]
        assert "compacted" in continuity_msgs[0]["content"].lower()

    def test_no_continuity_message_when_none(self, tmp_path):
        """No continuity system message should be added when context is None."""
        from nanobot.agent.context import ContextBuilder

        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "MEMORY.md").write_text("")

        builder = ContextBuilder(tmp_path)
        messages = builder.build_messages(
            history=[],
            current_message="Hello",
            continuity_context=None,
        )

        continuity_msgs = [
            m for m in messages
            if m["role"] == "system" and "Session Continuity" in m.get("content", "")
        ]
        assert len(continuity_msgs) == 0

    def test_continuity_appears_before_history(self, tmp_path):
        """Continuity context should appear before conversation history messages."""
        from nanobot.agent.context import ContextBuilder

        (tmp_path / "memory").mkdir(parents=True)
        (tmp_path / "memory" / "MEMORY.md").write_text("")

        history = [
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "old response"},
        ]
        builder = ContextBuilder(tmp_path)
        messages = builder.build_messages(
            history=history,
            current_message="new message",
            continuity_context="**User:** Recent chat\n\n**Assistant:** Recent reply",
        )

        # Find indices
        continuity_idx = None
        first_history_idx = None
        for i, m in enumerate(messages):
            if m["role"] == "system" and "Session Continuity" in m.get("content", ""):
                continuity_idx = i
            if m.get("content") == "old message":
                first_history_idx = i

        assert continuity_idx is not None
        assert first_history_idx is not None
        assert continuity_idx < first_history_idx


class TestCompactionContinuityIntegration:
    """Integration tests for the full compaction → continuity → injection flow."""

    @pytest.mark.asyncio
    async def test_compaction_runs_before_llm_call(self, tmp_path):
        """When threshold is exceeded, compaction should run inline before the LLM call."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test-model",
        )
        call_order: list[str] = []

        async def _fake_chat(*args, **kwargs):
            call_order.append("chat")
            return LLMResponse(content="ok", tool_calls=[])

        loop.provider.chat = AsyncMock(side_effect=_fake_chat)
        loop.tools.get_definitions = MagicMock(return_value=[])

        session = loop.sessions.get_or_create("cli:test")
        for i in range(15):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")
        loop.sessions.save(session)
        loop._last_input_tokens[session.key] = loop._compaction_token_threshold

        async def _fake_consolidate(sess, archive_all=False):
            call_order.append("consolidate")
            sess.last_consolidated = len(sess.messages) - 5
            return True

        loop._consolidate_memory = _fake_consolidate

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="hello")
        await loop._process_message(msg)

        assert call_order[:2] == ["consolidate", "chat"]

    @pytest.mark.asyncio
    async def test_continuity_passed_inline_not_metadata(self, tmp_path):
        """Compaction should pass continuity directly to build_messages in the same turn."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test-model",
        )
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        session = loop.sessions.get_or_create("cli:test")
        for i in range(15):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")
        loop.sessions.save(session)
        loop._last_input_tokens[session.key] = loop._compaction_token_threshold

        captured: dict[str, str | None] = {"continuity_context": None}
        original_build_messages = loop.context.build_messages

        def _capture_build_messages(*args, **kwargs):
            captured["continuity_context"] = kwargs.get("continuity_context")
            return original_build_messages(*args, **kwargs)

        loop.context.build_messages = _capture_build_messages  # type: ignore[method-assign]

        async def _fake_consolidate(sess, archive_all=False):
            sess.last_consolidated = len(sess.messages) - 5
            return True

        loop._consolidate_memory = _fake_consolidate

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="continue")
        await loop._process_message(msg)

        assert captured["continuity_context"] is not None
        assert "Question" in captured["continuity_context"]
        assert "continuity_context" not in session.metadata

    @pytest.mark.asyncio
    async def test_no_continuity_for_archive_all(self, tmp_path):
        """archive_all mode (/new) should NOT save continuity context."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test-model",
        )
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        session = Session(key="cli:test")
        for i in range(5):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")

        async def _fake_consolidate(sess, archive_all=False):
            if archive_all:
                # Should NOT have continuity context for archive_all
                assert "continuity_context" not in sess.metadata
            return True

        loop._consolidate_memory = _fake_consolidate

        # Simulate /new which uses archive_all=True
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
        await loop._process_message(msg)

    @pytest.mark.asyncio
    async def test_user_notifications_sent_for_successful_compaction(self, tmp_path):
        """A start + complete system notice should be sent on successful compaction."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test-model",
        )
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        session = loop.sessions.get_or_create("cli:test")
        for i in range(15):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")
        loop.sessions.save(session)
        loop._last_input_tokens[session.key] = loop._compaction_token_threshold

        async def _fake_consolidate(sess, archive_all=False):
            sess.last_consolidated = len(sess.messages) - 5
            return True

        loop._consolidate_memory = _fake_consolidate

        # Collect outbound messages
        outbound_messages = []
        original_publish = bus.publish_outbound

        async def _capture_publish(msg):
            outbound_messages.append(msg)
            await original_publish(msg)

        bus.publish_outbound = _capture_publish

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="hello")
        await loop._process_message(msg)

        # Find the compaction notice
        notices = [
            m for m in outbound_messages
            if isinstance(m, OutboundMessage)
            and m.metadata
            and m.metadata.get("_system_notice")
        ]
        assert len(notices) == 2
        assert "compacting session" in notices[0].content.lower()
        assert "compacted" in notices[1].content.lower()

    @pytest.mark.asyncio
    async def test_only_start_notice_sent_on_failed_compaction(self, tmp_path):
        """Failed compaction should send only the start notice."""
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus, provider=provider, workspace=tmp_path,
            model="test-model",
        )
        loop.provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        session = loop.sessions.get_or_create("cli:test")
        for i in range(15):
            session.add_message("user", f"msg{i}")
            session.add_message("assistant", f"resp{i}")
        loop.sessions.save(session)
        loop._last_input_tokens[session.key] = loop._compaction_token_threshold

        async def _failing_consolidate(sess, archive_all=False):
            return False

        loop._consolidate_memory = _failing_consolidate

        outbound_messages = []
        original_publish = bus.publish_outbound

        async def _capture_publish(msg):
            outbound_messages.append(msg)
            await original_publish(msg)

        bus.publish_outbound = _capture_publish

        msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="hello")
        await loop._process_message(msg)

        notices = [
            m for m in outbound_messages
            if isinstance(m, OutboundMessage)
            and m.metadata
            and m.metadata.get("_system_notice")
        ]
        assert len(notices) == 1
        assert "compacting session" in notices[0].content.lower()
