"""Message tool for sending messages to users."""

from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


class _TurnSendsProxy:
    """Backwards-compatible mutable view over per-task turn sends."""

    def __init__(self, storage: ContextVar[tuple[tuple[str, str], ...]]):
        self._storage = storage

    def append(self, target: tuple[str, str]) -> None:
        sends = list(self._storage.get(()))
        sends.append(target)
        self._storage.set(tuple(sends))

    def clear(self) -> None:
        self._storage.set(())

    def to_list(self) -> list[tuple[str, str]]:
        return list(self._storage.get(()))

    def __iter__(self):
        return iter(self.to_list())

    def __len__(self) -> int:
        return len(self._storage.get(()))


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel_ctx: ContextVar[str] = ContextVar(
            "message_tool_default_channel", default=default_channel
        )
        self._default_chat_id_ctx: ContextVar[str] = ContextVar(
            "message_tool_default_chat_id", default=default_chat_id
        )
        self._default_message_id_ctx: ContextVar[str | None] = ContextVar(
            "message_tool_default_message_id", default=default_message_id
        )
        self._turn_sends_ctx: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
            "message_tool_turn_sends", default=()
        )
        self._turn_sends = _TurnSendsProxy(self._turn_sends_ctx)

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel_ctx.set(channel)
        self._default_chat_id_ctx.set(chat_id)
        self._default_message_id_ctx.set(message_id)

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._turn_sends.clear()

    def get_turn_sends(self) -> list[tuple[str, str]]:
        """Get (channel, chat_id) targets sent in the current turn."""
        return self._turn_sends.to_list()

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel_ctx.get()
        chat_id = chat_id or self._default_chat_id_ctx.get()
        message_id = message_id or self._default_message_id_ctx.get()

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            }
        )

        try:
            await self._send_callback(msg)
            self._turn_sends.append((channel, chat_id))
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
