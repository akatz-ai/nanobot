"""Typed session record helpers for persisted session state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict


MessageRole = Literal["system", "user", "assistant", "tool"]


class CompactionPlanRecord(TypedDict, total=False):
    summary_start: int
    summary_end: int
    extract_start: int
    extract_end: int
    cut_point_type: str


@dataclass(frozen=True)
class InterruptedToolResultRecord:
    """Durable tool-result placeholder for an interrupted pending tool call."""

    tool_call_id: str
    tool_name: str
    timestamp: str
    content: str = "[Tool execution interrupted during gateway restart]"

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class InterruptedAssistantNoteRecord:
    """Assistant note appended after interrupted tool results."""

    timestamp: str
    content: str = (
        "A previous tool execution was interrupted by a gateway restart. "
        "The interrupted call was marked failed; waiting for a new user message."
    )

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "timestamp": self.timestamp,
        }


def make_interrupted_tool_records(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pending assistant ``tool_calls`` into explicit interruption records."""
    timestamp = datetime.now().isoformat()
    out: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id") or (tool_call.get("function") or {}).get("id")
        tool_name = (tool_call.get("function") or {}).get("name") or "unknown_tool"
        if not tool_call_id:
            continue
        out.append(
            InterruptedToolResultRecord(
                tool_call_id=str(tool_call_id),
                tool_name=str(tool_name),
                timestamp=timestamp,
            ).to_message()
        )
    if out:
        out.append(InterruptedAssistantNoteRecord(timestamp=timestamp).to_message())
    return out
