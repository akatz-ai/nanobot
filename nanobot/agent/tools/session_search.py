"""Session search tool — search tool results from the current conversation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nanobot.agent.tools.base import Tool


class SessionSearchTool(Tool):
    """Tool to search this conversation's tool outputs and results."""

    name = "session_search"
    description = (
        "Search this conversation's tool outputs and results. "
        "Use when you need to find something from earlier in this session — a file you read, "
        "a command you ran, an error message, code you wrote, or a web page you fetched. "
        "Returns matching excerpts with their original message index and tool name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "tool_filter": {
                "type": "string",
                "description": "Optional tool name filter",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(self, get_evidence_index=None):
        self._get_evidence_index = get_evidence_index

    async def execute(
        self,
        query: str,
        tool_filter: str | None = None,
        max_results: int = 5,
        **kwargs: Any,
    ) -> str:
        if self._get_evidence_index is None:
            return "Session search is not available."

        index = self._get_evidence_index()
        if index is None or index.count() == 0:
            return "No evidence indexed for this session yet."

        results = index.search(query, tool_filter=tool_filter, max_results=max_results)
        if not results:
            return f"No results found for: {query}"

        lines = [f"Found {len(results)} results:", ""]
        for position, result in enumerate(results, start=1):
            timestamp = _format_timestamp(result.get("timestamp"))
            header = f"[{position}] {result['tool_name']} (message #{result['message_index']}"
            if timestamp:
                header += f", {timestamp}"
            header += ")"

            excerpt = str(result.get("excerpt") or "").replace("\n", "\n  ")
            char_count = int(result.get("char_count") or 0)

            lines.append(header)
            lines.append(f"  {excerpt}")
            lines.append(f"  ({char_count:,} chars total)")
            lines.append("")

        return "\n".join(lines).rstrip()


def _format_timestamp(timestamp: str | None) -> str:
    """Format an ISO timestamp to a short human-readable string."""
    if not timestamp:
        return ""

    try:
        dt = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return ""

    hour = dt.strftime("%I").lstrip("0") or "0"
    return f"{dt.strftime('%b')} {dt.day} {hour}:{dt.strftime('%M %p')}"
