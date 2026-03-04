"""Durable sidecar inbox for cross-session external events."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

_MAX_EVENT_CONTENT_CHARS = 16 * 1024


def _clip_content(text: str) -> str:
    return text[:_MAX_EVENT_CONTENT_CHARS]


@dataclass
class InboxEvent:
    """One external event queued for delivery into session history."""

    event_id: str
    source: str
    summary: str
    content: str
    occurred_at: str
    source_meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source: str,
        summary: str,
        content: str,
        source_meta: dict[str, Any] | None = None,
        occurred_at: str | None = None,
        event_id: str | None = None,
    ) -> InboxEvent:
        return cls(
            event_id=event_id or f"evt_{uuid.uuid4().hex[:12]}",
            source=source,
            summary=summary,
            content=_clip_content(content or ""),
            occurred_at=occurred_at or datetime.now().isoformat(),
            source_meta=dict(source_meta or {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InboxEvent:
        source_meta = data.get("source_meta")
        if not isinstance(source_meta, dict):
            source_meta = {}
        return cls(
            event_id=str(data.get("event_id") or f"evt_{uuid.uuid4().hex[:12]}"),
            source=str(data.get("source") or "unknown"),
            summary=str(data.get("summary") or ""),
            content=_clip_content(str(data.get("content") or "")),
            occurred_at=str(data.get("occurred_at") or ""),
            source_meta=source_meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id or f"evt_{uuid.uuid4().hex[:12]}",
            "source": self.source,
            "summary": self.summary,
            "content": _clip_content(self.content or ""),
            "occurred_at": self.occurred_at or datetime.now().isoformat(),
            "source_meta": dict(self.source_meta or {}),
        }


class SessionInbox:
    """Append-only durable inbox sidecar for a session JSONL file."""

    def __init__(self, session_path: Path):
        """Initialize with base session JSONL path (not inbox sidecar path)."""
        self._path = session_path.with_suffix(".inbox.jsonl")

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: InboxEvent) -> None:
        """Append a single event to the inbox sidecar."""
        payload = event.to_dict()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def drain(self) -> list[InboxEvent]:
        """Drain all pending events and clear the inbox file."""
        if not self._path.exists():
            return []

        draining_path = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex}.drain")
        try:
            self._path.replace(draining_path)
        except FileNotFoundError:
            return []

        # TODO(phase2): cursor-based ack — if we crash after drain (file truncated)
        # but before session.checkpoint(), events are lost. At current scale this is
        # acceptable; phase 2 should add cursor-based delivery with ack-after-checkpoint.
        events: list[InboxEvent] = []
        try:
            with open(draining_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed inbox line in {}", draining_path)
                        continue
                    if not isinstance(payload, dict):
                        continue
                    events.append(InboxEvent.from_dict(payload))
        finally:
            draining_path.unlink(missing_ok=True)

        return events

    def clear(self) -> None:
        """Remove the inbox sidecar file if it exists."""
        self._path.unlink(missing_ok=True)
