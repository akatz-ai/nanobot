from __future__ import annotations

import re
from typing import Any

import pytest

from nanobot.channels.discord import DiscordChannel
from nanobot.config.schema import DiscordConfig


class _DummyBus:
    async def publish_inbound(self, message: Any) -> None:
        return None


class _FakeHTTPResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeHTTPClient:
    def __init__(self, content: bytes):
        self._content = content

    async def get(self, url: str) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(self._content)


async def _process_attachment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    monkeypatch.setenv("HOME", str(tmp_path))
    channel = DiscordChannel(config=DiscordConfig(), bus=_DummyBus())
    channel._http = _FakeHTTPClient(content)

    captured: dict[str, Any] = {}

    async def _capture_handle_message(**kwargs: Any) -> None:
        captured.update(kwargs)

    async def _noop_start_typing(channel_id: str) -> None:
        return None

    monkeypatch.setattr(channel, "_handle_message", _capture_handle_message)
    monkeypatch.setattr(channel, "_start_typing", _noop_start_typing)

    payload = {
        "id": "message-1",
        "author": {"id": "user-1", "bot": False},
        "channel_id": "channel-1",
        "attachments": [
            {
                "id": "file-1",
                "url": "https://example.test/file",
                "filename": filename,
                "size": len(content),
            }
        ],
    }

    await channel._handle_message_create(payload)
    return captured


@pytest.mark.asyncio
async def test_discord_large_text_attachment_uses_structured_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    text_bytes = ("line\n" * 3000).encode("utf-8")
    captured = await _process_attachment(
        monkeypatch,
        tmp_path,
        filename="big.txt",
        content=text_bytes,
    )
    body = captured["content"]
    assert re.search(
        r"^\[File: big\.txt \(\d+KB, ~3,000 lines\) saved to .+\]\n",
        body,
    )
    assert "This is a large text file." in body
    assert "Use exec with grep, head, tail, or wc to explore it" in body


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,content", [
    ("small.txt", b"short text\n"),
    ("photo.png", b"\x89PNG\r\n\x1a\n" + (b"x" * 2048)),
    ("photo.jpg", b"\xff\xd8\xff" + (b"x" * 2048)),
    ("archive.zip", b"PK\x03\x04" + (b"x" * 2048)),
    ("doc.pdf", b"%PDF" + (b"x" * 2048)),
])
async def test_discord_non_large_text_attachments_keep_default_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    filename: str,
    content: bytes,
) -> None:
    captured = await _process_attachment(
        monkeypatch,
        tmp_path,
        filename=filename,
        content=content,
    )
    assert captured["content"] == f"[attachment: {captured['media'][0]}]"
