from __future__ import annotations

import pytest

from nanobot.agent.tools.filesystem import READ_FILE_ADVISORY_THRESHOLD, ReadFileTool


@pytest.mark.asyncio
async def test_read_file_large_content_appends_advisory(tmp_path) -> None:
    original = "abc123\n" * 5000
    path = tmp_path / "large.txt"
    path.write_text(original, encoding="utf-8")

    tool = ReadFileTool()
    result = await tool.execute(str(path))

    assert result.startswith(original)
    assert f"⚠️ This file is {len(original):,} characters." in result
    assert "For future reads, consider using exec with grep, head, or tail" in result


@pytest.mark.asyncio
async def test_read_file_small_content_has_no_advisory(tmp_path) -> None:
    original = "small file"
    path = tmp_path / "small.txt"
    path.write_text(original, encoding="utf-8")

    tool = ReadFileTool()
    result = await tool.execute(str(path))

    assert result == original
    assert "For future reads, consider using exec with grep, head, or tail" not in result


@pytest.mark.asyncio
async def test_read_file_advisory_does_not_truncate_content(tmp_path) -> None:
    original = "start\n" + ("z" * (READ_FILE_ADVISORY_THRESHOLD + 100)) + "\nend"
    path = tmp_path / "no-truncate.txt"
    path.write_text(original, encoding="utf-8")

    tool = ReadFileTool()
    result = await tool.execute(str(path))

    assert original in result
    assert result.startswith(original)
    assert result[len(original):].startswith("\n\n---\n⚠️")
