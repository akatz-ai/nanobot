from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.batch import BatchTool, MAX_OUTPUT_CHARS, _REGISTRY_HINT


def _make_tool(
    *,
    side_effect=None,
    return_value: str = "ok",
    timeout: int = 120,
) -> tuple[BatchTool, AsyncMock]:
    execute = AsyncMock(side_effect=side_effect, return_value=return_value)
    registry = SimpleNamespace(execute=execute)
    return BatchTool(registry=registry, timeout=timeout), execute


@pytest.mark.asyncio
async def test_basic_print_execution() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code='print("hello")')

    assert result == "hello"


@pytest.mark.asyncio
async def test_tool_wrapper_call_uses_registry() -> None:
    tool, execute = _make_tool(return_value="file-content")

    result = await tool.execute(code='value = await read_file("a.txt")\nprint(value)')

    assert result == "file-content"
    execute.assert_awaited_once_with("read_file", {"path": "a.txt"})


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_script() -> None:
    async def fake_execute(name: str, params: dict) -> str:
        return f"{name}:{params['path']}"

    tool, execute = _make_tool(side_effect=fake_execute)

    result = await tool.execute(
        code=(
            'a = await read_file("alpha.txt")\n'
            'b = await read_file("beta.txt")\n'
            "print(a)\n"
            "print(b)"
        )
    )

    assert result == "read_file:alpha.txt\nread_file:beta.txt"
    assert execute.await_count == 2


@pytest.mark.asyncio
async def test_stdout_capture_only_print_output_is_returned() -> None:
    tool, execute = _make_tool(return_value="secret-tool-output")

    result = await tool.execute(code='await read_file("hidden.txt")')

    assert result == "(no output)"
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_enforcement() -> None:
    async def slow_execute(name: str, params: dict) -> str:
        _ = (name, params)
        await asyncio.sleep(0.2)
        return "slow"

    tool, _ = _make_tool(side_effect=slow_execute)

    result = await tool.execute(
        code="while True:\n    await read_file('x.txt')",
        timeout=1,
    )

    assert result == "Error: Batch script timed out after 1 seconds"


@pytest.mark.asyncio
async def test_import_restrictions_block_os() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code="import os")

    assert "ImportError" in result
    assert "os" in result


@pytest.mark.asyncio
async def test_open_not_available_in_builtins() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code='open("x.txt", "w")')

    assert "NameError" in result
    assert "open" in result


@pytest.mark.asyncio
async def test_exceptions_return_traceback() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code='raise ValueError("boom")')

    assert "Traceback" in result
    assert "ValueError: boom" in result


@pytest.mark.asyncio
async def test_empty_output_returns_placeholder() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code="x = 1 + 2")

    assert result == "(no output)"


@pytest.mark.asyncio
async def test_output_is_truncated_at_limit() -> None:
    tool, _ = _make_tool()

    result = await tool.execute(code=f'print("x" * {MAX_OUTPUT_CHARS + 5000})')

    assert result.startswith("x" * 100)
    assert "(truncated, " in result
    assert len(result) > MAX_OUTPUT_CHARS


@pytest.mark.asyncio
async def test_exec_command_maps_to_exec_tool() -> None:
    tool, execute = _make_tool(return_value="/tmp")

    result = await tool.execute(code='print(await exec_command("pwd", working_dir="/tmp"))')

    assert result == "/tmp"
    execute.assert_awaited_once_with("exec", {"command": "pwd", "working_dir": "/tmp"})


@pytest.mark.asyncio
async def test_async_await_patterns_work() -> None:
    tool, _ = _make_tool(return_value="payload")

    result = await tool.execute(
        code=(
            "async def get_payload(path):\n"
            "    return await read_file(path)\n\n"
            'value = await get_payload("data.txt")\n'
            "print(value.upper())"
        )
    )

    assert result == "PAYLOAD"


@pytest.mark.asyncio
async def test_json_processing_with_allowed_json_module() -> None:
    tool, execute = _make_tool(return_value='{"score": 7, "tag": "ok"}')

    result = await tool.execute(
        code=(
            "import json\n"
            'raw = await read_file("result.json")\n'
            "data = json.loads(raw)\n"
            'print(f\"{data[\'tag\']}:{data[\'score\']}\")'
        )
    )

    assert result == "ok:7"
    execute.assert_awaited_once_with("read_file", {"path": "result.json"})


@pytest.mark.asyncio
async def test_web_fetch_wrapper_defaults() -> None:
    tool, execute = _make_tool(return_value='{"ok": true}')

    result = await tool.execute(code='print(await web_fetch("https://example.com"))')

    assert result == '{"ok": true}'
    execute.assert_awaited_once_with(
        "web_fetch",
        {"url": "https://example.com", "extractMode": "markdown", "maxChars": 50000},
    )


@pytest.mark.asyncio
async def test_realistic_multistep_workflow() -> None:
    async def fake_execute(name: str, params: dict) -> str:
        if name == "list_dir":
            return "alpha.py\nbeta.py\ngamma.py"
        if name == "read_file":
            data = {
                "alpha.py": "print('hello')\n",
                "beta.py": "TODO: fix this\n",
                "gamma.py": "x = 1\n# TODO later\n",
            }
            return data[params["path"]]
        return "unexpected"

    tool, _ = _make_tool(side_effect=fake_execute)

    result = await tool.execute(
        code=(
            'files = (await list_dir(".")).splitlines()\n'
            "matches = []\n"
            "for name in files:\n"
            "    text = await read_file(name)\n"
            '    if "TODO" in text:\n'
            "        matches.append(name)\n"
            "print(','.join(matches))"
        )
    )

    assert result == "beta.py,gamma.py"


@pytest.mark.asyncio
async def test_import_error_is_catchable() -> None:
    """ImportError must be in builtins so try/except works with guarded imports."""
    tool, _ = _make_tool()

    result = await tool.execute(
        code=(
            "try:\n"
            "    import os\n"
            "except ImportError as e:\n"
            '    print(f"caught: {e}")'
        )
    )

    assert "caught:" in result
    assert "os" in result


@pytest.mark.asyncio
async def test_name_error_is_catchable() -> None:
    """NameError must be in builtins for basic error handling."""
    tool, _ = _make_tool()

    result = await tool.execute(
        code=(
            "try:\n"
            "    x = undefined_var\n"
            "except NameError as e:\n"
            '    print(f"caught: {e}")'
        )
    )

    assert "caught:" in result
    assert "undefined_var" in result


@pytest.mark.asyncio
async def test_asyncio_gather_parallel_calls() -> None:
    """asyncio.gather should work for parallel tool calls."""
    call_order = []

    async def tracking_execute(name: str, params: dict) -> str:
        call_order.append(name)
        return f"result-{len(call_order)}"

    tool, _ = _make_tool(side_effect=tracking_execute)

    result = await tool.execute(
        code=(
            "import asyncio\n"
            "results = await asyncio.gather(\n"
            '    read_file("a.txt"),\n'
            '    read_file("b.txt"),\n'
            '    read_file("c.txt"),\n'
            ")\n"
            "for r in results:\n"
            "    print(r)"
        )
    )

    assert "result-1" in result
    assert "result-2" in result
    assert "result-3" in result
    assert len(call_order) == 3


@pytest.mark.asyncio
async def test_registry_hint_stripped_from_tool_errors() -> None:
    """The registry's '[Analyze the error...]' hint should be stripped in batch."""
    error_msg = "Error: File not found: missing.txt"

    async def fake_execute(name: str, params: dict) -> str:
        return error_msg + _REGISTRY_HINT

    tool, _ = _make_tool(side_effect=fake_execute)

    result = await tool.execute(
        code='result = await read_file("missing.txt")\nprint(result)'
    )

    assert result == error_msg
    assert "[Analyze" not in result
