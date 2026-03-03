"""Programmatic tool-calling batch tool."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import io
import json
import textwrap
import time
import traceback
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

MAX_OUTPUT_CHARS = 20_000
MAX_TIMEOUT_SECONDS = 300

# The registry appends this hint to error results — useful for direct tool calls
# but noise inside batch scripts where the script handles errors itself.
_REGISTRY_HINT = "\n\n[Analyze the error above and try a different approach.]"

ALLOWED_BUILTINS = (
    # Exceptions — include all common ones so try/except works naturally
    "Exception",
    "AttributeError",
    "FileNotFoundError",
    "ImportError",
    "IndexError",
    "KeyError",
    "NameError",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    # Functions
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "chr",
    "dict",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hex",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
)

ALLOWED_IMPORTS = frozenset(
    {
        "asyncio",
        "base64",
        "collections",
        "datetime",
        "difflib",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "pathlib",
        "re",
        "textwrap",
        "urllib.parse",
    }
)

_REAL_IMPORT = builtins.__import__


class BatchTool(Tool):
    """Execute Python code that calls other nanobot tools as async functions."""

    name = "batch"
    description = (
        "Run a Python script that calls other tools as async functions. "
        "Use this instead of many individual tool calls when you need to: "
        "(1) call the same tool multiple times (e.g. fetch 5 URLs), "
        "(2) filter or process tool results before they reach your context "
        "(e.g. read 10 files but only return the ones containing a pattern), or "
        "(3) chain tool calls where later calls depend on earlier results. "
        "Tool results stay inside the script and only your print() output enters the conversation. "
        "Available async tools: read_file(path), write_file(path, content), "
        "edit_file(path, old_text, new_text), list_dir(path), "
        "exec_command(command, working_dir=None), web_search(query, count=5), "
        "web_fetch(url, extractMode='markdown', maxChars=50000). "
        "All tool functions are async — use `result = await tool_name(args)`. "
        "Use print() for output that should reach the conversation. "
        "Allowed imports: json, re, math, datetime, collections, itertools, "
        "functools, asyncio, hashlib, base64, pathlib, textwrap, difflib, urllib.parse. "
        "Use asyncio.gather() for parallel tool calls."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python script to execute. Tool functions are available as async "
                    "functions. Use print() for output that should reach the conversation."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Max execution time in seconds (default: 120, max: 300)",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_SECONDS,
            },
        },
        "required": ["code"],
    }

    def __init__(self, registry: ToolRegistry, timeout: int = 120):
        self.registry = registry
        self.default_timeout = max(1, min(timeout, MAX_TIMEOUT_SECONDS))

    async def execute(self, code: str, timeout: int | None = None, **kwargs: Any) -> str:
        _ = kwargs
        resolved_timeout = max(1, min(timeout or self.default_timeout, MAX_TIMEOUT_SECONDS))
        namespace, stats = self._build_namespace()

        start = time.perf_counter()
        line_count = code.count("\n") + 1 if code else 0
        logger.info("Batch: executing script ({} lines, timeout={}s)", line_count, resolved_timeout)
        try:
            output = await asyncio.wait_for(
                self._run_script(code, namespace),
                timeout=resolved_timeout,
            )
        except asyncio.TimeoutError:
            logger.info("Batch: timed out after {}s", resolved_timeout)
            return f"Error: Batch script timed out after {resolved_timeout} seconds"

        elapsed = time.perf_counter() - start
        logger.info(
            "Batch: completed in {:.2f}s, output={} chars, tool_calls={}",
            elapsed,
            len(output),
            stats["tool_calls"],
        )
        return output

    def _build_namespace(self) -> tuple[dict[str, Any], dict[str, int]]:
        tool_stats = {"tool_calls": 0}
        safe_builtins: dict[str, Any] = {
            name: getattr(builtins, name)
            for name in ALLOWED_BUILTINS
            if hasattr(builtins, name)
        }
        safe_builtins["__import__"] = self._guarded_import

        read_file_impl = self._make_tool_wrapper("read_file", tool_stats)
        write_file_impl = self._make_tool_wrapper("write_file", tool_stats)
        edit_file_impl = self._make_tool_wrapper("edit_file", tool_stats)
        list_dir_impl = self._make_tool_wrapper("list_dir", tool_stats)
        exec_impl = self._make_tool_wrapper("exec", tool_stats)
        web_search_impl = self._make_tool_wrapper("web_search", tool_stats)
        web_fetch_impl = self._make_tool_wrapper("web_fetch", tool_stats)

        async def read_file(path: str) -> str:
            return await read_file_impl(path=path)

        async def write_file(path: str, content: str) -> str:
            return await write_file_impl(path=path, content=content)

        async def edit_file(path: str, old_text: str, new_text: str) -> str:
            return await edit_file_impl(path=path, old_text=old_text, new_text=new_text)

        async def list_dir(path: str) -> str:
            return await list_dir_impl(path=path)

        async def exec_command(command: str, working_dir: str | None = None) -> str:
            params = {"command": command}
            if working_dir is not None:
                params["working_dir"] = working_dir
            return await exec_impl(**params)

        async def web_search(query: str, count: int = 5) -> str:
            return await web_search_impl(query=query, count=count)

        async def web_fetch(
            url: str,
            extractMode: str = "markdown",
            maxChars: int = 50_000,
        ) -> str:
            return await web_fetch_impl(
                url=url,
                extractMode=extractMode,
                maxChars=maxChars,
            )

        namespace = {
            "__name__": "__batch__",
            "__builtins__": safe_builtins,
            "read_file": read_file,
            "write_file": write_file,
            "edit_file": edit_file,
            "list_dir": list_dir,
            "exec_command": exec_command,
            "web_search": web_search,
            "web_fetch": web_fetch,
        }
        return namespace, tool_stats

    async def _run_script(self, code: str, namespace: dict[str, Any]) -> str:
        wrapped_code = self._wrap_code(code)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(wrapped_code, namespace, namespace)
                batch_main = namespace.get("__batch_main__")
                if not callable(batch_main):
                    return "Error: Failed to initialize batch script"
                await batch_main()
        except Exception:
            return self._truncate_output(
                self._compose_output(
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    trace=traceback.format_exc(),
                )
            )

        return self._truncate_output(
            self._compose_output(
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )
        )

    def _make_tool_wrapper(
        self,
        tool_name: str,
        stats: dict[str, int],
    ) -> Callable[..., Awaitable[str]]:
        async def _wrapper(**params: Any) -> str:
            stats["tool_calls"] += 1
            preview = json.dumps(params, ensure_ascii=False, default=str)
            if len(preview) > 240:
                preview = f"{preview[:240]}..."
            logger.info("Batch tool call: {}({})", tool_name, preview)
            result = await self.registry.execute(tool_name, params)
            # Strip the registry's LLM hint suffix — it's meant for direct tool
            # calls where the LLM sees the error, but inside batch the script
            # handles errors itself.
            if result.endswith(_REGISTRY_HINT):
                result = result[: -len(_REGISTRY_HINT)]
            return result

        return _wrapper

    def _guarded_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are not allowed in batch scripts")
        if name not in ALLOWED_IMPORTS:
            raise ImportError(f"Import '{name}' is not allowed in batch scripts")
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    @staticmethod
    def _wrap_code(code: str) -> str:
        script = textwrap.dedent(code).strip("\n")
        if not script:
            script = "pass"
        return "async def __batch_main__():\n" + textwrap.indent(script, "    ") + "\n"

    @staticmethod
    def _compose_output(stdout: str, stderr: str, trace: str | None = None) -> str:
        parts: list[str] = []
        if stdout.strip():
            parts.append(stdout.rstrip())
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr.rstrip()}")
        if trace:
            parts.append(trace.rstrip())
        if not parts:
            return "(no output)"
        return "\n\n".join(parts)

    @staticmethod
    def _truncate_output(output: str) -> str:
        if len(output) <= MAX_OUTPUT_CHARS:
            return output
        remaining = len(output) - MAX_OUTPUT_CHARS
        return output[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {remaining} more chars)"
