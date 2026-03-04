"""Programmatic tool-calling batch tool."""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import io
import json
import keyword
import textwrap
import time
import traceback
from types import MappingProxyType
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

MAX_OUTPUT_CHARS = 20_000
MAX_TIMEOUT_SECONDS = 300
TASK_CLEANUP_TIMEOUT_SECONDS = 2.0

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

COMMON_TOOL_ALIASES: dict[str, str] = {
    "read_file": "read_file",
    "write_file": "write_file",
    "edit_file": "edit_file",
    "list_dir": "list_dir",
    "exec_command": "exec",
    "web_search": "web_search",
    "web_fetch": "web_fetch",
}

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

BLOCKED_INTROSPECTION_ATTRS = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__dict__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__getattribute__",
        "gi_frame",
        "cr_frame",
        "ag_frame",
        "tb_frame",
        "f_globals",
        "f_builtins",
    }
)

_INTROSPECTION_HELPERS = frozenset({"getattr", "hasattr", "setattr", "delattr"})
_BLOCKED_ATTR_ERROR = "Access to attribute '{name}' is not allowed in batch scripts"
_RESERVED_NAMESPACE_NAMES = frozenset(ALLOWED_BUILTINS) | frozenset(
    {"call_tool"} | {name.split(".")[0] for name in ALLOWED_IMPORTS}
)


class _SafeAsyncCallable:
    """Callable wrapper that blocks private attribute introspection."""

    __slots__ = ("_name", "_impl")

    def __init__(self, name: str, impl: Callable[..., Awaitable[str]]):
        self._name = name
        self._impl = impl

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        impl = object.__getattribute__(self, "_impl")
        return await impl(*args, **kwargs)

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_name")
        return f"<batch-tool {name}>"

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{name!r} is not available in batch scripts")
        return object.__getattribute__(self, name)


class _AsyncioProxy:
    """Expose asyncio with tracked create_task and limited introspection."""

    __slots__ = ("_module", "_create_task", "_ensure_future", "_wrap_loop")

    def __init__(
        self,
        module: Any,
        create_task_impl: Callable[..., asyncio.Task[Any]],
        ensure_future_impl: Callable[..., Any],
        loop_wrapper: Callable[[Any], Any],
    ) -> None:
        self._module = module
        self._create_task = create_task_impl
        self._ensure_future = ensure_future_impl
        self._wrap_loop = loop_wrapper

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{name!r} is not available in batch scripts")
        if name == "create_task":
            return object.__getattribute__(self, "_create_task")
        if name == "ensure_future":
            return object.__getattribute__(self, "_ensure_future")
        if name in ("get_running_loop", "get_event_loop"):
            module = object.__getattribute__(self, "_module")
            getter = getattr(module, name)
            wrap_loop = object.__getattribute__(self, "_wrap_loop")

            def _wrapped_get_loop(*args: Any, **kwargs: Any) -> Any:
                return wrap_loop(getter(*args, **kwargs))

            return _wrapped_get_loop
        module = object.__getattribute__(self, "_module")
        return getattr(module, name)


class _AsyncioLoopProxy:
    """Expose event loop with tracked create_task and limited introspection."""

    __slots__ = ("_loop", "_track_task")

    def __init__(
        self,
        loop: Any,
        track_task: Callable[[asyncio.Task[Any]], asyncio.Task[Any]],
    ) -> None:
        self._loop = loop
        self._track_task = track_task

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{name!r} is not available in batch scripts")
        if name == "create_task":
            loop = object.__getattribute__(self, "_loop")
            track_task = object.__getattribute__(self, "_track_task")

            def _wrapped_create_task(
                coro: Awaitable[Any],
                *,
                name: str | None = None,
                context: Any = None,
            ) -> asyncio.Task[Any]:
                create_kwargs: dict[str, Any] = {}
                if name is not None:
                    create_kwargs["name"] = name
                if context is not None:
                    create_kwargs["context"] = context
                try:
                    task = loop.create_task(coro, **create_kwargs)
                except TypeError:
                    create_kwargs.pop("context", None)
                    task = loop.create_task(coro, **create_kwargs)
                return track_task(task)

            return _wrapped_create_task
        loop = object.__getattribute__(self, "_loop")
        return getattr(loop, name)


class _SandboxVisitor(ast.NodeVisitor):
    """Block obvious introspection routes to recover unsafe builtins/import."""

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in BLOCKED_INTROSPECTION_ATTRS:
            raise ValueError(
                _BLOCKED_ATTR_ERROR.format(name=node.attr)
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _INTROSPECTION_HELPERS:
            attr_name: str | None = None
            if len(node.args) >= 2:
                attr_name = self._const_string(node.args[1])
            if attr_name is None:
                for kw in node.keywords:
                    if kw.arg == "name":
                        attr_name = self._const_string(kw.value)
                        break
            if attr_name in BLOCKED_INTROSPECTION_ATTRS:
                raise ValueError(
                    _BLOCKED_ATTR_ERROR.format(name=attr_name)
                )
        self.generic_visit(node)

    @staticmethod
    def _const_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


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
        "Other registered tools are exposed as async functions when their names are valid Python identifiers; "
        "for everything else, use call_tool(name, **params). "
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
        namespace, stats, runtime = self._build_namespace()

        start = time.perf_counter()
        line_count = code.count("\n") + 1 if code else 0
        logger.info("Batch: executing script ({} lines, timeout={}s)", line_count, resolved_timeout)
        timed_out = False
        try:
            output = await asyncio.wait_for(
                self._run_script(code, namespace),
                timeout=resolved_timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            logger.info("Batch: timed out after {}s", resolved_timeout)
            output = f"Error: Batch script timed out after {resolved_timeout} seconds"
        finally:
            runtime["active"] = False
            await self._cancel_created_tasks(runtime["created_tasks"])

        if timed_out:
            return output
        elapsed = time.perf_counter() - start
        logger.info(
            "Batch: completed in {:.2f}s, output={} chars, tool_calls={}",
            elapsed,
            len(output),
            stats["tool_calls"],
        )
        return output

    def _build_namespace(
        self,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        tool_stats = {"tool_calls": 0}
        created_tasks: set[asyncio.Task[Any]] = set()
        runtime: dict[str, Any] = {
            "created_tasks": created_tasks,
            "active": True,
        }

        def _track_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
            created_tasks.add(task)
            task.add_done_callback(created_tasks.discard)
            return task

        def _tracked_create_task(
            coro: Awaitable[Any],
            *,
            name: str | None = None,
            context: Any = None,
        ) -> asyncio.Task[Any]:
            create_kwargs: dict[str, Any] = {}
            if name is not None:
                create_kwargs["name"] = name
            if context is not None:
                create_kwargs["context"] = context
            try:
                task = asyncio.create_task(coro, **create_kwargs)
            except TypeError:
                create_kwargs.pop("context", None)
                task = asyncio.create_task(coro, **create_kwargs)
            return _track_task(task)

        def _tracked_ensure_future(
            awaitable: Awaitable[Any],
            *,
            loop: Any = None,
        ) -> Any:
            future = asyncio.ensure_future(awaitable, loop=loop)
            if isinstance(future, asyncio.Task):
                _track_task(future)
            return future

        safe_asyncio = _AsyncioProxy(
            asyncio,
            _tracked_create_task,
            _tracked_ensure_future,
            lambda loop: _AsyncioLoopProxy(loop, _track_task),
        )
        guarded_import = self._make_guarded_import(safe_asyncio)

        def _guard_attr_name(attr_name: Any) -> None:
            if isinstance(attr_name, str) and attr_name in BLOCKED_INTROSPECTION_ATTRS:
                raise AttributeError(_BLOCKED_ATTR_ERROR.format(name=attr_name))

        _attr_missing = object()

        def _safe_getattr(obj: Any, name: str, default: Any = _attr_missing) -> Any:
            _guard_attr_name(name)
            if default is _attr_missing:
                return getattr(obj, name)
            return getattr(obj, name, default)

        def _safe_hasattr(obj: Any, name: str) -> bool:
            _guard_attr_name(name)
            return hasattr(obj, name)

        def _safe_setattr(obj: Any, name: str, value: Any) -> None:
            _guard_attr_name(name)
            setattr(obj, name, value)

        safe_builtins_data: dict[str, Any] = {
            name: getattr(builtins, name)
            for name in ALLOWED_BUILTINS
            if hasattr(builtins, name)
        }
        safe_builtins_data["getattr"] = _safe_getattr
        safe_builtins_data["hasattr"] = _safe_hasattr
        safe_builtins_data["setattr"] = _safe_setattr
        safe_builtins_data["__import__"] = guarded_import
        safe_builtins = MappingProxyType(safe_builtins_data)

        async def call_tool_impl(name: str, **params: Any) -> str:
            if not runtime["active"]:
                raise RuntimeError(
                    "Batch script has finished; detached tasks cannot call tools"
                )
            return await self._call_tool(name, params, tool_stats)

        namespace: dict[str, Any] = {
            "__name__": "__batch__",
            "__builtins__": safe_builtins,
            "call_tool": _SafeAsyncCallable("call_tool", call_tool_impl),
        }

        for tool_name in sorted(self._registry_tool_names()):
            if tool_name == self.name:
                continue
            namespace_name = self._namespace_name_for_tool(tool_name)
            if (
                not namespace_name
                or namespace_name in namespace
                or namespace_name in _RESERVED_NAMESPACE_NAMES
            ):
                continue

            async def _dynamic_wrapper(
                *args: Any,
                _tool_name: str = tool_name,
                **params: Any,
            ) -> str:
                if args:
                    raise TypeError(
                        f"{_tool_name}() only accepts keyword arguments in batch scripts"
                    )
                return await call_tool_impl(_tool_name, **params)

            namespace[namespace_name] = _SafeAsyncCallable(namespace_name, _dynamic_wrapper)

        async def read_file(path: str) -> str:
            return await call_tool_impl("read_file", path=path)

        async def write_file(path: str, content: str) -> str:
            return await call_tool_impl("write_file", path=path, content=content)

        async def edit_file(path: str, old_text: str, new_text: str) -> str:
            return await call_tool_impl("edit_file", path=path, old_text=old_text, new_text=new_text)

        async def list_dir(path: str) -> str:
            return await call_tool_impl("list_dir", path=path)

        async def exec_command(command: str, working_dir: str | None = None) -> str:
            params = {"command": command}
            if working_dir is not None:
                params["working_dir"] = working_dir
            return await call_tool_impl("exec", **params)

        async def web_search(query: str, count: int = 5) -> str:
            return await call_tool_impl("web_search", query=query, count=count)

        async def web_fetch(
            url: str,
            extractMode: str = "markdown",
            maxChars: int = 50_000,
        ) -> str:
            return await call_tool_impl(
                "web_fetch",
                url=url,
                extractMode=extractMode,
                maxChars=maxChars,
            )

        namespace["read_file"] = _SafeAsyncCallable("read_file", read_file)
        namespace["write_file"] = _SafeAsyncCallable("write_file", write_file)
        namespace["edit_file"] = _SafeAsyncCallable("edit_file", edit_file)
        namespace["list_dir"] = _SafeAsyncCallable("list_dir", list_dir)
        namespace["exec_command"] = _SafeAsyncCallable("exec_command", exec_command)
        namespace["web_search"] = _SafeAsyncCallable("web_search", web_search)
        namespace["web_fetch"] = _SafeAsyncCallable("web_fetch", web_fetch)

        return namespace, tool_stats, runtime

    async def _run_script(self, code: str, namespace: dict[str, Any]) -> str:
        validation_error = self._validate_script(code)
        if validation_error:
            return f"Error: {validation_error}"

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

    async def _call_tool(self, tool_name: str, params: dict[str, Any], stats: dict[str, int]) -> str:
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

    async def _cancel_created_tasks(self, created_tasks: set[asyncio.Task[Any]]) -> None:
        pending = [task for task in list(created_tasks) if not task.done()]
        if not pending:
            return
        logger.info("Batch: cancelling {} detached task(s)", len(pending))
        for task in pending:
            task.cancel()
        done, still_pending = await asyncio.wait(
            pending,
            timeout=TASK_CLEANUP_TIMEOUT_SECONDS,
        )
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if still_pending:
            logger.warning(
                "Batch: {} detached task(s) did not stop after {:.1f}s",
                len(still_pending),
                TASK_CLEANUP_TIMEOUT_SECONDS,
            )
        created_tasks.clear()

    def _make_guarded_import(
        self,
        safe_asyncio: _AsyncioProxy,
    ) -> Callable[..., Any]:
        def _guarded_import(
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
            if name == "asyncio":
                return safe_asyncio
            return _REAL_IMPORT(name, globals, locals, fromlist, level)

        return _guarded_import

    @staticmethod
    def _validate_script(code: str) -> str | None:
        script = textwrap.dedent(code).strip("\n")
        if not script:
            return None
        try:
            parsed = ast.parse(script, mode="exec")
        except SyntaxError:
            return None
        try:
            _SandboxVisitor().visit(parsed)
        except ValueError as exc:
            return str(exc)
        return None

    def _registry_tool_names(self) -> set[str]:
        names = set(COMMON_TOOL_ALIASES.values())
        registry_names = getattr(self.registry, "tool_names", None)
        if isinstance(registry_names, (list, tuple, set, frozenset)):
            names.update(name for name in registry_names if isinstance(name, str))
        elif registry_names is not None and not isinstance(registry_names, str):
            with contextlib.suppress(TypeError):
                names.update(name for name in registry_names if isinstance(name, str))
        return names

    @staticmethod
    def _namespace_name_for_tool(tool_name: str) -> str | None:
        if tool_name.isidentifier() and not keyword.iskeyword(tool_name):
            return tool_name
        sanitized = tool_name.replace("-", "_").replace(".", "_")
        if sanitized.isidentifier() and not keyword.iskeyword(sanitized):
            return sanitized
        return None

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
