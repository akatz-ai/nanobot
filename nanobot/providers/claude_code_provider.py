"""Claude Code SDK provider implementation."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.litellm_provider import LiteLLMProvider

try:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.query import Query as SDKQuery
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
except ImportError:  # pragma: no cover - compatibility for older package name
    from claude_code_sdk import ClaudeCodeOptions as ClaudeAgentOptions  # type: ignore
    from claude_code_sdk._internal.query import Query as SDKQuery  # type: ignore
    from claude_code_sdk._internal.transport.subprocess_cli import (  # type: ignore
        SubprocessCLITransport,
    )


_SESSION_RE = re.compile(
    r"Channel:\s*(?P<channel>[^\n]+)\s*\nChat ID:\s*(?P<chat_id>[^\n]+)",
    re.IGNORECASE,
)


class ClaudeCodeProvider(LLMProvider):
    """Use Claude Code CLI via the claude-agent-sdk package."""

    def __init__(self, default_model: str = "claude-sonnet-4-5"):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self._session_ids: dict[str, str] = {}
        self._fallback_provider = LiteLLMProvider(default_model=default_model)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        requested_model = model or self.default_model
        if not self._is_claude_model(requested_model):
            return await self._fallback_provider.chat(
                messages=messages,
                tools=tools,
                model=requested_model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        model_name = self._strip_model_prefix(requested_model)
        clean_messages = self._sanitize_empty_content(messages)
        system_prompt, prompt = self._build_prompt(clean_messages, tools)
        session_key = self._derive_session_key(clean_messages, system_prompt)

        options = ClaudeAgentOptions(
            model=model_name,
            system_prompt=system_prompt or None,
            tools=[],
        )

        previous_session_id = self._session_ids.get(session_key)
        if previous_session_id:
            options.continue_conversation = True
            options.resume = previous_session_id

        try:
            content, finish_reason, usage, latest_session_id = await self._run_query(prompt, options)
            if latest_session_id:
                self._session_ids[session_key] = latest_session_id

            # Parse XML tool calls from Claude's text response
            clean_content, xml_tool_calls = self._parse_xml_tool_calls(content)
            return LLMResponse(
                content=clean_content,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=xml_tool_calls,
            )
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling Claude Code: {exc}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # XML tool-call protocol
    # ------------------------------------------------------------------
    _FUNC_BLOCK_RE = re.compile(
        r"<function_calls>\s*(.*?)\s*</function_calls>", re.DOTALL
    )
    _INVOKE_RE = re.compile(
        r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL
    )
    _PARAM_RE = re.compile(
        r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL
    )

    @classmethod
    def _parse_xml_tool_calls(cls, text: str) -> tuple[str, list[ToolCallRequest]]:
        """Extract ``<function_calls>`` XML blocks and convert to ToolCallRequest.

        Returns (clean_text, tool_calls).  Text before/after the XML blocks
        is preserved.  Each ``<invoke>`` becomes a ToolCallRequest that
        nanobot's agent loop will execute.
        """
        if not text:
            return text, []

        matches = list(cls._FUNC_BLOCK_RE.finditer(text))
        if not matches:
            return text, []

        tool_calls: list[ToolCallRequest] = []
        for match in matches:
            block = match.group(1)
            for invoke_match in cls._INVOKE_RE.finditer(block):
                name = invoke_match.group(1)
                params_block = invoke_match.group(2)
                arguments: dict[str, Any] = {}
                for param_match in cls._PARAM_RE.finditer(params_block):
                    value = param_match.group(2).strip()
                    # Try to parse JSON values (numbers, bools, objects)
                    try:
                        arguments[param_match.group(1)] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[param_match.group(1)] = value
                tool_calls.append(
                    ToolCallRequest(
                        id=f"xmltc_{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=arguments,
                    )
                )

        # Keep text outside the XML blocks
        clean = cls._FUNC_BLOCK_RE.sub("", text).strip()
        return clean, tool_calls

    # ------------------------------------------------------------------
    # Tool definition formatting
    # ------------------------------------------------------------------
    _TOOL_CALLING_INSTRUCTIONS = """\

## Tool Calling

You have access to tools. To call one or more tools, output this EXACT XML format
(do NOT wrap it in markdown code fences):

<function_calls>
<invoke name="tool_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</function_calls>

Rules:
- Output the <function_calls> XML block DIRECTLY in your response — no ``` fences
- You may include text before the XML block to explain what you're doing
- You may call multiple tools by including multiple <invoke> elements
- After your tool calls are executed, you will receive the results and should provide a final answer
- Only call tools that are listed below
"""

    @staticmethod
    def _format_tools_for_prompt(tools: list[dict[str, Any]] | None) -> str:
        """Convert OpenAI-format tool definitions to a readable text block."""
        if not tools:
            return ""
        lines = ["\n## Available Tools\n"]
        for tool in tools:
            fn = tool.get("function", tool)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            properties = params.get("properties", {})
            required = set(params.get("required", []))

            lines.append(f"### {name}")
            if desc:
                lines.append(f"{desc}")
            if properties:
                lines.append("**Parameters:**")
                for pname, pinfo in properties.items():
                    req_tag = " *(required)*" if pname in required else ""
                    ptype = pinfo.get("type", "")
                    pdesc = pinfo.get("description", "")
                    parts = [f"- `{pname}`{req_tag}"]
                    if ptype:
                        parts.append(f"({ptype})")
                    if pdesc:
                        parts.append(f"— {pdesc}")
                    lines.append(" ".join(parts))
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        if model.startswith("claude-code/") or model.startswith("claude_code/"):
            return model.split("/", 1)[1]
        return model

    @staticmethod
    def _is_claude_model(model: str) -> bool:
        model_lower = model.lower()
        if model_lower.startswith("claude-code/") or model_lower.startswith("claude_code/"):
            return True
        if "/" in model_lower:
            prefix = model_lower.split("/", 1)[0]
            if prefix in {"anthropic", "claude"}:
                return True
        return "claude" in model_lower

    async def _run_query(
        self,
        prompt: str,
        options: Any,
    ) -> tuple[str, str, dict[str, int], str | None]:
        transport = SubprocessCLITransport(prompt=prompt, options=options)
        query_engine = SDKQuery(transport=transport, is_streaming_mode=True)

        content_parts: list[str] = []
        content: str | None = None
        finish_reason = "stop"
        usage: dict[str, int] = {}
        latest_session_id: str | None = None

        await transport.connect()
        try:
            await query_engine.start()
            await query_engine.initialize()

            request = {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }
            await transport.write(json.dumps(request) + "\n")
            await transport.end_input()

            async for message in query_engine.receive_messages():
                msg_type = message.get("type")

                if msg_type == "assistant":
                    blocks = (message.get("message") or {}).get("content") or []
                    for block in blocks:
                        if not isinstance(block, dict) or block.get("type") != "text":
                            continue
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
                    continue

                if msg_type == "result":
                    result = message.get("result")
                    if isinstance(result, str) and result:
                        content = result
                    finish_reason = "error" if message.get("is_error") else "stop"
                    sid = message.get("session_id")
                    if isinstance(sid, str) and sid:
                        latest_session_id = sid
                    usage = self._parse_usage(message.get("usage"))
                    continue

                if msg_type == "error":
                    finish_reason = "error"
                    error_text = message.get("error")
                    if isinstance(error_text, str) and error_text:
                        content = error_text
                    continue

                # Ignore auxiliary stream events such as rate_limit_event.
                continue

            final_content = content or "".join(content_parts).strip()
            if not final_content and finish_reason == "error":
                final_content = "Claude Code returned an error."
            return final_content, finish_reason, usage, latest_session_id
        finally:
            await query_engine.close()

    def _build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        system_parts: list[str] = []
        transcript: list[str] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = self._content_to_text(message.get("content"))

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "assistant":
                if content:
                    transcript.append(f"Assistant: {content}")
                for tc in message.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name") or "tool"
                    args = fn.get("arguments")
                    if args:
                        transcript.append(f"Assistant tool call {name}: {args}")
                continue

            if role == "tool":
                tool_name = message.get("name") or "tool"
                call_id = message.get("tool_call_id")
                suffix = f" ({call_id})" if call_id else ""
                if content:
                    transcript.append(f"Tool {tool_name}{suffix} result:\n{content}")
                continue

            if content:
                transcript.append(f"User: {content}" if role == "user" else f"{role.title()}: {content}")

        prompt = (
            "Use the conversation context below and provide the next assistant reply.\n\n"
            + "\n\n".join(transcript).strip()
            + "\n\nAssistant:"
        ).strip()
        if not transcript:
            prompt = "Please provide a helpful assistant response."

        system_prompt = "\n\n".join(part for part in system_parts if part).strip()

        # Append tool definitions and calling instructions
        if tools:
            system_prompt += self._TOOL_CALLING_INSTRUCTIONS
            system_prompt += self._format_tools_for_prompt(tools)

        return system_prompt, prompt

    def _derive_session_key(self, messages: list[dict[str, Any]], system_prompt: str) -> str:
        for message in messages:
            value = message.get("session_key")
            if isinstance(value, str) and value:
                return value

        match = _SESSION_RE.search(system_prompt)
        if match:
            channel = match.group("channel").strip()
            chat_id = match.group("chat_id").strip()
            if channel and chat_id:
                return f"{channel}:{chat_id}"

        first_user = ""
        for message in messages:
            if message.get("role") == "user":
                first_user = self._content_to_text(message.get("content")).strip()
                if first_user:
                    break

        digest_src = f"{system_prompt}\n{first_user}".encode("utf-8")
        digest = hashlib.sha1(digest_src).hexdigest()[:16]
        return f"claude-code:{digest}"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                if item_type == "image_url":
                    url = (item.get("image_url") or {}).get("url")
                    if isinstance(url, str) and url:
                        parts.append(f"[image: {url}]")
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    @staticmethod
    def _parse_usage(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        out: dict[str, int] = {}
        mapping = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "prompt_tokens": "prompt_tokens",
            "completion_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }
        for src_key, dst_key in mapping.items():
            value = usage.get(src_key)
            if isinstance(value, int):
                out[dst_key] = value
        if "total_tokens" not in out and {"prompt_tokens", "completion_tokens"} <= out.keys():
            out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        return out
