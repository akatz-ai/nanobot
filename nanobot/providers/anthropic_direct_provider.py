"""LLM provider that calls Anthropic Messages API directly with OAuth token."""

from __future__ import annotations

import json
from typing import Any

import httpx
import json_repair
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14"
SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

_FINISH_MAP = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


class AnthropicDirectProvider(LLMProvider):
    """Provider that calls Anthropic Messages API directly using OAuth token."""

    def __init__(self, oauth_token: str, default_model: str = "claude-opus-4-6"):
        super().__init__(api_key=oauth_token)
        self.oauth_token = oauth_token
        self.default_model = default_model
        if "/" in self.default_model:
            self.default_model = self.default_model.split("/", 1)[1]

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.oauth_token}",
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_BETA,
        }

    @staticmethod
    def _coerce_tool_args(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json_repair.loads(raw_args)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except Exception:
                return {"raw": raw_args}
        return {"value": raw_args}

    @staticmethod
    def _assistant_content_blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
            return blocks
        return []

    @staticmethod
    def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop orphaned tool results/calls that would cause API errors.

        This is a lightweight safety net at the provider level.  The primary
        sanitization lives in ``SessionManager``, but messages built from
        other paths (e.g. system messages, cron) also benefit from this.
        """
        # Collect all tool_call IDs offered by assistant messages.
        offered: set[str] = set()
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    tc_id = tc.get("id") or (tc.get("function") or {}).get("id")
                    if tc_id:
                        offered.add(tc_id)

        # Collect IDs that have both an offer and a result.
        answered: set[str] = set()
        for m in messages:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                if tc_id and tc_id in offered:
                    answered.add(tc_id)

        cleaned: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "tool":
                if m.get("tool_call_id") not in offered:
                    logger.warning(
                        "Provider: dropping orphaned tool result (id={})",
                        m.get("tool_call_id"),
                    )
                    continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                kept = [
                    tc for tc in m["tool_calls"]
                    if (tc.get("id") or (tc.get("function") or {}).get("id"))
                    in answered
                ]
                if kept != m["tool_calls"]:
                    m = dict(m)
                    if kept:
                        m["tool_calls"] = kept
                    else:
                        m.pop("tool_calls", None)
                    if not m.get("content") and not m.get("tool_calls"):
                        continue
            cleaned.append(m)
        return cleaned

    @staticmethod
    def _tool_result_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Convert OpenAI-style messages/tools to Anthropic Messages API format."""
        system_parts: list[str] = [SYSTEM_PREFIX]
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                continue

            if role == "assistant":
                blocks = self._assistant_content_blocks(content)
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": self._coerce_tool_args(fn.get("arguments", "{}")),
                        }
                    )
                if blocks:
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": self._tool_result_content(content),
                            }
                        ],
                    }
                )
                continue

            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": content})
                continue

            if isinstance(content, list):
                blocks: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        image_url = (item.get("image_url") or {}).get("url", "")
                        if image_url.startswith("data:"):
                            header, data = image_url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
                        elif image_url:
                            blocks.append({"type": "text", "text": f"[image: {image_url}]"})
                    elif item.get("type") == "text":
                        blocks.append({"type": "text", "text": item.get("text", "")})
                if blocks:
                    anthropic_messages.append({"role": "user", "content": blocks})
                continue

            if content is not None:
                anthropic_messages.append({"role": "user", "content": str(content)})

        system_blocks, anthropic_messages = self._apply_cache_control(system_parts, anthropic_messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
            "system": system_blocks,
            "messages": anthropic_messages,
        }

        if tools:
            anthropic_tools = []
            for tool_def in tools:
                fn = tool_def.get("function", tool_def)
                anthropic_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            body["tools"] = anthropic_tools

        return body

    def _apply_cache_control(
        self,
        system_parts: list[str],
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply Anthropic prompt caching directives."""
        system_blocks: list[dict[str, Any]] = []
        for i, part in enumerate(system_parts):
            block: dict[str, Any] = {"type": "text", "text": part}
            if i == len(system_parts) - 1:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)

        new_messages = [dict(msg) for msg in messages]
        if len(new_messages) >= 2:
            target_idx = len(new_messages) - 2
            msg = new_messages[target_idx]
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                new_content = list(content)
                last_block = new_content[-1]
                if isinstance(last_block, dict):
                    new_content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
                    msg["content"] = new_content

        return system_blocks, new_messages

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        resolved_model = model or self.default_model
        if "/" in resolved_model:
            resolved_model = resolved_model.split("/", 1)[1]

        clean_messages = self._sanitize_empty_content(messages)
        clean_messages = self._sanitize_tool_pairs(clean_messages)
        body = self._build_body(clean_messages, tools, resolved_model, max_tokens, temperature)

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    headers=self._build_headers(),
                    json=body,
                )

            if resp.status_code in (401, 403):
                # Try re-reading credentials file first (may have been refreshed externally)
                from nanobot.providers.anthropic_auth import get_oauth_token, refresh_oauth_token
                fresh_token = get_oauth_token()
                if fresh_token and fresh_token != self.oauth_token:
                    self.oauth_token = fresh_token
                    logger.info("Loaded fresh token from credentials file, retrying...")
                else:
                    # Try OAuth refresh
                    fresh_token = refresh_oauth_token()
                    if fresh_token:
                        self.oauth_token = fresh_token
                        logger.info("Token refreshed after {}, retrying...", resp.status_code)

                if fresh_token:
                    async with httpx.AsyncClient(timeout=300) as retry_client:
                        resp = await retry_client.post(
                            ANTHROPIC_API_URL,
                            headers=self._build_headers(),
                            json=body,
                        )
                    if resp.status_code == 200:
                        return self._parse_response(resp.json())
                    logger.error("Retry after refresh also failed ({})", resp.status_code)

                logger.error("OAuth token rejected ({}) and refresh failed.", resp.status_code)
                return LLMResponse(
                    content="OAuth token expired/revoked and auto-refresh failed. Run `claude login` to refresh.",
                    finish_reason="error",
                )

            if resp.status_code != 200:
                error_text = resp.text
                logger.error("Anthropic API error {}: {}", resp.status_code, error_text[:500])
                return LLMResponse(
                    content=f"Anthropic API error: {resp.status_code} - {error_text[:300]}",
                    finish_reason="error",
                )

            return self._parse_response(resp.json())
        except Exception as e:
            logger.error("AnthropicDirectProvider error: {}", e)
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse Anthropic Messages API response."""
        content_text: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = _FINISH_MAP.get(stop_reason, "stop")
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cache_read = usage_data.get("cache_read_input_tokens", 0)
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)
        # Anthropic's input_tokens only reports non-cached tokens.
        # Total input = input_tokens + cache_read + cache_creation (per Anthropic docs).
        total_input = input_tokens + cache_read + cache_creation
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_input + output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        }

        return LLMResponse(
            content="\n".join(content_text) if content_text else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def get_default_model(self) -> str:
        return self.default_model
