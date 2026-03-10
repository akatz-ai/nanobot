"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Any, AsyncGenerator
from uuid import uuid4

import httpx
from loguru import logger

from oauth_cli_kit import get_token as get_codex_token
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.content import content_to_text

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
_SUPPORTED_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_CODEX_MAX_RETRIES = 3


class CodexAPIError(RuntimeError):
    """Structured error for Codex HTTP/API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        response_status: str | None = None,
        raw_error: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.error_type = error_type
        self.error_code = error_code
        self.request_id = request_id
        self.response_status = response_status
        self.raw_error = raw_error


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(self, default_model: str = "openai-codex/gpt-5.1-codex"):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        request_id = str(uuid4())
        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages, tools),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        normalized_effort = _normalize_reasoning_effort(reasoning_effort, model)
        if normalized_effort is not None:
            body["reasoning"] = {"effort": normalized_effort}

        if tools:
            body["tools"] = _convert_tools(tools)

        payload_stats = _payload_stats(
            model=model,
            system_prompt=system_prompt,
            input_items=input_items,
            tools=tools,
            body=body,
        )

        url = DEFAULT_CODEX_URL

        try:
            content, tool_calls, finish_reason, usage = await _request_codex_with_retries(
                url=url,
                headers=headers,
                body=body,
                request_id=request_id,
                payload_stats=payload_stats,
            )
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as e:
            _log_codex_failure(
                stage='final_exhausted',
                request_id=request_id,
                error=e,
                payload_stats=payload_stats,
                retries=_CODEX_MAX_RETRIES,
            )
            return LLMResponse(
                content=f"Error calling Codex: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _normalize_reasoning_effort(reasoning_effort: str | None, model: str) -> str | None:
    if not reasoning_effort:
        return None
    effort = str(reasoning_effort).strip().lower()
    if effort not in _SUPPORTED_REASONING_EFFORTS:
        logger.warning("Ignoring unsupported Codex reasoning effort '{}'", reasoning_effort)
        return None
    if "gpt-5" not in _strip_model_prefix(model).lower():
        return None
    return effort


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _format_codex_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    if isinstance(cause, Exception):
        cause_text = _format_codex_error(cause)
        if cause_text:
            return f'{exc.__class__.__name__}: {cause_text}'
    return exc.__class__.__name__


def _extract_request_id_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r'request id\s+([0-9a-fA-F-]{8,})', text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_request_id_from_headers(headers: httpx.Headers | dict[str, Any] | None) -> str | None:
    if not headers:
        return None
    for key in ('x-request-id', 'request-id', 'openai-request-id'):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _payload_stats(
    *,
    model: str,
    system_prompt: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    body_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    return {
        'model': model,
        'system_prompt_chars': len(system_prompt),
        'input_item_count': len(input_items),
        'tool_count': len(tools or []),
        'body_chars': len(body_text),
        'body_tokens_est': max(1, len(body_text) // 4),
        'prompt_cache_key': body.get('prompt_cache_key'),
        'parallel_tool_calls': bool(body.get('parallel_tool_calls')),
        'has_reasoning': 'reasoning' in body,
    }


def _log_codex_event(level: str, message: str, **fields: Any) -> None:
    logger.bind(provider='codex', **fields).log(level.upper(), message)


def _log_codex_failure(
    *,
    stage: str,
    request_id: str,
    error: Exception,
    payload_stats: dict[str, Any],
    attempt: int | None = None,
    retries: int | None = None,
) -> None:
    codex_request_id = None
    if isinstance(error, CodexAPIError):
        codex_request_id = error.request_id or _extract_request_id_from_text(str(error))
    else:
        codex_request_id = _extract_request_id_from_text(str(error))
    fields = {
        'stage': stage,
        'request_id': request_id,
        'upstream_request_id': codex_request_id,
        'exception_class': error.__class__.__name__,
        'error_message': _format_codex_error(error),
        'attempt': attempt,
        'retries': retries,
        **payload_stats,
    }
    if isinstance(error, CodexAPIError):
        fields.update(
            {
                'status_code': error.status_code,
                'error_type': error.error_type,
                'error_code': error.error_code,
                'retry_after': error.retry_after,
                'response_status': error.response_status,
                'raw_error': error.raw_error,
            }
        )
    _log_codex_event('error', 'Codex request failed', **fields)


def _is_retryable_codex_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, CodexAPIError):
        if exc.status_code in {500, 502, 503, 504, 408, 409}:
            return True
        if exc.status_code == 429:
            return exc.error_code != 'insufficient_quota'
        # Responses/Codex can emit terminal SSE error events without an HTTP status.
        # Treat upstream/server/incomplete failures as retryable unless they are clearly
        # quota/auth/permission problems.
        retryable_error_types = {'server_error', 'internal_error', 'timeout_error'}
        retryable_error_codes = {
            'server_error',
            'internal_error',
            'timeout',
            'overloaded',
            'temporarily_unavailable',
        }
        non_retryable_error_types = {
            'authentication_error',
            'permission_error',
            'invalid_request_error',
            'quota_exceeded',
        }
        non_retryable_error_codes = {
            'insufficient_quota',
            'invalid_api_key',
            'invalid_request',
            'permission_denied',
        }
        if exc.error_type in non_retryable_error_types or exc.error_code in non_retryable_error_codes:
            return False
        if exc.error_type in retryable_error_types or exc.error_code in retryable_error_codes:
            return True
        if (exc.response_status or '').lower() in {'failed', 'incomplete'}:
            return True
    cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    if isinstance(cause, Exception) and _is_retryable_codex_error(cause):
        return True
    text = _format_codex_error(exc).lower()
    return any(fragment in text for fragment in [
        'http 500',
        'http 502',
        'http 503',
        'http 504',
        'server_error',
        'upstream connect error',
        'disconnect/reset before headers',
        'transport failure reason',
        'delayed connect error',
        'no route to host',
        'connecterror',
        'readtimeout',
        'pooltimeout',
        'temporarily unavailable',
    ])


async def _request_codex_with_retries(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    request_id: str,
    payload_stats: dict[str, Any],
) -> tuple[str, list[ToolCallRequest], str, dict[str, int]]:
    last_error: Exception | None = None
    verify = True
    for attempt in range(1, _CODEX_MAX_RETRIES + 1):
        try:
            return await _request_codex(
                url,
                headers,
                body,
                verify=verify,
                request_id=request_id,
                payload_stats=payload_stats,
            )
        except Exception as e:
            last_error = e
            if 'CERTIFICATE_VERIFY_FAILED' in str(e):
                if verify:
                    _log_codex_event(
                        'warning',
                        'SSL certificate verification failed for Codex API; retrying with verify=False',
                        stage='ssl_retry',
                        request_id=request_id,
                        attempt=attempt,
                        retries=_CODEX_MAX_RETRIES,
                        **payload_stats,
                    )
                    verify = False
                    continue
            if attempt >= _CODEX_MAX_RETRIES or not _is_retryable_codex_error(e):
                raise
            retry_after = getattr(e, 'retry_after', None)
            delay = retry_after if retry_after is not None else [2, 8, 16][attempt - 1]
            delay = max(1.0, min(float(delay), 60.0))
            _log_codex_failure(
                stage='retryable_error',
                request_id=request_id,
                error=e,
                payload_stats=payload_stats,
                attempt=attempt,
                retries=_CODEX_MAX_RETRIES,
            )
            _log_codex_event(
                'warning',
                'Codex request will be retried',
                stage='retry_scheduled',
                request_id=request_id,
                attempt=attempt,
                retries=_CODEX_MAX_RETRIES,
                retry_delay_sec=delay,
                **payload_stats,
            )
            await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
    *,
    request_id: str,
    payload_stats: dict[str, Any],
) -> tuple[str, list[ToolCallRequest], str, dict[str, int]]:
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raw_text = text.decode("utf-8", "ignore")
                retry_after = _parse_retry_after(response.headers.get('retry-after'))
                message, error_type, error_code = _friendly_error(response.status_code, raw_text)
                upstream_request_id = _extract_request_id_from_headers(response.headers) or _extract_request_id_from_text(raw_text)
                raise CodexAPIError(
                    message,
                    status_code=response.status_code,
                    retry_after=retry_after,
                    error_type=error_type,
                    error_code=error_code,
                    request_id=upstream_request_id,
                    response_status=str(response.status_code),
                    raw_error=raw_text[:1000],
                )
            return await _consume_sse(response, request_id=request_id, payload_stats=payload_stats)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            # Handle text first.
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            # Then handle tool calls.
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return "\n\n".join(system_parts), input_items


def _content_to_text(content: Any) -> str:
    return content_to_text(content)


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Compute a stable cache key from the static prompt prefix.

    We key on the first system message plus tool schema, so per-turn user/history
    changes (and dynamic memory snippets injected as later messages) do not churn
    the cache key.
    """
    first_system = ""
    for msg in messages:
        if msg.get("role") == "system":
            first_system = _content_to_text(msg.get("content")).strip()
            break

    payload = {
        "v": 2,
        "system": first_system,
        "tools": _convert_tools(tools) if tools else [],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(
    response: httpx.Response,
    *,
    request_id: str | None = None,
    payload_stats: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [l[5:].strip() for l in buffer if l.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception as e:
                    _log_codex_event(
                        'warning',
                        'Codex SSE event parse failed',
                        stage='sse_parse_error',
                        request_id=request_id,
                        parse_error=str(e),
                        event_preview=data[:500],
                        **(payload_stats or {}),
                    )
                    continue
            continue
        buffer.append(line)


def _extract_event_error(event: dict[str, Any]) -> str:
    for key in ('error', 'response'):
        value = event.get(key)
        if not isinstance(value, dict):
            continue
        err = value.get('error') if key == 'response' else value
        if isinstance(err, dict):
            message = str(err.get('message') or '').strip()
            code = str(err.get('code') or '').strip()
            err_type = str(err.get('type') or '').strip()
            parts = [part for part in [message, code, err_type] if part]
            if parts:
                return ' | '.join(parts)
        if key == 'response':
            status = str(value.get('status') or '').strip()
            incomplete = value.get('incomplete_details')
            if isinstance(incomplete, dict):
                reason = str(incomplete.get('reason') or '').strip()
                if reason:
                    return f'{status}: {reason}' if status else reason
            if status:
                return status
    message = str(event.get('message') or '').strip()
    if message:
        return message
    return 'Codex response failed'


async def _consume_sse(
    response: httpx.Response,
    *,
    request_id: str | None = None,
    payload_stats: dict[str, Any] | None = None,
) -> tuple[str, list[ToolCallRequest], str, dict[str, int]]:
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"
    usage: dict[str, int] = {}

    async for event in _iter_sse(response, request_id=request_id, payload_stats=payload_stats):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            content += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=buf.get("name") or item.get("name"),
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            response_obj = event.get("response") or {}
            status = response_obj.get("status")
            finish_reason = _map_finish_reason(status)
            usage_obj = response_obj.get("usage") or {}
            input_tokens = int(usage_obj.get("input_tokens") or 0)
            cached_tokens = int((usage_obj.get("input_tokens_details") or {}).get("cached_tokens") or 0)
            output_tokens = int(usage_obj.get("output_tokens") or 0)
            usage = {
                # Codex/Responses usage.input_tokens already reflects the total prompt size.
                # cached_tokens is reported separately for cache-hit accounting and should not
                # be added again or the caller will double-count cached prompt segments.
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "cache_read_input_tokens": cached_tokens,
            }
        elif event_type in {"error", "response.failed", "response.incomplete"}:
            message = _extract_event_error(event)
            response_obj = event.get('response') if isinstance(event.get('response'), dict) else {}
            response_error = response_obj.get('error') if isinstance(response_obj, dict) else {}
            request_id_upstream = _extract_request_id_from_headers(response.headers) or _extract_request_id_from_text(message)
            raise CodexAPIError(
                message,
                status_code=None,
                error_type=(response_error.get('type') if isinstance(response_error, dict) else None),
                error_code=(response_error.get('code') if isinstance(response_error, dict) else None),
                request_id=request_id_upstream,
                response_status=(response_obj.get('status') if isinstance(response_obj, dict) else None),
                raw_error=json.dumps(event, ensure_ascii=False)[:1000],
            )

    return content, tool_calls, finish_reason, usage


_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _friendly_error(status_code: int, raw: str) -> tuple[str, str | None, str | None]:
    error_type = None
    error_code = None
    message = raw.strip()
    try:
        payload = json.loads(raw)
        error = payload.get('error') if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = str(error.get('message') or message or '').strip()
            error_type = str(error.get('type') or '') or None
            error_code = str(error.get('code') or '') or None
    except Exception:
        pass

    lower = message.lower()
    if status_code == 401:
        return ('Codex authentication failed. Please reconnect the ChatGPT/Codex OAuth session.', error_type or 'authentication_error', error_code)
    if status_code == 403:
        return ('Codex access was denied for this request. Check subscription access and model permissions.', error_type or 'permission_error', error_code)
    if status_code == 404:
        return ('Codex endpoint or requested resource was not found.', error_type or 'not_found_error', error_code)
    if status_code == 408:
        return ('Codex request timed out before a response was available.', error_type or 'timeout_error', error_code)
    if status_code == 409:
        return ('Codex request conflicted with current backend state. Please retry.', error_type or 'conflict_error', error_code)
    if status_code == 413:
        return ('Codex rejected the request because it was too large.', error_type or 'request_too_large', error_code)
    if status_code == 422:
        return ('Codex rejected the request payload as invalid.', error_type or 'invalid_request_error', error_code)
    if status_code == 429:
        if 'insufficient_quota' in lower or error_code == 'insufficient_quota':
            return ('ChatGPT/Codex quota is exhausted for the current plan or billing pool. Please try again later.', error_type or 'quota_exceeded', error_code or 'insufficient_quota')
        return ('ChatGPT/Codex rate limit triggered. Please wait a bit and retry.', error_type or 'rate_limit_error', error_code)
    if status_code in {500, 502, 503, 504}:
        return (f'Codex upstream service is temporarily unavailable (HTTP {status_code}).', error_type or 'server_error', error_code)

    if not message:
        message = f'HTTP {status_code}'
    return (f'HTTP {status_code}: {message}', error_type, error_code)
