import pytest

from nanobot.providers.openai_codex_provider import (
    CodexAPIError,
    OpenAICodexProvider,
    _is_retryable_codex_error,
)


def test_openai_codex_server_error_without_http_status_is_retryable() -> None:
    err = CodexAPIError(
        'An error occurred while processing your request. Please include the request ID 58756663-4402-4680-9ce1-709faf11099a in your message. | server_error | server_error',
        error_type='server_error',
        error_code='server_error',
    )
    assert _is_retryable_codex_error(err) is True


@pytest.mark.asyncio
async def test_openai_codex_provider_retries_sse_server_error_without_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Token:
        account_id = 'acct'
        access = 'secret'

    attempts = {'count': 0}

    async def _fake_request(url, headers, body, verify, request_id, payload_stats):
        _ = (url, headers, body, verify, request_id, payload_stats)
        attempts['count'] += 1
        if attempts['count'] < 3:
            raise CodexAPIError(
                'An error occurred while processing your request. Please include the request ID 58756663-4402-4680-9ce1-709faf11099a in your message. | server_error | server_error',
                error_type='server_error',
                error_code='server_error',
                request_id='58756663-4402-4680-9ce1-709faf11099a',
            )
        return 'OK', [], 'stop', {'prompt_tokens': 10, 'completion_tokens': 2}

    delays = []

    async def _no_sleep(delay):
        delays.append(delay)
        return None

    monkeypatch.setattr('nanobot.providers.openai_codex_provider.get_codex_token', lambda: _Token())
    monkeypatch.setattr('nanobot.providers.openai_codex_provider._request_codex', _fake_request)
    monkeypatch.setattr('nanobot.providers.openai_codex_provider.asyncio.sleep', _no_sleep)

    provider = OpenAICodexProvider(default_model='openai-codex/gpt-5.4')
    response = await provider.chat(messages=[{'role': 'user', 'content': 'hello'}])

    assert response.content == 'OK'
    assert attempts['count'] == 3
    assert delays == [2, 8]
