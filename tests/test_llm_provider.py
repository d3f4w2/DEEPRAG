import asyncio
import json

import httpx

import backend.llm_provider as llm_provider_module


class FakeJsonResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    async def aread(self):
        return json.dumps(self._payload).encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class FakeStreamResponse:
    def __init__(self, chunks, status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return b"".join(self._chunks)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class FakeAsyncClient:
    def __init__(self, post_responses, stream_responses):
        self.post_responses = list(post_responses)
        self.stream_responses = list(stream_responses)
        self.post_calls = []
        self.stream_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        return self.post_responses.pop(0)

    def stream(self, method, url, json, headers):
        self.stream_calls.append(
            {"method": method, "url": url, "json": json, "headers": headers}
        )
        return self.stream_responses.pop(0)


async def _collect_chunks(provider, messages, stream=False):
    chunks = []
    async for chunk in provider.chat_completion(messages=messages, tools=None, stream=stream):
        chunks.append(json.loads(chunk))
    return chunks


def _build_provider(monkeypatch, fake_client):
    monkeypatch.setattr(
        llm_provider_module.httpx,
        "AsyncClient",
        lambda timeout=120.0: fake_client,
    )
    provider = llm_provider_module.LLMProvider("openai")
    provider.config = {
        "base_url": "https://example.com/v1",
        "model": "gpt-5.4-mini",
        "headers": {},
        "api_key": "test-key",
    }
    return provider


def test_chat_completion_uses_non_stream_content_when_available(monkeypatch):
    fake_client = FakeAsyncClient(
        post_responses=[
            FakeJsonResponse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        ],
        stream_responses=[],
    )
    provider = _build_provider(monkeypatch, fake_client)

    chunks = asyncio.run(
        _collect_chunks(
            provider,
            messages=[{"role": "user", "content": "Say OK"}],
            stream=False,
        )
    )

    assert chunks == [{"type": "content", "content": "OK"}]
    assert fake_client.stream_calls == []


def test_chat_completion_retries_in_stream_mode_when_non_stream_content_is_empty(
    monkeypatch,
):
    fake_client = FakeAsyncClient(
        post_responses=[
            FakeJsonResponse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                }
            )
        ],
        stream_responses=[
            FakeStreamResponse(
                [
                    b'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
                    b'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n',
                    b"data: [DONE]\n",
                ]
            )
        ],
    )
    provider = _build_provider(monkeypatch, fake_client)

    chunks = asyncio.run(
        _collect_chunks(
            provider,
            messages=[{"role": "user", "content": "Say hello world"}],
            stream=False,
        )
    )

    assert chunks == [{"type": "content", "content": "Hello world"}]
    assert len(fake_client.stream_calls) == 1
    assert fake_client.stream_calls[0]["json"]["stream"] is True
    assert fake_client.stream_calls[0]["json"]["stream_options"] == {"include_usage": True}
