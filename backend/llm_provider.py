from typing import Any, AsyncIterator, Dict, List, Optional
import codecs
import json

import httpx

from backend.config import settings


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _collect_text_from_content(content: Any, max_depth: int = 6) -> str:
    """Extract textual payload from varied OpenAI-compatible content shapes.

    Common forms seen in the wild:
    - "plain string"
    - [{"type": "text", "text": "..."}]
    - [{"type": "output_text", "text": "..."}]
    - {"text": "..."}
    - providers that place text in nested "value" / "content"
    """

    collected: List[str] = []
    seen: set[str] = set()

    def _push(text_like: Any) -> None:
        text = _normalize_text(text_like)
        if not text:
            return
        if text in seen:
            return
        seen.add(text)
        collected.append(text)

    def _walk(node: Any, depth: int) -> None:
        if depth > max_depth or node is None:
            return

        if isinstance(node, str):
            _push(node)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)
            return

        if isinstance(node, dict):
            preferred_keys = (
                "text",
                "content",
                "output_text",
                "reasoning_content",
                "value",
            )

            for key in preferred_keys:
                if key in node:
                    _walk(node.get(key), depth + 1)

            # Some providers put multimodal parts under these keys.
            for key in ("parts", "items", "output", "message", "delta"):
                if key in node:
                    _walk(node.get(key), depth + 1)
            return

    _walk(content, 0)
    return "\n".join(collected).strip()


def _extract_choice_text(choice: Dict[str, Any]) -> str:
    message = choice.get("message") or {}

    # Standard field first.
    text = _collect_text_from_content(message.get("content"))
    if text:
        return text

    # Provider-specific fallbacks.
    for key in ("output_text", "reasoning_content", "text"):
        text = _collect_text_from_content(message.get(key))
        if text:
            return text

    # Rare gateways place text directly on choice.
    for key in ("text", "content", "output_text", "reasoning_content"):
        text = _collect_text_from_content(choice.get(key))
        if text:
            return text

    return ""


def _messages_contain_image(messages: List[Dict[str, Any]]) -> bool:
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"image_url", "input_image"}:
                    return True
    return False


def _normalize_image_url(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _normalize_text(value.get("url") or value.get("image_url"))
    return ""


def _to_responses_content(content: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []

    if isinstance(content, str):
        text = content.strip()
        if text:
            parts.append({"type": "input_text", "text": text})
        return parts

    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()

            if item_type in {"text", "input_text", "output_text"}:
                text = _normalize_text(item.get("text") or item.get("content"))
                if text:
                    parts.append({"type": "input_text", "text": text})
                continue

            if item_type in {"image_url", "input_image"}:
                image_url = _normalize_image_url(item.get("image_url") or item.get("url"))
                if image_url:
                    parts.append({"type": "input_image", "image_url": image_url})
                continue

            fallback_text = _collect_text_from_content(item)
            if fallback_text:
                parts.append({"type": "input_text", "text": fallback_text})

        if parts:
            return parts

    fallback_text = _collect_text_from_content(content)
    if fallback_text:
        parts.append({"type": "input_text", "text": fallback_text})
    return parts


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages or []:
        role = str(message.get("role") or "user").strip() or "user"
        parts = _to_responses_content(message.get("content"))
        if not parts:
            continue
        converted.append({"role": role, "content": parts})
    return converted


def _build_responses_url(chat_completions_url: str) -> str:
    if "/chat/completions" in chat_completions_url:
        return chat_completions_url.replace("/chat/completions", "/responses")
    return f"{chat_completions_url.rstrip('/')}/responses"


def _extract_text_chunk(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            chunk = _extract_text_chunk(item)
            if chunk:
                chunks.append(chunk)
        if chunks:
            return "".join(chunks)

    if isinstance(value, dict):
        for key in ("text", "content", "output_text", "reasoning_content", "value"):
            if key in value:
                chunk = _extract_text_chunk(value.get(key))
                if chunk:
                    return chunk
    return ""


def _extract_delta_text(delta: Dict[str, Any]) -> str:
    text = _extract_text_chunk(delta.get("content"))
    if text:
        return text

    for fallback_key in ("output_text", "reasoning_content", "text"):
        text = _extract_text_chunk(delta.get(fallback_key))
        if text:
            return text
    return ""


class LLMProvider:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.api_provider
        self.config = settings.get_provider_config(self.provider)

    async def _collect_content_via_stream_fallback(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> str:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}

        content_parts: List[str] = []
        async with client.stream("POST", url, json=stream_payload, headers=headers) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                print(
                    "[ERROR] Stream fallback API Error "
                    f"{response.status_code}: {error_text.decode('utf-8', errors='ignore')}"
                )
            response.raise_for_status()

            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
            async for chunk_bytes in response.aiter_bytes():
                decoded_chunk = decoder.decode(chunk_bytes, False)
                buffer += decoded_chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] Stream fallback JSON decode error: {e}")
                        continue

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        delta_text = _extract_delta_text(delta)
                        if delta_text:
                            content_parts.append(delta_text)

            final_chunk = decoder.decode(b"", True)
            if final_chunk:
                buffer += final_chunk
                if buffer.strip():
                    print(f"[DEBUG] Stream fallback trailing buffer: {buffer}")

        return "".join(content_parts).strip()

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Unified OpenAI-compatible chat completion."""
        base_url = self.config.get("base_url", "")
        if not base_url:
            raise ValueError(f"BASE_URL not configured for provider: {self.provider}")

        if "/chat/completions" in base_url:
            url = base_url
        else:
            url = f"{base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            **self.config.get("headers", {}),
        }

        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"

        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": settings.temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        # Newer models use max_completion_tokens, older models use max_tokens.
        model_name = str(self.config.get("model") or "")
        if "gpt-5" in model_name or "o1" in model_name:
            payload["max_completion_tokens"] = settings.max_tokens
        else:
            payload["max_tokens"] = settings.max_tokens

        if tools:
            payload["tools"] = tools

        print(f"[DEBUG] Calling API: {url}")
        print(f"[DEBUG] Model: {self.config['model']}")
        print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            print(f"[ERROR] API Error {response.status_code}: {error_text.decode('utf-8')}")
                        response.raise_for_status()

                        buffer = ""
                        chunk_count = 0
                        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

                        async for chunk_bytes in response.aiter_bytes():
                            chunk_count += 1
                            if chunk_count == 1:
                                print("[DEBUG] First chunk received!")

                            decoded_chunk = decoder.decode(chunk_bytes, False)
                            buffer += decoded_chunk

                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()

                                if not line or not line.startswith("data:"):
                                    continue

                                data = line[5:].strip()

                                if data == "[DONE]":
                                    break

                                try:
                                    chunk = json.loads(data)

                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        delta = chunk["choices"][0].get("delta", {})

                                        if "tool_calls" in delta:
                                            yield json.dumps(
                                                {
                                                    "type": "tool_calls",
                                                    "tool_calls": delta["tool_calls"],
                                                }
                                            )
                                        else:
                                            delta_text = _extract_delta_text(delta)
                                            if delta_text:
                                                yield json.dumps(
                                                    {
                                                        "type": "content",
                                                        "content": delta_text,
                                                    }
                                                )

                                    if chunk.get("usage"):
                                        yield json.dumps(
                                            {
                                                "type": "usage",
                                                "usage": chunk["usage"],
                                            }
                                        )
                                except json.JSONDecodeError as e:
                                    print(f"[DEBUG] JSON decode error: {e}")
                                    continue

                        # Flush decoder to handle any remaining possible bytes.
                        final_chunk = decoder.decode(b"", True)
                        if final_chunk:
                            buffer += final_chunk
                            if buffer.strip():
                                print(f"[DEBUG] Final buffer: {buffer}")
                else:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    all_parts: List[str] = []
                    for choice in data.get("choices") or []:
                        part = _extract_choice_text(choice)
                        if part:
                            all_parts.append(part)

                    # Fallback for gateways returning Responses-like payloads
                    # (e.g., top-level "output" instead of "choices[].message.content").
                    if not all_parts:
                        fallback_part = _collect_text_from_content(data)
                        if fallback_part:
                            all_parts.append(fallback_part)

                    content = "\n".join(all_parts).strip()
                    used_stream_fallback = False
                    used_responses_fallback = False
                    if not content:
                        print(
                            "[DEBUG] Empty non-stream content, retrying request in stream mode."
                        )
                        try:
                            streamed_content = await self._collect_content_via_stream_fallback(
                                client=client,
                                url=url,
                                headers=headers,
                                payload=payload,
                            )
                            if streamed_content:
                                content = streamed_content
                                used_stream_fallback = True
                        except Exception as fallback_error:
                            print(f"[WARN] Stream fallback failed: {fallback_error}")

                    if not content and tools is None and _messages_contain_image(messages):
                        responses_url = _build_responses_url(url)
                        responses_payload = {
                            "model": self.config["model"],
                            "input": _to_responses_input(messages),
                            "temperature": settings.temperature,
                            "max_output_tokens": settings.max_tokens,
                        }
                        print(f"[DEBUG] Empty multimodal content, trying Responses fallback: {responses_url}")
                        try:
                            responses_response = await client.post(
                                responses_url,
                                json=responses_payload,
                                headers=headers,
                            )
                            responses_response.raise_for_status()
                            responses_data = responses_response.json()
                            responses_text = _collect_text_from_content(responses_data)
                            if responses_text:
                                content = responses_text.strip()
                                used_responses_fallback = True
                            else:
                                top_level_keys = (
                                    list(responses_data.keys())
                                    if isinstance(responses_data, dict)
                                    else []
                                )
                                print(
                                    "[WARN] Responses fallback returned no text. "
                                    f"top_level_keys={top_level_keys}"
                                )
                        except Exception as fallback_error:
                            print(f"[WARN] Responses fallback failed: {fallback_error}")

                    if not content:
                        # Add high-signal diagnostics for multimodal compatibility issues.
                        sample_choice = (data.get("choices") or [{}])[0]
                        message_keys = list((sample_choice.get("message") or {}).keys())
                        choice_keys = list(sample_choice.keys())
                        top_level_keys = list(data.keys()) if isinstance(data, dict) else []
                        print(
                            "[WARN] Empty content extracted from non-stream response. "
                            f"choice_keys={choice_keys}, message_keys={message_keys}, "
                            f"top_level_keys={top_level_keys}"
                        )
                    elif used_stream_fallback:
                        print("[DEBUG] Stream fallback succeeded for empty non-stream response.")
                    elif used_responses_fallback:
                        print("[DEBUG] Responses fallback succeeded for multimodal request.")
                    yield json.dumps({"type": "content", "content": content})
        except httpx.ConnectError as e:
            print(f"[ERROR] Connection failed: {e}")
            yield json.dumps({"type": "content", "content": f"Connection error: {str(e)}"})
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            yield json.dumps({"type": "content", "content": f"Error: {str(e)}"})


llm_provider = LLMProvider()
