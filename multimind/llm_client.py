from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from multimind.config import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    REPETITION_SUFFIX_MAX_LENGTH,
    REPETITION_SUFFIX_MIN_LENGTH,
    REPETITION_SUFFIX_REPEAT_COUNT,
    REQUEST_TIMEOUT_SECONDS,
)
from multimind.discovery import normalize_base_url


class LocalLLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        *,
        provider_kind: str,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        ollama_think: bool = False,
    ) -> AsyncIterator[str]:
        repetition_buffer = ""

        if provider_kind == "ollama":
            async for token in self._stream_ollama(
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=temperature,
                ollama_think=ollama_think,
            ):
                repetition_buffer += token
                yield token
                if _has_repetitive_suffix(repetition_buffer):
                    break
            return

        async for token in self._stream_openai(
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
        ):
            repetition_buffer += token
            yield token
            if _has_repetitive_suffix(repetition_buffer):
                break

    async def _stream_ollama(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        ollama_think: bool,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": DEFAULT_TOP_P,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
            },
        }

        if _supports_boolean_think(model):
            payload["think"] = ollama_think

        async with self._client.stream(
            "POST",
            f"{normalize_base_url(base_url)}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue

                payload = json.loads(line)
                message = payload.get("message") or {}
                content = message.get("content")
                if content:
                    yield content
                if payload.get("done"):
                    break

    async def _stream_openai(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": DEFAULT_TOP_P,
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        }

        async with self._client.stream(
            "POST",
            f"{normalize_base_url(base_url)}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                payload = json.loads(data)
                choices = payload.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content


def _has_repetitive_suffix(text: str) -> bool:
    text = " ".join(text.split())
    repeat_count = REPETITION_SUFFIX_REPEAT_COUNT
    max_suffix = min(REPETITION_SUFFIX_MAX_LENGTH, len(text) // repeat_count)

    if max_suffix < REPETITION_SUFFIX_MIN_LENGTH:
        return False

    for suffix_length in range(REPETITION_SUFFIX_MIN_LENGTH, max_suffix + 1):
        suffix = text[-suffix_length:]
        if not suffix.strip():
            continue
        if text.endswith(suffix * repeat_count):
            return True

    return False


def _supports_boolean_think(model: str) -> bool:
    return not model.lower().startswith("gpt-oss")