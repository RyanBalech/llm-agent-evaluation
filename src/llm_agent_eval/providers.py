from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class MistralChatClient:
    """Thin structured-output adapter around the official Mistral SDK.

    Install with `pip install -e ".[providers]"`.
    Credentials are read from MISTRAL_API_KEY unless explicitly supplied.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        random_seed: int | None = 0,
    ) -> None:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise RuntimeError(
                'Mistral support requires: pip install -e ".[providers]"'
            ) from exc

        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not set")

        self.model = model
        self.temperature = temperature
        self.random_seed = random_seed
        self._client = Mistral(api_key=key)
        self.last_usage = Usage()

    def complete(self, prompt: str) -> str:
        response = self._client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            random_seed=self.random_seed,
        )
        self.last_usage = _usage_from_response(response)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError("Expected string content from Mistral chat completion")
        return content


def _usage_from_response(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()

    def read(name: str) -> int | None:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        return int(value) if value is not None else None

    return Usage(
        prompt_tokens=read("prompt_tokens"),
        completion_tokens=read("completion_tokens"),
        total_tokens=read("total_tokens"),
    )
