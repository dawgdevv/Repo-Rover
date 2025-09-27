"""Lightweight abstraction over LLM providers with a deterministic fallback."""

from __future__ import annotations

import textwrap
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from ..core.config import Settings


class LLMClient:
    """Best-effort wrapper that uses OpenAI if credentials are available."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Optional[Any] = None
        if settings.openai_api_key and OpenAI is not None:
            self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._client is None:
            return self._fallback_response(system_prompt, user_prompt)

        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _fallback_response(system_prompt: str, user_prompt: str) -> str:
        combined = f"System: {system_prompt}\nUser: {user_prompt}"
        return textwrap.dedent(
            f"""
            No LLM credentials detected. Here's a heuristic summary based on the prompts provided:

            {combined[:1200]}

            (Configure an OpenAI API key to replace this fallback with real model generations.)
            """
        ).strip()
