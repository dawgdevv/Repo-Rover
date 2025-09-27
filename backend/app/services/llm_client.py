"""Lightweight abstraction over Gemini with a deterministic fallback."""

from __future__ import annotations

import logging
import textwrap
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

from ..core.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Google Gemini that gracefully degrades when unavailable."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Optional["genai.GenerativeModel"] = None  # type: ignore[name-defined]

        if settings.google_api_key and genai is not None:
            try:
                genai.configure(api_key=settings.google_api_key)
                self._model = genai.GenerativeModel(settings.llm_model)
            except Exception as exc:  # pragma: no cover - network/config errors
                logger.warning("Failed to initialise Gemini model '%s': %s", settings.llm_model, exc)
                self._model = None
        elif settings.google_api_key:
            logger.warning("google-generativeai package is not installed; falling back to heuristics.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._model is None:
            return self._fallback_response(system_prompt, user_prompt)

        composite_prompt = textwrap.dedent(
            f"""
            You must follow the system instruction first.

            System instruction:
            {system_prompt}

            User request:
            {user_prompt}
            """
        ).strip()

        try:
            response = self._model.generate_content(
                composite_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1200,
                },
            )
        except Exception as exc:  # pragma: no cover - upstream failure
            logger.warning("Gemini generation failed; returning fallback response: %s", exc)
            return self._fallback_response(system_prompt, user_prompt)

        text = getattr(response, "text", None)
        if not text:
            logger.warning("Gemini returned empty response; using fallback")
            return self._fallback_response(system_prompt, user_prompt)
        return text

    @property
    def is_configured(self) -> bool:
        return self._model is not None

    @staticmethod
    def _fallback_response(system_prompt: str, user_prompt: str) -> str:
        combined = f"System: {system_prompt}\nUser: {user_prompt}"
        return textwrap.dedent(
            f"""
            No Gemini credentials detected (or initialisation failed). Here's a heuristic summary based on the
            prompts provided:

            {combined[:1200]}

            (Configure GOOGLE_API_KEY to replace this fallback with real model generations.)
            """
        ).strip()
