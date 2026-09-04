"""
Anthropic Claude provider.
Supports all Claude models; model can be changed live via set_model().
"""
from __future__ import annotations

import anthropic
from backend.llm.base import LLMProvider
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Canonical list of known Anthropic models (shown in UI selector).
# The user can still type any valid model id; this list drives the dropdown.
ANTHROPIC_MODELS: list[dict] = [
    {"id": "claude-opus-4-5",    "label": "Claude Opus 4.5"},
    {"id": "claude-sonnet-4-5",  "label": "Claude Sonnet 4.5 (Recommended)"},
    {"id": "claude-haiku-4-5",   "label": "Claude Haiku 4.5 (Fast)"},
    {"id": "claude-opus-4-0",    "label": "Claude Opus 4"},
    {"id": "claude-sonnet-4-0",  "label": "Claude Sonnet 4"},
    {"id": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet"},
    {"id": "claude-3-5-haiku-20241022",  "label": "Claude 3.5 Haiku"},
    {"id": "claude-3-opus-20240229",     "label": "Claude 3 Opus"},
]


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", max_tokens: int = 2048):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
        self._model = model
        self._max_tokens = max_tokens

    # ── Interface ─────────────────────────────────────────────────────────────

    @property
    def current_model(self) -> str:
        return self._model

    def set_model(self, model_id: str) -> None:
        logger.info("Switching Anthropic model", old=self._model, new=model_id)
        self._model = model_id

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            logger.error("Anthropic API error", error=str(exc))
            raise

        return self._normalize(response)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalize(self, response: anthropic.types.Message) -> dict:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            {"name": b.name, "input": b.input, "id": b.id}
            for b in response.content
            if b.type == "tool_use"
        ]
        return {
            "content": "\n".join(text_blocks),
            "tool_calls": tool_calls,
        }

    # ── Class helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def available_models() -> list[dict]:
        """Return the model catalog for the UI dropdown."""
        return ANTHROPIC_MODELS
