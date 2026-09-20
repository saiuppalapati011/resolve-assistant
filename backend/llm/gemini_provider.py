"""
Google Gemini provider.
Supports all Gemini models via REST API.
"""
from __future__ import annotations

import time
import requests
from backend.llm.base import LLMProvider
from backend.llm.errors import is_transient_provider_error, provider_error_message, redact_secrets
from backend.logging_config import get_logger

logger = get_logger(__name__)

GEMINI_MODELS: list[dict] = [
    {"id": "gemini-3.7-flash", "label": "Gemini 3.7 Flash"},
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite"},
    {"id": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash-Lite"},
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro"},
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash"},
    {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
    {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash-Lite"},
]


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite", max_tokens: int = 2048):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    # ── Interface ─────────────────────────────────────────────────────────────

    @property
    def current_model(self) -> str:
        return self._model

    def set_model(self, model_id: str) -> None:
        logger.info("Switching Gemini model", old=self._model, new=model_id)
        self._model = model_id

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        url = f"{self._base_url}/{self._model}:generateContent?key={self._api_key}"
        
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self._max_tokens
            }
        }
        
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }
            
        # Keep popup failures bounded. A stalled provider should not leave the
        # Resolve window waiting for roughly a minute before it can recover.
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=12,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as exc:
                if attempt < attempts and is_transient_provider_error(exc):
                    delay = 0.6 * (2 ** (attempt - 1))
                    logger.warning(
                        "Transient Gemini API failure; retrying",
                        attempt=attempt,
                        delay_s=delay,
                        error=redact_secrets(exc),
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "Gemini API error",
                    error=redact_secrets(exc),
                    status=getattr(getattr(exc, "response", None), "status_code", None),
                )
                raise RuntimeError(provider_error_message("Gemini", exc)) from exc

        return self._normalize(data)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalize(self, data: dict) -> dict:
        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                if "text" in p:
                    content += p["text"]
                    
        return {
            "content": content,
            "tool_calls": [],
        }

    # ── Class helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def available_models() -> list[dict]:
        """Return the model catalog for the UI dropdown."""
        return GEMINI_MODELS
