"""
Ollama local model provider.
Fetches available models from the running Ollama instance at startup.
Model can be changed live via set_model().
"""
from __future__ import annotations

import requests
from backend.llm.base import LLMProvider
from backend.llm.errors import provider_error_message, redact_secrets
from backend.logging_config import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434", max_tokens: int = 2048):
        self._model = model
        self._host = host.rstrip("/")
        self._max_tokens = max_tokens

    # ── Interface ─────────────────────────────────────────────────────────────

    @property
    def current_model(self) -> str:
        return self._model

    def set_model(self, model_id: str) -> None:
        logger.info("Switching Ollama model", old=self._model, new=model_id)
        self._model = model_id

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self._max_tokens},
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + messages
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(f"{self._host}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama request failed", error=redact_secrets(exc))
            raise RuntimeError(provider_error_message("Ollama", exc)) from exc

        return self._normalize(resp.json())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _normalize(self, data: dict) -> dict:
        message = data.get("message", {})
        # Ollama tool_calls format: [{function: {name, arguments}}]
        raw_tools = message.get("tool_calls") or []
        tool_calls = []
        for tc in raw_tools:
            fn = tc.get("function", {})
            tool_calls.append({
                "name": fn.get("name", ""),
                "input": fn.get("arguments", {}),
                "id": None,
            })
        return {
            "content": message.get("content", ""),
            "tool_calls": tool_calls,
        }

    # ── Class helpers ─────────────────────────────────────────────────────────

    def available_models(self) -> list[dict]:
        """
        Fetch models currently pulled in Ollama.
        Returns [] if Ollama isn't running (non-fatal; shown in UI).
        """
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [
                {"id": m["name"], "label": m["name"]}
                for m in models
                if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"].strip()
            ]
        except Exception as exc:
            # Ollama is optional. Connection errors, invalid JSON, and an
            # unexpected response shape should all be treated as an empty
            # optional list rather than breaking the provider selector.
            logger.warning(
                "Could not fetch Ollama model list; is Ollama running?",
                error=str(exc),
            )
            return []
