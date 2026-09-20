"""
LLM provider factory.
This is the ONLY file that imports concrete provider classes.
Everything else gets the LLMProvider interface via get_provider().
"""
from __future__ import annotations

from backend.config import settings
from backend.llm.base import LLMProvider
from backend.logging_config import get_logger
from backend.llm.errors import redact_secrets


logger = get_logger(__name__)


def get_provider(
    provider: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """
    Build and return the configured LLMProvider instance.

    Args:
        provider: Override "anthropic" | "ollama" (falls back to settings).
        model:    Override model id (falls back to settings default).
    """
    chosen = (provider or settings.llm_provider).lower()

    if chosen == "anthropic":
        from backend.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=model or settings.anthropic_default_model,
            max_tokens=settings.anthropic_max_tokens,
        )

    elif chosen == "gemini":
        from backend.llm.gemini_provider import GeminiProvider
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=model or settings.gemini_default_model,
            max_tokens=settings.gemini_max_tokens,
        )

    elif chosen == "ollama":
        from backend.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=model or settings.ollama_default_model,
            host=settings.ollama_host,
            max_tokens=settings.ollama_max_tokens,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{chosen}'. "
            "Valid options are 'anthropic', 'gemini', or 'ollama'."
        )


def get_model_catalog(provider: str | None = None) -> dict:
    """
    Return the model list for the UI dropdown.
    Always returns all providers' lists for the full selector.
    """
    from backend.llm.anthropic_provider import AnthropicProvider
    from backend.llm.gemini_provider import GeminiProvider
    from backend.llm.ollama_provider import OllamaProvider

    def safe_models(provider_name: str, loader) -> list[dict]:
        """Return one provider's models without making the whole catalog fail."""
        try:
            return loader()
        except Exception as exc:
            # Model discovery is only for the selector. A provider being
            # offline or returning malformed data must not make /api/models
            # return HTTP 500 and prevent the popup from opening.
            logger.warning(
                "Could not load provider model list",
                provider=provider_name,
                error=redact_secrets(exc),
            )
            return []

    ollama = OllamaProvider(host=settings.ollama_host)
    return {
        "anthropic": safe_models("anthropic", AnthropicProvider.available_models),
        "gemini": safe_models("gemini", GeminiProvider.available_models),
        "ollama": safe_models("ollama", ollama.available_models),
        "current_provider": provider or settings.llm_provider,
        "current_model": settings.llm_model,
    }
