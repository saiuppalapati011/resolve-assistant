"""Abstract base for all LLM providers."""
from __future__ import annotations
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Single method surface all providers must satisfy.
    Returns a normalized dict so no downstream code cares which
    provider is active.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        """
        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            tools:    Optional list of tool schemas (provider-specific format
                      is handled inside each implementation)
            system:   Optional system prompt string

        Returns:
            {
                "content":    str,          # text response (may be empty if tool_calls present)
                "tool_calls": list[dict],   # [{name, input/arguments, id}]
            }
        """
        ...

    @property
    @abstractmethod
    def current_model(self) -> str:
        """Return the currently active model identifier."""
        ...

    @abstractmethod
    def set_model(self, model_id: str) -> None:
        """Switch to a different model (live, no restart needed)."""
        ...
