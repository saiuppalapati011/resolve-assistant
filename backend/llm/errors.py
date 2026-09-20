"""Shared provider error classification and safe user-facing messages."""
from __future__ import annotations

import re


_SECRET_QUERY_RE = re.compile(
    r"([?&](?:key|api[_-]?key|token|access_token|secret)=)[^&\s]+",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"((?:api[_-]?key|token|access[_-]?token|secret)\s*[:=]\s*)[^\s,;]+",
    re.IGNORECASE,
)


def redact_secrets(value: object) -> str:
    """Remove API keys and tokens from exception text before logging/display."""
    text = str(value or "")
    text = _SECRET_QUERY_RE.sub(r"\1[REDACTED]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)


def is_transient_provider_error(exc: BaseException) -> bool:
    """Return True for provider failures that are worth retrying."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return True

    name = type(exc).__name__.lower()
    text = redact_secrets(exc).lower()
    return (
        any(token in name for token in ("timeout", "connection", "ssl"))
        or any(
            marker in text
            for marker in (
                "ssleoferror",
                "unexpected_eof",
                "temporarily unavailable",
                "service unavailable",
                "connection reset",
                "connection aborted",
                "max retries exceeded",
                "timed out",
            )
        )
    )


def provider_error_message(provider: str | None, exc: BaseException) -> str:
    """Convert provider exceptions into concise messages without secrets."""
    name = (provider or "AI provider").title()
    text = redact_secrets(exc)
    lower = text.lower()
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)

    if "api_key" in lower or "api key" in lower or "not set" in lower:
        return f"{name} is not configured. Add its API key in the backend .env file."
    if status in {401, 403} or "unauthorized" in lower or "permission" in lower:
        return f"{name} rejected the request. Check the provider API key and model access."
    if is_transient_provider_error(exc) or status in {408, 425, 429, 500, 502, 503, 504}:
        return (
            f"{name} is temporarily unavailable. I retried the request, but it did not complete. "
            "Please try again shortly or switch providers."
        )
    return f"The {name} request failed. Check the provider settings and try again."
