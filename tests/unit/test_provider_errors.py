from types import SimpleNamespace

import requests

from backend.llm.errors import provider_error_message, redact_secrets


def test_redact_secrets_removes_query_string_api_key():
    text = "https://example.test/generate?key=super-secret-value"

    safe = redact_secrets(text)

    assert "super-secret-value" not in safe
    assert "key=[REDACTED]" in safe


def test_transient_provider_failure_gets_retry_message():
    error = requests.HTTPError(
        "503 Server Error",
        response=SimpleNamespace(status_code=503),
    )

    message = provider_error_message("gemini", error)

    assert "temporarily unavailable" in message
    assert "try again shortly" in message
    assert "503" not in message

