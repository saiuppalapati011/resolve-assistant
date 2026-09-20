from backend.llm.factory import get_model_catalog


def test_model_catalog_survives_optional_provider_failure(monkeypatch):
    from backend.llm.ollama_provider import OllamaProvider

    def fail_ollama(self):
        raise RuntimeError("Ollama returned invalid JSON")

    monkeypatch.setattr(OllamaProvider, "available_models", fail_ollama)

    catalog = get_model_catalog("gemini")

    assert catalog["anthropic"]
    assert catalog["gemini"]
    assert catalog["ollama"] == []
    assert catalog["current_provider"] == "gemini"
