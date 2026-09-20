import pytest

from backend.agent.nodes import rag


class _FakeRetriever:
    def query(self, text):
        return [{"text": "irrelevant excerpt", "source": "manual.pdf", "score": 1.2, "idx": 1}]


class _FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, messages, system=None):
        self.calls.append(messages[-1]["content"])
        if len(self.calls) == 1:
            return {"content": "The provided documentation does not contain this information."}
        return {"content": "Use the Resolve scripting API page method.\n\nSources: https://example.com/resolve"}


@pytest.mark.asyncio
async def test_docs_are_tried_before_web_fallback(monkeypatch):
    provider = _FakeProvider()
    monkeypatch.setattr(rag, "_get_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(rag, "get_provider", lambda *args: provider)
    monkeypatch.setattr(
        rag,
        "search_web",
        lambda query: [{"title": "Official Resolve guide", "url": "https://example.com/resolve", "snippet": "Page API"}],
    )

    result = await rag.run({
        "messages": [{"role": "user", "content": "How do I switch pages?"}],
        "llm_provider": "gemini",
        "llm_model": "test",
        "global_memory": [],
    })

    assert len(provider.calls) == 2
    assert "Web search results" in provider.calls[1]
    assert result["final_response"].startswith("Use the Resolve scripting API")


def test_web_fallback_trigger_detection():
    assert rag._needs_web_fallback("The documentation does not contain this information.")
    assert not rag._needs_web_fallback("The API uses GetCurrentProject().")

