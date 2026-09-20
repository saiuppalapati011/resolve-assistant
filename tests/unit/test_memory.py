import json

from backend import memory


def test_explicit_memory_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "assistant_memory.json"
    monkeypatch.setattr(memory, "MEMORY_PATH", path)

    item = memory.add_memory("Prefer frame numbers in timeline answers.")

    assert item["text"] == "Prefer frame numbers in timeline answers"
    assert memory.load_memory()["items"][0]["id"] == item["id"]
    assert memory.memory_context() == [{
        "category": "preference",
        "text": "Prefer frame numbers in timeline answers",
    }]

    assert memory.delete_memory(item["id"]) is True
    assert memory.load_memory() == {"items": []}


def test_memory_parser_requires_explicit_request():
    assert memory.extract_explicit_memory("remember that I prefer concise answers") == (
        "I prefer concise answers"
    )
    assert memory.extract_explicit_memory("What do you remember?") is None
