"""Small, explicit long-term memory for the Resolve popup assistant.

This deliberately stores only user-approved facts such as preferences and
terminology. It is not a transcript database and does not attempt semantic
search.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


_ROOT = Path(__file__).parent.parent.resolve()
_DEFAULT_PATH = _ROOT / "data" / "assistant_memory.json"
MEMORY_PATH = Path(os.getenv("ASSISTANT_MEMORY_PATH", str(_DEFAULT_PATH)))
_LOCK = RLock()


def _empty_memory() -> dict:
    return {"items": []}


def load_memory() -> dict:
    """Load the small memory profile, returning an empty profile on errors."""
    with _LOCK:
        try:
            if not MEMORY_PATH.exists():
                return _empty_memory()
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            items = data.get("items", []) if isinstance(data, dict) else []
            return {"items": items if isinstance(items, list) else []}
        except (OSError, json.JSONDecodeError, TypeError):
            return _empty_memory()


def _write_memory(data: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MEMORY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(MEMORY_PATH)


def add_memory(text: str, category: str = "preference") -> dict:
    """Add one explicit memory item, avoiding exact duplicates."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" .")
    if not cleaned:
        raise ValueError("Memory text cannot be empty")
    if len(cleaned) > 240:
        raise ValueError("Memory text must be 240 characters or fewer")

    with _LOCK:
        data = load_memory()
        for item in data["items"]:
            if item.get("text", "").casefold() == cleaned.casefold():
                return item
        item = {
            "id": str(uuid.uuid4()),
            "category": category or "preference",
            "text": cleaned,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data["items"].append(item)
        _write_memory(data)
        return item


def delete_memory(memory_id: str) -> bool:
    with _LOCK:
        data = load_memory()
        original = len(data["items"])
        data["items"] = [item for item in data["items"] if item.get("id") != memory_id]
        if len(data["items"]) == original:
            return False
        _write_memory(data)
        return True


def clear_memory() -> None:
    with _LOCK:
        _write_memory(_empty_memory())


def memory_context() -> list[dict]:
    """Return only the fields safe and useful to include in an LLM prompt."""
    return [
        {"category": item.get("category", "preference"), "text": item.get("text", "")}
        for item in load_memory().get("items", [])
        if item.get("text")
    ]


def extract_explicit_memory(message: str) -> str | None:
    """Recognize only an explicit 'remember ...' request."""
    match = re.match(
        r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*[.!]?\s*$",
        message or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    text = match.group(1).strip(" .")
    return text[:240] if text else None
