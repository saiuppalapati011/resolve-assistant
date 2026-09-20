"""Shared memory formatting utility for agent nodes."""
from __future__ import annotations


def format_memory(memory: list[dict] | None) -> str:
    """Keep the prompt addition small and readable."""
    if not memory:
        return "(no saved preferences or terminology)"
    return "\n".join(
        f"- {item.get('category', 'preference')}: {item.get('text', '')}"
        for item in memory[:20]
        if item.get("text")
    ) or "(no saved preferences or terminology)"
