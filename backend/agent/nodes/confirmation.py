"""
Confirmation gate node.
If any planned tool is tagged destructive, this node pauses execution
and sets pending_confirmation so the frontend can show confirm/cancel UI.
"""
from __future__ import annotations
import json
import yaml
from pathlib import Path
from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.logging_config import get_logger

logger = get_logger(__name__)

_classification_cache: dict | None = None

def _load_classification() -> dict:
    """Load (and cache) the tool classification YAML. Reload if file changes."""
    global _classification_cache
    if _classification_cache is not None:
        return _classification_cache
    yaml_path = Path(__file__).parent.parent.parent.parent / "data" / "mcp_tool_classification.yaml"
    if not yaml_path.exists():
        _classification_cache = {}
        return _classification_cache
    try:
        with open(yaml_path, "r") as f:
            _classification_cache = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error(f"Failed to load classification yaml: {exc}")
        _classification_cache = {}
    return _classification_cache

def invalidate_classification_cache():
    """Call this after writing a new mcp_tool_classification.yaml."""
    global _classification_cache
    _classification_cache = None

def is_destructive(tool_name: str) -> bool:
    """
    Check if a tool is destructive based on data/mcp_tool_classification.yaml.
    If the tool is missing or its value is null, default to True (fail-safe).
    """
    data = _load_classification()
    if not data:
        return True  # no YAML → fail-safe
    entry = data.get(tool_name)
    if entry is None:
        return True
    val = entry.get("destructive")
    if val is None:
        return True
    return bool(val)


async def run(state: AgentState) -> AgentState:
    planned_calls = state.get("planned_calls", [])

    # Find all destructive calls in the plan
    destructive_calls = []
    for call in planned_calls:
        tool_name = call.get("tool")
        destructive = is_destructive(tool_name)
        logger.debug(f"Confirmation gate check: tool='{tool_name}', destructive={destructive}")
        if destructive:
            destructive_calls.append(call)

    if not destructive_calls:
        # No confirmation needed — proceed directly
        return {"pending_confirmation": None}

    # Generate a natural language confirmation string using LLM
    provider = get_provider(state.get("llm_provider"), state.get("llm_model"))
    
    system_prompt = (
        "You are an assistant for DaVinci Resolve. The user has requested an action that includes "
        "destructive operations (like deleting clips, modifying tracks). "
        "Write a short, conversational confirmation message asking if the user wants to proceed. "
        "CRITICAL: Always name the specific target of the action (clip name, track, index, timeline name, file path) "
        "so the user can verify this is what they meant. Never generalize to just 'a destructive operation'. "
        "End by asking if they want to go ahead."
    )
    
    prompt = f"Planned tool calls:\n{json.dumps(planned_calls, indent=2)}\n\nProvide the natural language confirmation."
    
    try:
        response = provider.generate(
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
        )
        confirmation_msg = response["content"].strip()
    except Exception as exc:
        logger.error(f"Confirmation LLM generation failed: {exc}")
        # Fallback
        descriptions = []
        for call in planned_calls:
            tool = call.get("tool", "")
            args = call.get("args", {})
            tag = " ⚠️ DESTRUCTIVE" if is_destructive(tool) else ""
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            descriptions.append(f"• `{tool}({args_str})`{tag}")
            
        confirmation_msg = (
            "⚠️ This action includes destructive operations that cannot be undone. "
            "Please confirm you want to proceed:\n\n" + "\n".join(descriptions)
        )

    confirmation_payload = {
        "message": confirmation_msg,
        "details": planned_calls,
        "destructive_tools": [c["tool"] for c in destructive_calls],
    }

    logger.info(
        "Confirmation required",
        destructive_tools=[c["tool"] for c in destructive_calls],
    )
    return {"pending_confirmation": confirmation_payload, "confirmed": None}
