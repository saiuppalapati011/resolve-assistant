"""
Planner node: retrieves relevant tool signatures from RAG, then decomposes
the user's request into an ordered list of tool calls.
"""
from __future__ import annotations

import json
import re
import ast

from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.rag.retriever import Retriever
from backend.logging_config import get_logger
from backend.resolve.mcp_client import get_mcp_client

logger = get_logger(__name__)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


ALLOWLISTED_DOMAINS = [
    "project_management",
    "media_pool",
    "timeline_operations",
    "render_delivery",
    "clip_properties",
    "timeline_item_editing"
]

def _build_system(tool_list_str: str, available_domains: list[str]) -> str:
    return f"""You are a planner for a DaVinci Resolve automation agent.

Available tools (ONLY use tools from this list):
{tool_list_str}

If the user requests an action that requires a domain not currently active, you can activate one of these allowed domains:
{available_domains}
To do so, plan a single call to `davinci-resolve_activate_domain` with the domain name. Do NOT plan any other actions in the same step.
CRITICAL: You are STRICTLY PROHIBITED from activating or using any domains not in the allowed list above. If a user asks for something outside these domains (e.g., color grading, Fairlight audio mixing, or AI features), you MUST reject the request.

═══ DaVinci Resolve API Conventions — READ CAREFULLY ═══

1. TIMELINE NAME REQUIRED FOR TRACK/ITEM TOOLS
   Many tools (get_items_in_track, get_track_count, get_track_name, add_track, delete_track,
   get_track_enabled, lock_track, add_timeline_marker, get_timeline_markers, etc.) have a
   required argument called `name`. This is the TIMELINE NAME, not the track name.
   - If the user hasn't specified a timeline name, you MUST first call `get_current_timeline`
     to retrieve it, then use the returned name in the subsequent tool call.
   - Example multi-step plan for "list clips on video track 1":
       Step 1: get_current_timeline  (returns the timeline name, e.g. "AI Test Timeline")
       Step 2: get_items_in_track(name="AI Test Timeline", track_type="video", track_index=1)

2. TRACK INDICES ARE 1-BASED
   `track_index` is always 1-based (first track = 1, not 0).
   "Video track" with no number specified → track_index=1.
   "Second video track" → track_index=2.

3. TRACK TYPE VALUES
   Use exactly: "video", "audio", or "subtitle" (lowercase).

4. CLIP ITEM INDICES ARE 1-BASED
   When tools accept `item_index`, use the 1-based index returned by the MCP
   contract. "first clip" → item_index=1, "third clip" → item_index=3.

5. MULTI-STEP CHAINING
   When step N requires a value that step N-1 returns (e.g., timeline name, frame count),
   always plan both steps in sequence. The executor runs them in order and stops on failure.

6. SWITCH PAGE — EXACT ENUM VALUES
   `switch_page` requires `page` to be EXACTLY one of (lowercase):
   "media", "cut", "edit", "fusion", "color", "fairlight", "deliver"
   Example: switch to edit page → {{"tool": "davinci-resolve_switch_page", "args": {{"page": "edit"}}}}

7. ADD MARKER DEFAULTS
   `add_timeline_marker` has these required fields. Use these defaults when user doesn't specify:
   - `marker_name`: ""  (empty string)
   - `custom_data`: ""  (empty string)
   - `duration`: 1      (one frame)
   - `frame_id`: the frame number the user specified
   - `color`: the color the user specified (capitalize first letter: "Red", "Blue", "Green", etc.)
   - `note`: the note text the user specified (or "" if none)

8. GET TIMELINE SETTINGS — NULLABLE FIELD
   `get_timeline_settings` requires `setting_name` but it can be null.
   To get ALL settings: {{"name": "<timeline>", "setting_name": null}}
   To get frame rate: {{"name": "<timeline>", "setting_name": "timelineFrameRate"}}

═══════════════════════════════════════════════════════

Your task: decompose the user's action request into an ordered sequence of tool calls.
Respond with ONLY valid JSON in this exact format when you have a plan:
{{
  "planned_calls": [
    {{"tool": "<tool_name>", "args": {{<arg_name>: <value>, ...}}}},
    ...
  ]
}}

Rules:
- Only use tool names that appear in the Available tools list above, OR `davinci-resolve_activate_domain` if activating a domain.
- Never invent tool names or argument names not in the schema.
- **IMPORTANT**: If the user refers to an item by its ordinal position (e.g., "the third clip"), you MUST convert it to the MCP's 1-based index (e.g., `item_index: 3`).
- **IMPORTANT**: For any tool whose schema requires `name` (timeline name), always supply it. If unknown, add a preceding `get_current_timeline` call.

CRITICAL — When you cannot produce a plan, you MUST return a structured reason code:

  If the user's intent requires an action that doesn't exist in the Available tools list and isn't in an allowed domain:
    {{"planned_calls": [], "reason": "out_of_scope", "error": "That's not something I can help with yet."}}

  If all tools that could satisfy the request exist, but the user has not provided required arguments:
    {{"planned_calls": [], "reason": "missing_arguments", "error": "<natural clarifying question>"}}
"""



async def run(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]

    # 1. Retrieve relevant API doc chunks to ground the planner
    try:
        docs = _get_retriever().query(user_message, k=6)
    except RuntimeError:
        docs = []

    doc_context = "\n\n".join(d["text"] for d in docs[:4]) if docs else "(no docs available)"

    # 2. Build the available tools list and available domains
    mcp_client = get_mcp_client()
    tools = await mcp_client.get_tools()
    
    # Format schemas so the LLM sees required fields prominently
    all_schemas = {}
    valid_tool_names = set()
    list_domains_tool = None
    for t in tools:
        if t.name == "davinci-resolve_list_domains":
            list_domains_tool = t
        valid_tool_names.add(t.name)
        s = getattr(t, "inputSchema", None)
        if s is None:
            schema_obj = getattr(t, "args_schema", None)
            if isinstance(schema_obj, dict):
                s = schema_obj
            elif schema_obj is not None and hasattr(schema_obj, "schema"):
                s = schema_obj.schema()
            else:
                s = {}
        required = s.get("required", [])
        props = s.get("properties", {})
        all_schemas[t.name] = {
            "description": t.description,
            "required_args": required,           # ← explicit required list for the LLM
            "optional_args": [k for k in props if k not in required],
            "arg_details": {
                k: {"type": v.get("type", "string"),
                    **({"enum": v["enum"]} if "enum" in v else {}),
                    **({"description": v["description"]} if "description" in v else {})}
                for k, v in props.items()
            },
        }
    tool_list_str = json.dumps(all_schemas, indent=2)

    # Check deactivated domains
    domains_result = await list_domains_tool.ainvoke({}) if list_domains_tool else None

    inactive_allowed = []
    try:
        if domains_result and not getattr(domains_result, "is_error", False):
            # _ToolResult has .text; plain str is also accepted
            if hasattr(domains_result, "text"):
                text = domains_result.text
            elif isinstance(domains_result, str):
                text = domains_result
            else:
                text = str(domains_result)

            # Try JSON first, fall back to ast.literal_eval for Python repr strings
            try:
                domain_list = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                domain_list = ast.literal_eval(text)

            for d in domain_list:
                if not d.get("active") and d.get("name") in ALLOWLISTED_DOMAINS:
                    inactive_allowed.append(d.get("name"))
    except Exception as exc:
        logger.error("Failed to parse domains_result", error=str(exc))


    llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]
    last_msg = llm_messages[-1]["content"]
    llm_messages[-1]["content"] = (
        f"Relevant documentation:\n{doc_context}\n\n"
        f"User request: {last_msg}\n\n"
        "Produce the planned_calls JSON now."
    )

    provider = get_provider(state.get("llm_provider"), state.get("llm_model"))

    try:
        response = provider.generate(
            messages=llm_messages,
            system=_build_system(tool_list_str, inactive_allowed),
        )
        raw = response["content"].strip()

        # Extract JSON even if the model wraps it in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        plan_data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Planner failed to parse LLM output", error=str(exc))
        return {
            "planned_calls": [],
            "final_response": f"I couldn't plan the action: {exc}. Please rephrase your request.",
        }

    # 4. Validate every tool name AND required args against active tool schemas
    raw_calls = plan_data.get("planned_calls", [])
    validated_calls: list[dict] = []
    rejected: list[str] = []

    # ALL tools whose required 'name' arg means the TIMELINE NAME
    # (every tool that operates on a specific timeline by name)
    TIMELINE_NAME_ARG_TOOLS = {
        t for t in valid_tool_names
        if any(kw in t for kw in [
            "track", "timeline_marker", "timeline_mark", "items_in_track",
            "rename_timeline", "get_timeline_settings", "set_timeline_setting",
            "get_timeline_start_frame", "get_timeline_end_frame",
            "get_timeline_markers", "get_timeline_marks", "set_timeline_marks",
            "get_timeline_node_graph", "get_timeline_media_pool_item",
            "duplicate_timeline", "export_timeline", "import_into_timeline",
            "get_start_timecode", "set_start_timecode",
            "switch_timeline",  # switch_timeline takes a timeline name as 'name'
        ])
    }

    # Fields that are declared 'required' in the schema but can safely be
    # defaulted when the user doesn't provide them
    SMART_DEFAULTS: dict[str, dict] = {
        "davinci-resolve_add_timeline_marker": {
            "custom_data": "",
            "marker_name": "",
            "duration": 1,
        },
        "davinci-resolve_add_clip_marker": {
            "custom_data": "",
            "marker_name": "",
            "duration": 1,
        },
        "davinci-resolve_add_item_marker": {
            "custom_data": "",
            "marker_name": "",
            "duration": 1,
        },
    }

    for i, call in enumerate(raw_calls):
        tool_name = call.get("tool", "")
        if tool_name not in valid_tool_names and tool_name != "davinci-resolve_activate_domain":
            rejected.append(tool_name)
            logger.warning("Planner proposed unknown tool — rejected", tool=tool_name)
            continue

        # ── Required-field validation & auto-repair ──────────────────────────
        schema = all_schemas.get(tool_name, {})
        required_fields = schema.get("required_args", [])
        arg_details = schema.get("arg_details", {})
        args = call.get("args", {})

        # Apply smart defaults for known "required but safely defaultable" fields
        if tool_name in SMART_DEFAULTS:
            for field, default_val in SMART_DEFAULTS[tool_name].items():
                if field not in args or args[field] is None:
                    args[field] = default_val
                    logger.info("Smart default applied", tool=tool_name, field=field, value=default_val)
            call["args"] = args

        # Identify truly missing fields — exclude nullable fields (type includes 'null')
        def _is_nullable(field: str) -> bool:
            detail = arg_details.get(field, {})
            t = detail.get("type", "string")
            return isinstance(t, list) and "null" in t

        missing = [
            f for f in required_fields
            if not _is_nullable(f) and (f not in args or args[f] is None or args[f] == "")
        ]

        if missing:
            # Special case: only 'name' missing → it means timeline name → auto-repair
            if missing == ["name"] and tool_name in TIMELINE_NAME_ARG_TOOLS:
                already_prefixed = (
                    i > 0 and raw_calls[i - 1].get("tool") == "davinci-resolve_get_current_timeline"
                )
                if not already_prefixed:
                    validated_calls.append({"tool": "davinci-resolve_get_current_timeline", "args": {}})
                    logger.info("Auto-repaired: prepended get_current_timeline", for_tool=tool_name)
                args["name"] = "__prev_result__"
                call["args"] = args
                logger.info("Auto-repaired: set name=__prev_result__", tool=tool_name)
            else:
                missing_str = ", ".join(f"`{f}`" for f in missing)
                return {
                    "planned_calls": [],
                    "final_response": (
                        f"To complete this action I need more information. "
                        f"The tool `{tool_name}` requires: {missing_str}. "
                        f"Please provide these details and try again."
                    ),
                }

        validated_calls.append(call)

    if plan_data.get("reason") or plan_data.get("error"):
        reason = plan_data.get("reason", "out_of_scope")
        error_text = plan_data.get("error", "")
        logger.info("Planner returned structured result", reason=reason, error=error_text[:120])

        if reason == "missing_arguments":
            # Route directly to clarifying question — no gap detection
            return {
                "planned_calls": [],
                "final_response": error_text,
                "tool_proposal": None,
            }

        elif reason == "no_matching_tool":
            # Flat decline
            return {
                "planned_calls": [],
                "final_response": f"I don't currently have a tool that can do that. {error_text}",
            }

        else:  # out_of_scope or legacy error without reason
            return {
                "planned_calls": [],
                "final_response": error_text,
                "tool_proposal": None,
            }

    if rejected:
        logger.warning("Some planned tools were rejected as unknown", rejected=rejected)

    # Deterministic repair for the common current-playhead question. Some
    # models satisfy the timeline-name prerequisite but omit the actual
    # get_current_timecode call, producing a misleading answer containing
    # only the current timeline name.
    timecode_request = bool(re.search(r"\b(playhead|current\s+timecode|timecode\s+position)\b", user_message, re.IGNORECASE))
    current_timecode_tool = "davinci-resolve_get_current_timecode"
    current_timeline_tool = "davinci-resolve_get_current_timeline"
    if timecode_request and current_timecode_tool in valid_tool_names:
        if not any(c.get("tool") == current_timecode_tool for c in validated_calls):
            if not any(c.get("tool") == current_timeline_tool for c in validated_calls):
                validated_calls.insert(0, {"tool": current_timeline_tool, "args": {}})
            validated_calls.append({"tool": current_timecode_tool, "args": {"name": "__prev_result__"}})
            logger.info("Auto-repaired current timecode plan")

    # 5. Post-planning validation for ordinals
    validation_error = validate_ordinals(user_message, validated_calls)
    if validation_error:
        return {
            "planned_calls": [],
            "final_response": validation_error["error"],
        }

    logger.info("Planner produced plan", calls=len(validated_calls))
    return {
        "retrieved_docs": docs,
        "planned_calls": validated_calls,
        "pending_confirmation": None,
        "confirmed": None,
    }


def validate_ordinals(user_message: str, planned_calls: list[dict]) -> dict | None:
    """
    Validates that the LLM correctly converted clip ordinals (first, second,
    third, etc.) to the MCP contract's 1-based ``item_index``. Track indices
    are also named ``*_index`` but are numeric track identifiers, not ordinal
    clip positions, so they must never be rewritten here.
    Returns an error dict if validation fails, or None if successful.
    """
    ordinals = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8,
        "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
        "last": -1
    }
    
    words = [w.strip(".,!?\\\"'") for w in user_message.lower().split()]
    found_ordinals = [ordinals[w] for w in words if w in ordinals]
            
    if not found_ordinals:
        return None
        
    for call in planned_calls:
        args = call.get("args", {})
        for k, v in args.items():
            if isinstance(v, int) and k.lower() == "item_index":
                if len(found_ordinals) == 1:
                    expected = found_ordinals[0]
                    if v != expected:
                        logger.info(f"Ordinal validator auto-correcting {k} from {v} to {expected}")
                        args[k] = expected
                else:
                    if v not in found_ordinals:
                        return {"error": f"The requested action references multiple ordinals. Please clarify the exact position for '{k}'."}
                        
    return None
