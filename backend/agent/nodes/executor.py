"""
Executor node: runs each planned tool call in order via the MCP client.
Records per-call results; never raises — all errors become ToolResult failures.
"""
from __future__ import annotations

import time
import re
import inspect

from backend.agent.state import AgentState
from backend.logging_config import get_logger
from backend.resolve.mcp_client import get_mcp_client
from backend.agent.nodes.confirmation import is_destructive

logger = get_logger(__name__)

from langchain_core.runnables.config import RunnableConfig

KNOWN_ERRORS = {
    r"No project is currently open": "No project is currently open. Please open a project first.",
    r"Not connected to DaVinci Resolve": "Not connected to DaVinci Resolve. Please ensure DaVinci Resolve is running and scripting is enabled.",
    r"Failed to get Media Pool": "Could not access the Media Pool.",
    r"Timeline '.*?' not found": "The requested timeline could not be found.",
    r"No current timeline": "No timeline is currently active.",
    r"Failed to get root folder": "Could not access the root folder.",
}

FAILURE_RESPONSES = {
    "failed",
    "delete failed",
    "rename failed",
    "duplicate failed",
    "update failed",
    "setting update failed",
    "mcp tool failed",
}

def map_error_message(raw_msg: str) -> str:
    """Map DaVinci Resolve error messages to plain-language user messages."""
    prefix = "DaVinci Resolve error: "
    if raw_msg.startswith(prefix):
        core_msg = raw_msg[len(prefix):].strip()
        for pattern, replacement in KNOWN_ERRORS.items():
            if re.match(pattern, core_msg):
                return replacement
        # Graceful fallback for unknown patterns
        return f"DaVinci Resolve reported an issue: {core_msg}"
    
    # Generic fallback
    return raw_msg

async def run(state: AgentState, config: RunnableConfig) -> AgentState:
    planned_calls = state.get("planned_calls", [])
    results: list[dict] = []
    
    from backend.agent.graph import get_graph
    graph = await get_graph()
    mcp_client = get_mcp_client()
    
    tools = await mcp_client.get_tools()
    tool_dict = {t.name: t for t in tools}
    def _get_schema(t):
        s = getattr(t, "inputSchema", None)
        if s is not None:
            return s
        schema_obj = getattr(t, "args_schema", None)
        if schema_obj is None:
            return {}
        # args_schema can be a Pydantic model class, a dict, or None
        if isinstance(schema_obj, dict):
            return schema_obj
        if hasattr(schema_obj, "schema"):
            return schema_obj.schema()
        return {}
    schemas = {t.name: _get_schema(t) for t in tools}

    current_state: dict = {}
    prev_result_text: str = ""   # output text from the last successful tool call
    successful_result_texts: list[str] = []
    current_timeline_text: str = ""

    for call in planned_calls:
        # Strip checkpoint_id to ensure we get the absolute latest state
        thread_config = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
        current_state = (await graph.aget_state(thread_config)).values or {}

        if current_state.get("cancel_requested"):
            logger.warning("Execution cancelled by user. Skipping remaining tools.")
            break

        tool_name = call.get("tool", "")
        args = call.get("args", {})

        # ── Result chaining ────────────────────────────────────────────────────
        # If any arg value is the sentinel "__prev_result__", substitute the
        # text output of the previous successful tool call. This lets the planner
        # write multi-step plans like:
        #   step1: get_current_timeline → returns "AI Test Timeline"
        #   step2: get_items_in_track(name="__prev_result__", ...)
        for key, val in list(args.items()):
            # Support the placeholder spellings used by older planner
            # prompts and by different LLMs. All currently represent the
            # immediately preceding successful tool result.
            value_text = val.lower() if isinstance(val, str) else ""
            indexed_match = (
                re.fullmatch(r"\$prev\[(\d+)\]", val, re.IGNORECASE)
                if isinstance(val, str)
                else None
            )
            is_previous_result = (
                val == "__prev_result__"
                or value_text == "$prev_result"
                or indexed_match is not None
            )
            is_current_timeline_alias = value_text in {
                "current_timeline",
                "__current_timeline__",
                "$current_timeline",
            }
            if is_previous_result or is_current_timeline_alias:
                if is_current_timeline_alias:
                    resolved_value = current_timeline_text or prev_result_text
                elif indexed_match:
                    result_index = int(indexed_match.group(1))
                    resolved_value = (
                        successful_result_texts[result_index]
                        if result_index < len(successful_result_texts)
                        else ""
                    )
                elif key == "name" and current_timeline_text:
                    # A timeline-dependent step should continue using the
                    # timeline returned by get_current_timeline, even when a
                    # model uses the generic previous-result sentinel after
                    # an intervening count/query step.
                    resolved_value = current_timeline_text
                else:
                    resolved_value = prev_result_text
                args[key] = resolved_value
                logger.info("Result chaining applied", arg=key, value=resolved_value)

        if tool_name not in schemas:
            result = {
                "tool": tool_name,
                "success": False,
                "message": f"Tool '{tool_name}' not found.",
                "data": {},
            }
            results.append(result)
            logger.error("Unknown tool in executor", tool=tool_name)
            continue

        # Defense in depth: the graph normally pauses in the confirmation
        # node, but execution must remain safe even if a caller invokes this
        # node directly or a future graph change bypasses that edge.
        if (
            tool_name.startswith("davinci-resolve_")
            and is_destructive(tool_name)
            and current_state.get("confirmed") is not True
        ):
            result = {
                "tool": tool_name,
                "args": args,
                "success": False,
                "message": "Confirmation is required before this destructive action can execute.",
                "data": {},
                "elapsed_s": 0.0,
            }
            results.append(result)
            logger.warning("Executor blocked destructive tool without confirmation", tool=tool_name)
            break

        # ── Required-field guard ───────────────────────────────────────────────
        # Last line of defence: check all JSON Schema `required` fields are
        # present before we even touch the MCP server.  This catches anything
        # that slipped past the planner's post-validation (e.g. __prev_result__
        # that was empty, or a field the LLM forgot on the first call).
        schema = schemas.get(tool_name, {})
        required_fields = schema.get("required", [])
        props = schema.get("properties", {})

        def _field_is_nullable(f: str) -> bool:
            """True when the schema type includes 'null' (e.g. ['string','null'])."""
            t = props.get(f, {}).get("type", "string")
            return isinstance(t, list) and "null" in t

        missing_required = [
            f for f in required_fields
            if not _field_is_nullable(f) and (f not in args or (args[f] is None and not _field_is_nullable(f)) or args[f] == "")
        ]
        if missing_required:
            missing_str = ", ".join(f"'{f}'" for f in missing_required)
            result = {
                "tool": tool_name,
                "args": args,
                "success": False,
                "message": (
                    f"Missing required argument(s) for '{tool_name}': {missing_str}. "
                    f"Please provide the missing information and try again."
                ),
                "data": {},
                "elapsed_s": 0.0,
            }
            results.append(result)
            logger.error(
                "Executor blocked: missing required fields",
                tool=tool_name,
                missing=missing_required,
            )
            break   # stop the plan — subsequent steps likely depend on this one

        # Parameter validation logic using the MCP schemas (reuse schema fetched above)
        props = schema.get("properties", {})

        validation_failed = False
        original_args = args.copy()
        
        for arg_name, arg_val in args.items():
            if not isinstance(arg_val, str):
                continue
            prop = props.get(arg_name, {})
            allowed = prop.get("enum")
            if allowed:
                # 1. Exact match check
                if arg_val in allowed:
                    continue
                # 2. Case-insensitive normalization
                lower_val = arg_val.strip().lower()
                normalized = next((a for a in allowed if isinstance(a, str) and a.lower() == lower_val), None)
                if normalized:
                    logger.info("Normalized parameter", param=arg_name, original=arg_val, normalized=normalized)
                    args[arg_name] = normalized
                else:
                    # 3. No match found
                    validation_failed = True
                    result = {
                        "tool": tool_name,
                        "args": args,
                        "success": False,
                        "message": f"Invalid value '{arg_val}' for '{arg_name}'. Expected one of: {', '.join([str(a) for a in allowed])}.",
                        "data": {"original_args": original_args, "normalized_args": args},
                        "elapsed_s": 0.0,
                    }
                    results.append(result)
                    logger.warning("Tool execution blocked by parameter validation", tool=tool_name, param=arg_name, value=arg_val)
                    break
        
        if validation_failed:
            break

        # Execute — call the underlying coroutine directly so kwargs are
        # always unpacked correctly (LangChain's ainvoke may not unpack
        # when args_schema=None).
        t_start = time.monotonic()
        try:
            target_tool = tool_dict.get(tool_name)
            # Try direct coroutine call first; fall back to ainvoke
            coroutine = getattr(target_tool, "coroutine", None)
            if coroutine is not None and inspect.iscoroutinefunction(coroutine):
                mcp_res = await coroutine(**args)
            else:
                mcp_res = await target_tool.ainvoke(args)
            elapsed = round(time.monotonic() - t_start, 3)

            # _ToolResult is returned by our persistent-session tool wrapper
            if hasattr(mcp_res, "is_error") and hasattr(mcp_res, "text"):
                output_text = mcp_res.text
                is_error = mcp_res.is_error
            elif hasattr(mcp_res, "isError") and hasattr(mcp_res, "content"):
                output_text = "\n".join(getattr(item, "text", str(item)) for item in (mcp_res.content or []))
                is_error = bool(mcp_res.isError) or output_text.startswith(("DaVinci Resolve error:", "Unexpected error:", "Error:"))
            elif isinstance(mcp_res, str):
                output_text = mcp_res
                is_error = (
                    output_text.startswith(("DaVinci Resolve error:", "Unexpected error:", "Error:"))
                    or output_text.strip().lower() in FAILURE_RESPONSES
                    or output_text.startswith("DESTRUCTIVE")
                )
            else:
                output_text = str(mcp_res)
                is_error = False

            if is_error or output_text.startswith("DaVinci Resolve error:"):
                success = False
                message = map_error_message(output_text)
                data = {}
            else:
                success = True
                message = output_text
                data = {}

            result = {
                "tool": tool_name,
                "args": args,
                "success": success,
                "message": message,
                "data": {**data, "original_args": original_args, "normalized_args": args},
                "elapsed_s": elapsed,
            }
        except Exception as exc:
            elapsed = round(time.monotonic() - t_start, 3)
            logger.error("Executor caught unexpected exception", tool=tool_name, error=str(exc))
            result = {
                "tool": tool_name,
                "args": args,
                "success": False,
                "message": f"Unexpected error in {tool_name}: {exc}",
                "data": {},
                "elapsed_s": elapsed,
            }

        logger.info(
            "Tool executed",
            tool=tool_name,
            success=result["success"],
            elapsed_s=result.get("elapsed_s"),
            message=result["message"],
        )
        results.append(result)

        # Track the last successful output for result chaining (__prev_result__)
        if result["success"]:
            prev_result_text = result["message"]
            successful_result_texts.append(result["message"])
            if tool_name == "davinci-resolve_get_current_timeline":
                current_timeline_text = result["message"]

        # Stop the plan early if a critical step fails
        if not result["success"]:
            logger.warning("Stopping plan early due to tool failure", tool=tool_name)
            break

    return {"execution_results": results, "cancel_requested": current_state.get("cancel_requested")}
