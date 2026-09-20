"""
Reporter node: turns execution_results into a clear plain-language summary.
Explicitly calls out partial failures. Uses an LLM with a deterministic fallback guard.
"""
from __future__ import annotations
import json
from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.llm.errors import redact_secrets
from backend.logging_config import get_logger

logger = get_logger(__name__)


async def run(state: AgentState) -> AgentState:
    results = state.get("execution_results", [])

    if not results:
        return {"final_response": "No actions were executed.", "technical_details": []}

    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]
    cancel_req = state.get("cancel_requested")

    # 1. Build deterministic fallback (the old format)
    lines = []
    if successes:
        lines.append("**Completed successfully:**")
        for r in successes:
            msg = r['message']
            if cancel_req:
                msg += " *(This action had already completed before the cancellation was processed.)*"
            lines.append(f"  ✅ `{r['tool']}` — {msg}")

    if failures:
        lines.append("\n**Failed / Skipped:**")
        for r in failures:
            lines.append(f"  ❌ `{r['tool']}` — {r['message']}")

    planned_count = len(state.get("planned_calls", []))
    executed_count = len(results)
    if executed_count < planned_count:
        skipped = planned_count - executed_count
        if cancel_req:
            lines.append(f"\n⚠️ {skipped} step(s) were skipped due to cancellation.")
        else:
            lines.append(f"\n⚠️ {skipped} step(s) were skipped because an earlier step failed.")

    fallback_summary = "\n".join(lines)

    # 2. Call LLM for natural language summary
    system_prompt = (
        "You are an assistant for DaVinci Resolve. "
        "Describe the outcome of the user's requested actions conversationally based on the execution results. "
        "Do not include raw function names, JSON, or checkmark/cross symbols. "
        "Always explicitly surface any failures or partial failures in plain language. "
        "Never soften a failure or imply success when something failed."
    )

    llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]
    last_msg = llm_messages[-1]["content"]
    llm_messages[-1]["content"] = (
        f"User request: {last_msg}\n\n"
        f"Execution results:\n{json.dumps(results, indent=2)}\n\n"
        f"Was cancellation requested? {bool(cancel_req)}\n\n"
        "Provide the natural language conversational summary."
    )

    try:
        provider = get_provider(state.get("llm_provider"), state.get("llm_model"))
        response = provider.generate(
            messages=llm_messages,
            system=system_prompt,
        )
        llm_summary = response["content"].strip()
        
        # 3. Deterministic Guard
        if failures or (executed_count < planned_count):
            failure_keywords = [
                'fail', 'error', 'could not', "couldn't", 'did not', "didn't",
                'skip', 'cancel', 'unable', 'problem', 'unsuccessful', 'stop'
            ]
            if not any(k in llm_summary.lower() for k in failure_keywords):
                logger.warning("LLM omitted failure. Using deterministic fallback.")
                final_response = f"I wasn't able to complete everything as requested. Here is what happened:\n\n{fallback_summary}"
            else:
                final_response = llm_summary
        else:
            final_response = llm_summary
            
    except Exception as exc:
        logger.error("Reporter LLM generation failed", error=redact_secrets(exc))
        if failures or (executed_count < planned_count):
            final_response = f"I wasn't able to complete everything as requested. Here is what happened:\n\n{fallback_summary}"
        else:
            final_response = f"Here is what happened:\n\n{fallback_summary}"

    logger.info("Reporter summary", successes=len(successes), failures=len(failures))
    return {"final_response": final_response, "technical_details": results}
