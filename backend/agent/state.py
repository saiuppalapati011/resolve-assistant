"""Agent state schema for LangGraph."""
from __future__ import annotations
from typing import TypedDict, Literal, Annotated
import operator


class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[list[dict], operator.add]

    # Router output
    intent: Literal["qa", "action", "query"] | None

    # RAG output
    retrieved_docs: list[dict]

    # Planner output
    planned_calls: list[dict]        # [{"tool": str, "args": dict}]

    # Confirmation gate & cancellation
    pending_confirmation: dict | None  # set if any destructive call is planned
    confirmed: bool | None            # True after user confirms, False if cancelled
    cancel_requested: bool | None     # True if user requested to stop execution
    
    # Executor output
    execution_results: list[dict]     # [{"tool": str, "success": bool, "message": str, "data": dict}]

    # Reporter / final output
    final_response: str
    technical_details: list[dict] | None

    # Current LLM provider / model (can change mid-session)
    llm_provider: str
    llm_model: str

    # Explicit long-term memory items. This is a small profile, not chat history.
    global_memory: list[dict]

    # Tool discovery proposal — STRICTLY separate from pending_confirmation.
    # pending_confirmation = "should I execute a known destructive tool?"
    # tool_proposal        = "should I add a newly-discovered method as a tool?"
    # These two decision types must never share the same state field or payload key.
    tool_proposal: dict | None
