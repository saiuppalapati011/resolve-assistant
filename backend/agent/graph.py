"""
LangGraph agent graph definition.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes import router, rag, planner, confirmation, executor, reporter


def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router",       router.run)
    graph.add_node("rag",          rag.run)
    graph.add_node("planner",      planner.run)

    graph.add_node("confirmation", confirmation.run)
    graph.add_node("executor",     executor.run)
    graph.add_node("reporter",     reporter.run)

    # Entry point
    graph.set_entry_point("router")

    # Router → RAG or Planner
    graph.add_conditional_edges(
        "router",
        lambda s: "end" if s.get("cancel_requested") else s["intent"],
        {"qa": "rag", "action": "planner", "end": END},
    )

    # RAG → END (answer is in final_response)
    graph.add_edge("rag", END)

    # Planner → Confirmation (if any destructive calls), END (if tool_proposal), or Executor directly
    graph.add_conditional_edges(
        "planner",
        lambda s: (
            "end" if s.get("cancel_requested")
            else "end" if not s.get("planned_calls") and not s.get("tool_proposal")
            else "end" if not s.get("planned_calls") and s.get("tool_proposal")  # halt for proposal
            else "confirmation"
        ),
        {"confirmation": "confirmation", "end": END},
    )

    # Confirmation gate:
    # - If cancel_requested is true     → cancel (END)
    # - If pending_confirmation is set  → halt and return to frontend
    # - If confirmed == True            → proceed to executor
    # - If confirmed == False           → cancel (END)
    # - If no confirmation needed       → executor
    graph.add_conditional_edges(
        "confirmation",
        lambda s: (
            "cancel"   if s.get("cancel_requested")
            else "halt"     if s.get("pending_confirmation") and s.get("confirmed") is None
            else "executor" if s.get("confirmed") is True
            else "cancel"   if s.get("confirmed") is False
            else "executor"
        ),
        {
            "halt":     END,      # Paused — frontend will resume with confirm/cancel
            "executor": "executor",
            "cancel":   END,
        },
    )

    graph.add_edge("executor", "reporter")
    graph.add_edge("reporter", END)

    return graph


# Module-level compiled graph (lazy)
_graph = None
_db_conn = None


async def get_graph():
    global _graph, _db_conn
    if _graph is None:
        import os
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "chats.sqlite")

        _db_conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(_db_conn)
        await saver.setup()

        _graph = build_graph().compile(checkpointer=saver)
    return _graph


async def close_graph():
    """Close the aiosqlite connection cleanly on shutdown."""
    global _graph, _db_conn
    if _db_conn is not None:
        await _db_conn.close()
        _db_conn = None
    _graph = None
