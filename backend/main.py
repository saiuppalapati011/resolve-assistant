"""
FastAPI backend entrypoint.

Routes:
  GET  /                          → serves frontend/index.html
  GET  /health                    → Resolve connection status
  GET  /api/models                → available LLM models for UI dropdown
  POST /api/set-model             → switch active model without restart
  WS   /ws/chat                   → main conversation WebSocket
"""
from __future__ import annotations

import json
import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from contextlib import asynccontextmanager
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.graph import get_graph
from backend.agent.state import AgentState
from backend.config import settings
from backend.llm.factory import get_model_catalog, get_provider
from backend.logging_config import setup_logging, get_logger
from backend.agent.nodes.confirmation import invalidate_classification_cache


setup_logging()
logger = get_logger(__name__)


async def _ensure_chat_titles_table() -> None:
    """Create app-owned chat metadata after the LangGraph DB is available."""
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return
    await _db_conn.execute(
        """CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            custom INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )"""
    )
    await _db_conn.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.resolve.mcp_client import init_mcp_client, close_mcp_client

    # Start the persistent MCP subprocess, activate MVP domains, cache tools
    mcp_client = await init_mcp_client()

    # Chat titles are application metadata and intentionally live outside
    # LangGraph's checkpoint schema.
    # get_graph() must run first because it opens the aiosqlite connection.
    await get_graph()
    await _ensure_chat_titles_table()

    # Build / update data/mcp_tool_classification.yaml with the post-activation tool list
    tools = await mcp_client.get_tools()
    yaml_path = Path(__file__).parent.parent / "data" / "mcp_tool_classification.yaml"
    existing_data = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            existing_data = yaml.safe_load(f) or {}

    for tool in tools:
        if tool.name not in existing_data:
            if tool.name == "davinci-resolve_activate_domain":
                existing_data[tool.name] = {"destructive": False}
            else:
                existing_data[tool.name] = {"destructive": None}

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(existing_data, f, default_flow_style=False)
    # Invalidate the in-memory cache so the new entries are picked up
    invalidate_classification_cache()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    from backend.agent.graph import close_graph
    await close_graph()
    await close_mcp_client()
    logger.info("MCP client and database connection closed cleanly.")


app = FastAPI(title="Resolve AI Assistant", version="1.0.0", lifespan=lifespan)

_FRONTEND = Path(__file__).parent.parent / "frontend"

# Serve static frontend files
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Active session state (Multi-user / Multi-thread via SQLite) ─────────────

_current_provider = settings.llm_provider
_current_model    = settings.llm_model
_active_threads: set[str] = set()
_active_threads_guard = asyncio.Lock()

# ── HTTP routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = _FRONTEND / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>Frontend not found. Run from project root.</h1>", status_code=404)


@app.get("/health")
async def health():
    """Check backend status."""
    from backend.resolve.mcp_client import is_resolve_connected
    connected = is_resolve_connected()
    return {
        "status": "ok",
        "message": "Backend running" if connected else "Backend running — DaVinci Resolve not connected",
        "resolve_connected": connected,
        "resolve_version": "Managed by MCP" if connected else "Not connected",
        "api_level": "Managed by MCP" if connected else "N/A",
    }


@app.get("/api/models")
async def get_models():
    """Return available models for all providers."""
    return get_model_catalog(_current_provider)


@app.post("/api/set-model")
async def set_model(body: dict):
    """
    Switch the active LLM provider and/or model.
    Body: {"provider": "anthropic"|"ollama", "model": "<model_id>"}
    """
    global _current_provider, _current_model
    provider = body.get("provider", _current_provider)
    model    = body.get("model",    _current_model)

    if provider not in {"anthropic", "gemini", "ollama"} or not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="Invalid provider or model")

    _current_provider = provider
    _current_model    = model

    logger.info("Model switched", provider=provider, model=model)
    return {"ok": True, "provider": provider, "model": model}


@app.get("/api/chats")
async def list_chats():
    """List conversations with stable human-readable titles."""
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return {"threads": []}
    await _ensure_chat_titles_table()
    
    try:
        async with _db_conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
        ) as cur:
            rows = await cur.fetchall()
        threads = []
        for (thread_id,) in rows:
            title = await _get_or_create_chat_title(thread_id)
            threads.append({"id": thread_id, "title": title})
        return {"threads": threads}
    except Exception as e:
        logger.error(f"Error fetching threads: {e}")
        return {"threads": []}

@app.get("/api/chats/{thread_id}")
async def get_chat_history(thread_id: str):
    """Get the message history for a given thread."""
    graph = await get_graph()
    await _ensure_chat_titles_table()
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if not state.values:
        return {"messages": []}
    return {
        "messages": state.values.get("messages", []),
        "title": await _get_or_create_chat_title(thread_id),
    }


def _title_from_message(content: str) -> str:
    """Create a compact default title without sending chat content to an LLM."""
    normalized = re.sub(r"\s+", " ", (content or "").strip())
    if not normalized:
        return "New conversation"
    return normalized if len(normalized) <= 56 else normalized[:53].rstrip() + "..."


async def _ensure_chat_title(thread_id: str, content: str) -> str:
    from backend.agent.graph import _db_conn
    title = _title_from_message(content)
    if not _db_conn:
        return title
    await _ensure_chat_titles_table()
    await _db_conn.execute(
        """INSERT INTO chat_titles(thread_id, title, custom, updated_at)
           VALUES (?, ?, 0, ?)
           ON CONFLICT(thread_id) DO NOTHING""",
        (thread_id, title, datetime.now(timezone.utc).isoformat()),
    )
    await _db_conn.commit()
    return title


async def _get_or_create_chat_title(thread_id: str) -> str:
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return "New conversation"
    await _ensure_chat_titles_table()
    async with _db_conn.execute(
        "SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,)
    ) as cur:
        row = await cur.fetchone()
    if row:
        return row[0]
    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = (state.values or {}).get("messages", [])
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    return await _ensure_chat_title(thread_id, first_user)


@app.patch("/api/chats/{thread_id}")
async def rename_chat(thread_id: str, body: dict):
    """Rename a conversation explicitly chosen by the user."""
    from backend.agent.graph import _db_conn
    title = body.get("title") if isinstance(body, dict) else None
    if not isinstance(title, str) or not title.strip():
        raise HTTPException(status_code=400, detail="Chat title must be a non-empty string")
    title = re.sub(r"\s+", " ", title.strip())
    if len(title) > 100:
        raise HTTPException(status_code=400, detail="Chat title must be 100 characters or fewer")
    if not _db_conn:
        raise HTTPException(status_code=503, detail="Chat database is not initialized")
    await _ensure_chat_titles_table()
    await _db_conn.execute(
        """INSERT INTO chat_titles(thread_id, title, custom, updated_at)
           VALUES (?, ?, 1, ?)
           ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title,
             custom=1, updated_at=excluded.updated_at""",
        (thread_id, title, datetime.now(timezone.utc).isoformat()),
    )
    await _db_conn.commit()
    return {"ok": True, "id": thread_id, "title": title}

@app.delete("/api/chats/{thread_id}")
async def delete_chat(thread_id: str):
    """Delete a chat thread and all its checkpoints from the database."""
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return {"ok": False, "error": "Database not initialized"}
    await _ensure_chat_titles_table()
    
    try:
        await _db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        # AsyncSqliteSaver creates `writes` (not `checkpoint_writes`). Keep a
        # compatibility fallback for older databases that used the latter.
        async with _db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('writes', 'checkpoint_writes')"
        ) as cur:
            write_tables = [row[0] for row in await cur.fetchall()]
        for table in write_tables:
            await _db_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        await _db_conn.execute("DELETE FROM chat_titles WHERE thread_id = ?", (thread_id,))
        await _db_conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        return {"ok": False, "error": str(e)}

# ── State Lifecycle Helpers ──────────────────────────────────────────────────

def _start_of_turn_state(content: str, provider: str, model: str) -> dict:
    """Returns the clean state dict to inject at the start of a new turn."""
    return {
        "messages": [{"role": "user", "content": content}],
        "llm_provider": provider,
        "llm_model": model,
        "intent": None,
        "retrieved_docs": [],
        "planned_calls": [],
        "pending_confirmation": None,
        "confirmed": None,
        "cancel_requested": None,
        "execution_results": [],
        "final_response": "",
        "technical_details": None,
    }

async def _end_of_turn_cleanup(graph, config, result_state: dict):
    """
    Cleans up transient state (like confirmation flags) when a turn truly finishes.
    This prevents them from leaking into the next turn or being incorrectly
    echoed back to the frontend in the final response payload.
    Crucially, it also appends the assistant's final response back into the 
    `messages` array so it is preserved in the conversation history for resumes.
    """
    # If we are just halting to ask for confirmation, the turn isn't over.
    is_paused_for_confirmation = bool(result_state.get("pending_confirmation")) and result_state.get("confirmed") is None

    if not is_paused_for_confirmation:
        cleanup_state = {
            "pending_confirmation": None,
            "confirmed": None,
            "cancel_requested": None,
            "planned_calls": [],
            "execution_results": [],
        }

        # Resolve what the final response actually was (handling mid-turn cancellation)
        content = result_state.get("final_response", "")
        if result_state.get("cancel_requested") and not content:
            content = "Execution was cancelled by the user."

        # Append the AI's turn to the history
        if content:
            assistant_msg = {"role": "assistant", "content": content}
            details = result_state.get("technical_details")
            if details:
                assistant_msg["details"] = details
            cleanup_state["messages"] = [assistant_msg]

        # Save to LangGraph's checkpointer
        await graph.aupdate_state(config, cleanup_state)
        # Update the local dictionary so main.py doesn't echo transient state over WS
        result_state.update(cleanup_state)

# ── WebSocket chat ─────────────────────────────────────────────────────────────

# NOTE: _global_in_flight removed — it is now per-connection (local var in chat_socket)

@app.websocket("/ws/chat/{thread_id}")
async def chat_socket(ws: WebSocket, thread_id: str):
    # Per-connection in-flight flag — prevents data races between tabs
    in_flight = False
    await ws.accept()
    logger.info("WebSocket client connected", thread_id=thread_id)

    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    queue = asyncio.Queue()

    async def ws_reader():
        try:
            while True:
                raw = await ws.receive_text()
                await queue.put(raw)
        except WebSocketDisconnect:
            await queue.put(None)

    reader_task = asyncio.create_task(ws_reader())
    graph_task = None
    owns_thread_run = False
    queue_task = asyncio.create_task(queue.get())

    try:
        while True:
            waiters = [queue_task]
            if graph_task is not None:
                waiters.append(graph_task)
                
            done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            
            if queue_task in done:
                raw = queue_task.result()
                if raw is None:
                    break
                    
                queue_task = asyncio.create_task(queue.get())
                
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"type": "message", "content": raw}

                msg_type = payload.get("type", "message")

                if msg_type == "cancel":
                    if graph_task and not graph_task.done():
                        # Signal the graph to cancel
                        await graph.aupdate_state(config, {"cancel_requested": True})
                        logger.info("Cancel requested. Waiting for graph to cleanly halt.")
                    continue

                if msg_type == "confirm":
                    if in_flight:
                        await ws.send_json({"type": "response", "content": "Still finishing the previous action. Please wait.", "details": []})
                        continue
                        
                    confirmed = payload.get("confirmed", False)
                    if confirmed:
                        if not owns_thread_run:
                            async with _active_threads_guard:
                                if thread_id in _active_threads:
                                    await ws.send_json({"type": "response", "content": "This conversation is active in another connection.", "details": []})
                                    continue
                                _active_threads.add(thread_id)
                            owns_thread_run = True
                        await graph.aupdate_state(config, {"confirmed": True, "cancel_requested": None})
                        in_flight = True
                        graph_task = asyncio.create_task(graph.ainvoke(None, config))
                    else:
                        await graph.aupdate_state(config, {"confirmed": False, "pending_confirmation": None, "planned_calls": [], "cancel_requested": None})
                        await ws.send_json({"type": "response", "content": "Action cancelled.", "details": [], "needs_confirmation": None})
                    continue

                if msg_type == "message":
                    content = payload.get("content", "").strip()
                    if not content:
                        continue
                        
                    if in_flight:
                        await ws.send_json({"type": "response", "content": "Still finishing the previous action. Please wait.", "details": []})
                        continue

                    async with _active_threads_guard:
                        if thread_id in _active_threads:
                            await ws.send_json({"type": "response", "content": "Still finishing the previous action. Please wait.", "details": []})
                            continue
                        _active_threads.add(thread_id)
                    owns_thread_run = True
                        
                    provider = payload.get("provider", _current_provider)
                    model = payload.get("model", _current_model)

                    # Persist a readable title from the first user turn. The
                    # INSERT is idempotent, so a later message never replaces
                    # a user-defined title.
                    await _ensure_chat_title(thread_id, content)

                    await ws.send_json({"type": "typing", "content": ""})

                    input_state = _start_of_turn_state(content, provider, model)

                    in_flight = True
                    graph_task = asyncio.create_task(graph.ainvoke(input_state, config))

            if graph_task in done:
                # in_flight cleared in finally block below
                try:
                    result_state = graph_task.result()
                    
                    await _end_of_turn_cleanup(graph, config, result_state)
                    
                    content = result_state.get("final_response", "")
                    details = result_state.get("technical_details", [])
                    
                    if result_state.get("cancel_requested") and not content:
                        content = "Execution was cancelled by the user."
                        details = []
                        
                    response_payload = {
                        "type": "response",
                        "content": content,
                        "details": details,
                        "needs_confirmation": result_state.get("pending_confirmation"),
                    }
                    await ws.send_json(response_payload)
                except Exception as e:
                    logger.error(f"Error in graph execution: {e}")
                    await ws.send_json({"type": "response", "content": f"Error: {str(e)}"})
                finally:
                    in_flight = False
                    graph_task = None
                    if owns_thread_run:
                        async with _active_threads_guard:
                            _active_threads.discard(thread_id)
                        owns_thread_run = False

    except WebSocketDisconnect:
        reader_task.cancel()
        logger.info("WebSocket client disconnected", thread_id=thread_id)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
    finally:
        if graph_task is not None and not graph_task.done():
            graph_task.cancel()
        if owns_thread_run:
            async with _active_threads_guard:
                _active_threads.discard(thread_id)
        reader_task.cancel()


# ── Dev server entry ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level,
    )
