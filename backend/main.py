"""
FastAPI backend entrypoint.

Routes:
  GET  /                          → serves frontend/index.html
  GET  /health                    → Resolve connection status
  GET  /api/models                → available LLM models for UI dropdown
  POST /api/set-model             → switch active model without restart
  POST /api/simple/query          → one ephemeral popup assistant request
  POST /api/simple/confirm        → confirm/cancel a popup action
  GET/POST/DELETE /api/simple/memory → explicit global memory profile
  WS   /ws/chat                   → main conversation WebSocket
"""
from __future__ import annotations

import json
import asyncio
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from contextlib import asynccontextmanager, suppress
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.graph import get_graph
from backend.agent.state import AgentState
from backend.config import settings
from backend.llm.factory import get_model_catalog, get_provider
from backend.llm.errors import redact_secrets
from backend.logging_config import setup_logging, get_logger
from backend.agent.nodes.confirmation import invalidate_classification_cache
from backend.memory import (
    add_memory,
    clear_memory,
    delete_memory,
    extract_explicit_memory,
    load_memory,
    memory_context,
)


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

async def _warm_services() -> None:
    """Warm MCP, graph storage, and tool classification without blocking HTTP startup."""
    from backend.resolve.mcp_client import init_mcp_client

    try:
        # Start the persistent MCP subprocess, activate MVP domains, and cache tools.
        mcp_client = await init_mcp_client()

        # Chat titles are application metadata and intentionally live outside
        # LangGraph's checkpoint schema. get_graph() opens the SQLite connection.
        await get_graph()
        await _ensure_chat_titles_table()

        # Build/update classification after the active tool list is available.
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
        invalidate_classification_cache()
        logger.info("Background service warm-up complete", tools=len(tools))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # HTTP and model-selection routes remain useful when Resolve/MCP is
        # unavailable. The health route reports the disconnected state.
        logger.warning("Background service warm-up failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Let Uvicorn serve /health and /api/models immediately. MCP startup can
    # take time or fail independently when Resolve is closed.
    warmup_task = asyncio.create_task(_warm_services())
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    from backend.agent.graph import close_graph
    from backend.resolve.mcp_client import close_mcp_client
    if not warmup_task.done():
        warmup_task.cancel()
    with suppress(asyncio.CancelledError):
        await warmup_task
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

# The Resolve popup uses one hidden session id and never exposes the old
# conversation browser. These tasks exist only while a popup request is running.
_simple_tasks: dict[str, asyncio.Task] = {}
# A Resolve popup can deliver a duplicate click event while the first
# confirmation request is finishing. Keep the completed response briefly so
# the duplicate is answered idempotently instead of resuming a deleted graph
# checkpoint and asking for confirmation again.
_simple_completed: dict[str, tuple[float, dict]] = {}
_simple_confirmation_locks: dict[str, asyncio.Lock] = {}
_SIMPLE_RESULT_TTL_SECONDS = 60


def _cached_simple_result(session_id: str) -> dict | None:
    now = time.monotonic()
    expired = [
        key for key, (created_at, _) in _simple_completed.items()
        if now - created_at > _SIMPLE_RESULT_TTL_SECONDS
    ]
    for key in expired:
        _simple_completed.pop(key, None)
    cached = _simple_completed.get(session_id)
    return cached[1] if cached else None


def _cache_simple_result(session_id: str, payload: dict) -> None:
    _simple_completed[session_id] = (time.monotonic(), payload)

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
    from backend.resolve.mcp_client import check_resolve_connection
    connected = await check_resolve_connection()
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
    try:
        catalog = get_model_catalog(_current_provider)
    except Exception as exc:
        # This endpoint is used while the popup is starting. Keep it
        # available even if a future provider implementation fails during
        # discovery; the popup already has a small built-in fallback catalog.
        logger.warning("Model catalog lookup failed; returning empty optional lists", error=str(exc))
        catalog = {"anthropic": [], "gemini": [], "ollama": []}
    # The model catalog helper is also used outside the HTTP layer and falls
    # back to the startup configuration. Return the live selection here so
    # popup clients can initialise their controls correctly after a switch.
    catalog["current_provider"] = _current_provider
    catalog["current_model"] = _current_model
    return catalog


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


async def _delete_thread_state(thread_id: str) -> None:
    """Remove one LangGraph session from the legacy SQLite checkpointer."""
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return
    await _ensure_chat_titles_table()
    await _db_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    async with _db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('writes', 'checkpoint_writes')"
    ) as cur:
        write_tables = [row[0] for row in await cur.fetchall()]
    for table in write_tables:
        await _db_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
    await _db_conn.execute("DELETE FROM chat_titles WHERE thread_id = ?", (thread_id,))
    await _db_conn.commit()

@app.delete("/api/chats/{thread_id}")
async def delete_chat(thread_id: str):
    """Delete a chat thread and all its checkpoints from the database."""
    from backend.agent.graph import _db_conn
    if not _db_conn:
        return {"ok": False, "error": "Database not initialized"}
    try:
        await _delete_thread_state(thread_id)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        return {"ok": False, "error": str(e)}

# ── State Lifecycle Helpers ──────────────────────────────────────────────────

def _start_of_turn_state(
    content: str,
    provider: str,
    model: str,
    saved_memory: list[dict] | None = None,
) -> dict:
    """Returns the clean state dict to inject at the start of a new turn."""
    return {
        "messages": [{"role": "user", "content": content}],
        "llm_provider": provider,
        "llm_model": model,
        "global_memory": saved_memory or [],
        "intent": None,
        "retrieved_docs": [],
        "planned_calls": [],
        "pending_confirmation": None,
        "confirmed": None,
        "cancel_requested": None,
        "execution_results": [],
        "final_response": "",
        "technical_details": None,
        "tool_proposal": None,
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


# ── Small popup API ───────────────────────────────────────────────────────────

async def _run_simple_graph(session_id: str, input_state: dict | None) -> dict:
    """Run one popup turn and keep its task cancellable by the Stop button."""
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    task = asyncio.create_task(graph.ainvoke(input_state, config))
    _simple_tasks[session_id] = task
    try:
        return await task
    finally:
        _simple_tasks.pop(session_id, None)


async def _simple_payload(session_id: str, result_state: dict) -> dict:
    """Convert a graph result to the intentionally small popup response."""
    # Check for pending confirmation BEFORE cleanup, because cleanup will
    # wipe pending_confirmation from the graph state.
    pending = result_state.get("pending_confirmation")
    # A confirmed graph still carries the original pending payload until the
    # end-of-turn cleanup runs. Only pause the popup when confirmation has not
    # been supplied yet; after confirmed=True, return the executor/reporter
    # result instead of showing the same prompt again.
    if pending and result_state.get("confirmed") is None:
        return {
            "ok": True,
            "content": pending.get("message", "Please confirm this action."),
            "details": result_state.get("technical_details", []),
            "needs_confirmation": pending,
        }

    # No confirmation needed — run cleanup and return the final response.
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    await _end_of_turn_cleanup(graph, config, result_state)

    payload = {
        "ok": True,
        "content": result_state.get("final_response", "No response was returned."),
        "details": result_state.get("technical_details", []),
        "needs_confirmation": None,
    }
    # Popup sessions are deliberately ephemeral. The browser's legacy chat
    # endpoints remain available as a fallback, but this path leaves no chat
    # transcript in SQLite after a completed turn.
    await _delete_thread_state(session_id)
    return payload


@app.post("/api/simple/query")
async def simple_query(body: dict):
    """Run one prompt for the small Resolve popup client."""
    content = body.get("content", "") if isinstance(body, dict) else ""
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="Query content is required")

    session_id = body.get("session_id") or str(uuid.uuid4())
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must be a string")
    if session_id in _simple_tasks:
        raise HTTPException(status_code=409, detail="A request is already running")
    # This is a new user turn. A previous completed confirmation response is
    # no longer relevant once the popup submits new content.
    _simple_completed.pop(session_id, None)

    remembered = extract_explicit_memory(content)
    if remembered:
        add_memory(remembered)
        # Memory commands are deterministic and do not need an LLM or a
        # Resolve call. This keeps the popup path fast and clear.
        return {
            "ok": True,
            "content": "I will remember that: " + remembered,
            "details": [],
            "needs_confirmation": None,
        }

    provider = body.get("provider", _current_provider)
    model = body.get("model", _current_model)
    input_state = _start_of_turn_state(
        content.strip(),
        provider,
        model,
        saved_memory=memory_context(),
    )
    try:
        result = await _run_simple_graph(session_id, input_state)
        return await _simple_payload(session_id, result)
    except Exception as exc:
        safe_error = redact_secrets(exc)
        logger.error("Simple popup query failed", error=safe_error)
        await _delete_thread_state(session_id)
        return {"ok": False, "content": f"Assistant error: {safe_error}", "details": []}


@app.post("/api/simple/confirm")
async def simple_confirm(body: dict):
    """Continue or cancel the confirmation currently shown in the popup."""
    session_id = body.get("session_id") if isinstance(body, dict) else None
    confirmed = bool(body.get("confirmed")) if isinstance(body, dict) else False
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    # Serialize confirmation requests for this popup session. This protects
    # against double-clicks and duplicate native UI events.
    lock = _simple_confirmation_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        cached = _cached_simple_result(session_id)
        if cached is not None:
            return cached

        graph = await get_graph()
        config = {"configurable": {"thread_id": session_id}}
        if not confirmed:
            await graph.aupdate_state(
                config,
                {"confirmed": False, "pending_confirmation": None, "planned_calls": [], "cancel_requested": None},
            )
            await _delete_thread_state(session_id)
            payload = {"ok": True, "content": "Action cancelled.", "details": [], "needs_confirmation": None}
            _cache_simple_result(session_id, payload)
            return payload

        if session_id in _simple_tasks:
            raise HTTPException(status_code=409, detail="A request is already running")
        await graph.aupdate_state(config, {"confirmed": True, "cancel_requested": None})
        try:
            result = await _run_simple_graph(session_id, None)
            payload = await _simple_payload(session_id, result)
            # Cache only completed responses. If a malformed/stale checkpoint
            # still asks for confirmation, the caller must not be locked out.
            if payload.get("needs_confirmation") is None:
                _cache_simple_result(session_id, payload)
            return payload
        except Exception as exc:
            safe_error = redact_secrets(exc)
            logger.error("Simple popup confirmation failed", error=safe_error)
            await _delete_thread_state(session_id)
            return {"ok": False, "content": f"Assistant error: {safe_error}", "details": []}


@app.post("/api/simple/cancel")
async def simple_cancel(body: dict):
    """Ask a running popup request to stop at the next safe graph boundary."""
    session_id = body.get("session_id") if isinstance(body, dict) else None
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    if session_id in _simple_tasks:
        await graph.aupdate_state(config, {"cancel_requested": True})
        return {"ok": True, "content": "Stop requested."}

    await _delete_thread_state(session_id)
    return {"ok": True, "content": "Session cleared."}


@app.get("/api/simple/memory")
async def simple_memory():
    """Return the explicit global memory profile for the popup menu."""
    return load_memory()


@app.post("/api/simple/memory")
async def add_simple_memory(body: dict):
    text = body.get("text") if isinstance(body, dict) else None
    category = body.get("category", "preference") if isinstance(body, dict) else "preference"
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Memory text is required")
    try:
        return {"ok": True, "item": add_memory(text, category)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/simple/memory")
async def delete_simple_memory(memory_id: str | None = None):
    if memory_id:
        return {"ok": delete_memory(memory_id)}
    clear_memory()
    return {"ok": True}

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

                    remembered = extract_explicit_memory(content)
                    if remembered:
                        add_memory(remembered)

                    await ws.send_json({"type": "typing", "content": ""})

                    input_state = _start_of_turn_state(
                        content,
                        provider,
                        model,
                        saved_memory=memory_context(),
                    )

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
                    safe_error = redact_secrets(e)
                    logger.error("Error in graph execution", error=safe_error)
                    await ws.send_json({"type": "response", "content": f"Error: {safe_error}"})
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
        logger.error("Unexpected WebSocket error", error=redact_secrets(e))
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
