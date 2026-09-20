"""
MCP client for DaVinci Resolve.

Uses the raw MCP SDK to maintain a SINGLE persistent stdio session for
the lifetime of the server. This is required because:
  - langchain-mcp-adapters spawns a NEW subprocess per get_tools()/ainvoke() call
  - Domain activation state lives inside the MCP server subprocess
  - A new subprocess has NO activated domains → planner sees no domain tools

By keeping one persistent subprocess, domain activations done at startup
remain effective for all subsequent tool calls in the same session.
"""
from __future__ import annotations

import asyncio
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from langchain_core.tools import StructuredTool
from backend.logging_config import get_logger
from backend.config import settings

logger = get_logger(__name__)

# ── Persistent session state ──────────────────────────────────────────────────

_session: ClientSession | None = None
_tools_cache: list | None = None
_task: asyncio.Task | None = None
_shutdown_event: asyncio.Event | None = None

MVP_DOMAINS = [
    "project_management",
    "media_pool",
    "timeline_operations",
    "render_delivery",
    "clip_properties",
    "timeline_item_editing",
]

SERVER_CONFIG = StdioServerParameters(
    command=settings.mcp_python,
    args=[settings.mcp_server_path],
)


# ── Helper: convert raw mcp.types.Tool → LangChain StructuredTool ────────────

class _ToolResult:
    """Lightweight wrapper returned by every tool so executor can detect errors."""
    __slots__ = ("text", "is_error")
    def __init__(self, text: str, is_error: bool):
        self.text = text
        self.is_error = is_error


def _make_langchain_tool(tool_name: str, description: str, schema: dict) -> StructuredTool:
    """
    Build a LangChain StructuredTool that calls the live _session at invocation
    time (not at creation time). This means reconnects work without re-building
    the tool list, and there is no stale-session closure problem.
    """
    async def _arun(**kwargs: Any) -> _ToolResult:
        if _session is None:
            return _ToolResult("DaVinci Resolve is not connected.", is_error=True)
        try:
            result = await _session.call_tool(tool_name, kwargs)
            text_parts = [getattr(item, "text", str(item)) for item in (result.content or [])]
            text = "\n".join(text_parts)
            textual_error = text.startswith((
                "DaVinci Resolve error:",
                "Unexpected error:",
                "MCP call failed:",
                "Error:",
            ))
            return _ToolResult(text, is_error=bool(result.isError) or textual_error)
        except Exception as exc:
            return _ToolResult(f"MCP call failed: {exc}", is_error=True)

    st = StructuredTool.from_function(
        coroutine=_arun,
        name=f"davinci-resolve_{tool_name}",
        description=description or "",
        args_schema=None,
    )
    # Attach raw inputSchema so executor/planner schema extraction works
    object.__setattr__(st, "inputSchema", schema)
    return st


async def _refresh_tools() -> None:
    """Ask the server for its current tool list and rebuild the LangChain wrappers."""
    global _tools_cache
    assert _session is not None
    response = await _session.list_tools()
    _tools_cache = [
        _make_langchain_tool(t.name, t.description or "", t.inputSchema or {})
        for t in response.tools
    ]
    logger.info("Tool cache refreshed", count=len(_tools_cache))


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def init_mcp_client() -> "MCPClientFacade":
    """
    Start the persistent stdio subprocess, activate MVP domains, cache tools.
    Call once at startup; subsequent calls are no-ops.
    If DaVinci Resolve is not running the client starts in disconnected mode
    (get_tools returns [] and the health endpoint reports resolve_connected=False).
    """
    global _session, _task, _shutdown_event

    if _facade._initialized and _session is not None:
        return _facade
    if _facade._initialized and _session is None:
        _facade._initialized = False

    logger.info("Starting persistent MCP subprocess…")

    started = asyncio.Event()
    _shutdown_event = asyncio.Event()

    async def _keep_alive():
        global _session, _tools_cache
        try:
            async with stdio_client(SERVER_CONFIG) as (read, write):
                async with ClientSession(read, write) as session:
                    _session = session
                    await session.initialize()
                    started.set()
                    await _shutdown_event.wait()
        except Exception as exc:
            logger.warning("MCP subprocess exited unexpectedly", error=str(exc))
            started.set()   # unblock the waiter even on failure
        finally:
            _session = None
            _tools_cache = None

    _task = asyncio.create_task(_keep_alive())

    try:
        await asyncio.wait_for(started.wait(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("MCP server did not start within 30s — running in disconnected mode")
        _facade._initialized = True
        return _facade

    if _session is None:
        logger.warning("MCP session not established — DaVinci Resolve may not be running")
        _facade._initialized = True
        return _facade

    # Activate all MVP domains and cache tools — wrapped so any failure
    # (ClosedResourceError, TimeoutError, etc.) falls through to disconnected mode
    try:
        for domain in MVP_DOMAINS:
            try:
                await _session.call_tool("activate_domain", {"domain": domain})
                logger.info("Domain activated", domain=domain)
            except Exception as exc:
                logger.warning("Domain activation failed", domain=domain, error=str(exc))

        # Cache tools AFTER activation — domain-specific tools now visible
        await _refresh_tools()
        _facade._initialized = True
        logger.info("MCP client ready", tools=len(_tools_cache or []))
    except Exception as exc:
        logger.warning(
            "MCP session lost during startup — running in disconnected mode",
            error=str(exc),
        )
        _facade._initialized = True
        _session = None  # mark as disconnected so is_resolve_connected() returns False


    return _facade



async def close_mcp_client() -> None:
    """Cleanly shut down the persistent subprocess."""
    global _session, _tools_cache
    if _shutdown_event is not None:
        _shutdown_event.set()
    if _task is not None:
        try:
            await asyncio.wait_for(_task, timeout=5)
        except (asyncio.TimeoutError, Exception):
            _task.cancel()
    _session = None
    _tools_cache = None
    _facade._initialized = False
    logger.info("MCP client shut down")


# ── Public facade ─────────────────────────────────────────────────────────────

class MCPClientFacade:
    """
    Drop-in replacement for MultiServerMCPClient API (get_tools + ainvoke),
    but routes all calls through the single persistent session.
    """
    def __init__(self):
        self._initialized: bool = False  # instance-level, not class-level

    @property
    def is_connected(self) -> bool:
        return _session is not None

    async def get_tools(self) -> list:
        """Return cached tool list. Returns [] in disconnected mode."""
        if _tools_cache is None:
            return []
        return list(_tools_cache)


_facade = MCPClientFacade()


def get_mcp_client() -> MCPClientFacade:
    """Return the facade. Works in both connected and disconnected mode."""
    return _facade


def is_resolve_connected() -> bool:
    """True if the MCP stdio session exists.

    This is intentionally a cheap local check.  The stdio process can remain
    alive after Resolve closes, so request paths should use
    :func:`check_resolve_connection` when they need the real bridge status.
    """
    return _session is not None


async def check_resolve_connection() -> bool:
    """Probe the Resolve bridge through a harmless MCP read operation.

    Keeping the MCP session alive is not proof that DaVinci Resolve is still
    running.  The server reports the actual bridge error when Resolve is
    closed, so use ``get_current_project`` as a small, non-mutating probe.
    This also lets the same persistent MCP process recover when Resolve is
    reopened later.
    """
    if _session is None:
        return False

    try:
        result = await asyncio.wait_for(
            _session.call_tool("get_current_project", {}),
            timeout=3,
        )
        text = "\n".join(
            getattr(item, "text", str(item))
            for item in (getattr(result, "content", None) or [])
        )
        if getattr(result, "isError", False):
            return False

        lowered = text.lower()
        return not any(
            marker in lowered
            for marker in (
                "davinci resolve error:",
                "da vinci resolve error:",
                "mcp call failed:",
                "unexpected error:",
                "is not running",
                "not connected",
            )
        )
    except Exception as exc:
        logger.warning("Resolve connectivity probe failed", error=str(exc))
        return False
