# Resolve Assistant — Complete Implementation History

This document records the architecture, features, fixes, removals, testing, and
operational decisions made in the Resolve Assistant project. It is intended as
handoff context for teammates and future agents.

## 1. Project direction

The project started as a browser-based DaVinci Resolve assistant with a full
chat interface and conversation history. It was simplified into a college-level
Resolve-native popup with a small FastAPI sidecar:

```text
DaVinci Resolve
    └── ResolveAssistant.py popup
            └── local HTTP requests
                    └── FastAPI backend
                            ├── LangGraph agent
                            ├── MCP Resolve client
                            ├── local documentation/RAG
                            ├── LLM providers
                            └── optional web fallback
```

The popup is deliberately lightweight. Resolve's embedded Python owns the UI
and sends HTTP requests; the backend owns provider keys, RAG, planning,
confirmation, MCP, and execution.

## 2. Resolve-native popup UI

### Added

- `resolve_scripts/ResolveAssistant.py`
- A compact Resolve/Fusion UIManager window.
- Explicit Close button and native window close handling.
- Provider dropdown for Anthropic, Gemini, and Ollama.
- Model dropdown populated from the backend catalog.
- Apply button for live provider/model changes.
- Memory button for viewing saved preferences.
- Clear button for clearing the visible popup conversation.
- One Send button that changes to Stop while a request is running.
- Confirm and Cancel buttons shown only when a destructive action needs approval.
- Status text for:
  - backend startup;
  - backend ready;
  - Resolve connected/offline;
  - working;
  - confirmation required;
  - errors.
- Popup-only ephemeral sessions instead of displaying old chat threads.
- Standard-library HTTP client so Resolve does not need the project virtualenv.
- Automatic backend startup when the configured local backend is unavailable.
- Shared-backend behavior so closing one popup does not terminate a backend used
  by another popup.

### UI responsiveness fix

The popup originally called `/health` synchronously from the UI timer every
100 ms. This could block Resolve's UI thread and make buttons appear frozen.
The health request now runs in a worker thread and returns to the UI through a
thread-safe event queue. Network operations are no longer performed directly
from the UI timer.

### Resolve installation synchronization

The source script is:

`resolve_scripts/ResolveAssistant.py`

The installed Resolve copy is kept synchronized at:

`/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/ResolveAssistant.py`

After UI changes, the installed file is copied from the source and verified
with `cmp`.

## 3. Backend lifecycle and startup

### FastAPI lifecycle changes

`backend/main.py` now starts MCP and graph/database warm-up in a background
startup task. This allows `/health` and `/api/models` to become available while
MCP is still warming instead of making the popup wait for the entire startup
sequence.

The backend also shuts down the graph and MCP client cleanly during application
shutdown.

### Popup endpoints

Added or expanded endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Reports backend and real Resolve bridge status |
| `GET /api/models` | Returns all provider/model choices and current selection |
| `POST /api/set-model` | Changes the active backend provider/model |
| `POST /api/simple/query` | Runs one popup request |
| `POST /api/simple/confirm` | Confirms or cancels a pending write action |
| `POST /api/simple/cancel` | Requests cancellation of a running request |
| `GET /api/simple/memory` | Reads explicit global memory |
| `POST /api/simple/memory` | Adds explicit global memory |
| `DELETE /api/simple/memory` | Deletes one or all memory items |

## 4. Agent architecture

The LangGraph flow is:

```text
router
  ├── qa     → rag → end
  └── action → planner → confirmation → executor → reporter → end
```

### Router

- Added deterministic routing patterns for common Resolve live-state queries
  and commands.
- Current project, timeline, page, track, clip, marker, and settings questions
  are routed to MCP actions instead of documentation Q&A.
- Conceptual and how-to questions are routed to RAG.
- Ambiguous messages use the selected provider as a fallback classifier.
- If the provider fails during ambiguous routing, the safer action route is
  selected so the planner can report the actual capability/connection state.

### Planner

- Uses active MCP tool schemas rather than a static tool list.
- Retrieves relevant tool documentation before planning.
- Validates proposed tool names and required arguments.
- Enforces allowed MCP domains.
- Applies defaults for marker fields where safe.
- Repairs missing timeline names by inserting `get_current_timeline`.
- Enforces one-based track and item indexes in the planner prompt and tests.
- Performs a real Resolve bridge check before sending a planning request to the
  LLM. This prevents unnecessary provider calls when Resolve is offline.
- Returns a clear message when Resolve is unavailable.
- Returns a safe message when the provider returns invalid JSON or fails.

### Confirmation gate

- Destructive tools are classified through
  `data/mcp_tool_classification.yaml`.
- Destructive actions pause and display a confirmation request.
- Confirmation text includes the action target where available.
- Confirmation generation has a deterministic fallback if the LLM is offline.
- Duplicate confirmation events are protected with per-session locks and a
  short-lived completed-result cache.
- Confirmed turns no longer return the same stale confirmation prompt.

### Executor

- Executes planned calls in order.
- Stops the plan after a failed tool call.
- Supports result chaining between calls.
- Supports current timeline aliases including:
  - `__prev_result__`;
  - `$prev_result`;
  - `$current_timeline`;
  - `$get_current_timeline`;
  - `__get_current_timeline__`.

The last group was added after live GUI testing found that the LLM generated
`$get_current_timeline`, which was previously sent literally to MCP and caused
errors such as `Timeline '$get_current_timeline' not found`.

### Reporter

- Produces conversational summaries after tool execution.
- Preserves deterministic failure reporting when the LLM omits a failure.
- Reports partial execution and skipped steps.
- Falls back to a deterministic summary when the provider is unavailable.

## 5. DaVinci Resolve MCP integration

### Persistent MCP session

`backend/resolve/mcp_client.py` maintains one persistent MCP stdio session.
This preserves activated domain state and avoids spawning a new MCP process for
each tool request.

The backend activates these domains at startup:

- Project Management
- Media Pool
- Timeline Operations
- Render Delivery
- Clip Properties
- Timeline Item Editing

The startup tool cache currently contains the active MCP tools from these
domains.

### Real Resolve connectivity probe

The old connection check only tested whether the MCP stdio session existed.
That was insufficient because the MCP process can remain alive after Resolve
closes.

The new `check_resolve_connection()` function performs a harmless
`get_current_project` MCP read with a timeout. It recognizes Resolve bridge
errors and returns the real connection state. Both `/health` and the planner
use this check.

### Latest MCP reliability fixes — 2026-09-20

Two issues found during live popup testing were corrected:

- The MCP server previously refused to connect when macOS process inspection
  could not see DaVinci Resolve. On some systems `pgrep`/system process
  services fail even while Resolve is open and its scripting API is available.
  The client now treats process inspection as a warning and lets the actual
  `DaVinciResolveScript.scriptapp("Resolve")` connection determine whether
  Resolve is available.
- The planner/executor now normalizes timeline names case-insensitively before
  calling timeline tools. For example, a user request containing `test run`
  resolves to the actual timeline `Test Run`. Timeline creation and timeline
  listing are excluded from this normalization because they intentionally use
  the supplied name.

This removes two common causes of false MCP failures: a valid Resolve session
being rejected before the scripting API is tried, and LLM-generated casing
differences causing `Timeline not found` errors.

## 6. LLM providers and error handling

### Providers

The provider factory supports:

- Anthropic;
- Gemini;
- Ollama.

The model catalog is safe to load even when an optional provider is offline.
Ollama model discovery failures no longer make `/api/models` fail.

### Gemini reliability

Gemini requests now:

- use bounded request timeouts;
- retry transient HTTP, SSL, connection, and service-unavailable errors;
- stop after a bounded number of attempts;
- return a friendly provider-unavailable message.

### Safe errors

Added `backend/llm/errors.py` for shared error handling:

- API keys and tokens are redacted from logs and user-facing errors;
- transient provider errors are classified consistently;
- authentication/configuration errors have actionable messages;
- provider outages do not expose raw provider URLs or credentials.

These protections are used by the Gemini, Anthropic, Ollama, planner, router,
RAG, reporter, confirmation, popup, WebSocket, and model-catalog paths.

## 7. RAG and documentation behavior

### Local documentation first

The RAG node searches the local Chroma index first. It uses the retrieved
excerpts as the primary context for the selected LLM.

The local index is built from Resolve documentation and stores source metadata,
chunk indexes, and vector distances.

### Documentation-first web fallback

Added `backend/web_search.py` and optional web fallback behavior:

1. Search local documentation.
2. Ask the selected model for a grounded local answer.
3. Detect answers that explicitly say the documentation does not cover the
   question.
4. Search the web only in that case.
5. Prefer Blackmagic Design results by issuing an official-domain search first.
6. Ask the selected model to answer only from returned snippets.
7. Include exact source URLs in the answer.
8. If the web search or final synthesis fails, show the available links or a
   clear failure message.

The web fallback is disabled with:

```env
WEB_SEARCH_ENABLED=false
```

Additional settings:

```env
WEB_SEARCH_TIMEOUT=8
WEB_SEARCH_MAX_RESULTS=5
```

The search endpoint is intentionally simple and keyless for this college-level
project. Search snippets are treated as untrusted evidence; the model is told
not to invent information beyond them.

## 8. Global memory

Persistent conversation-thread browsing was removed from the popup workflow.
Instead, the assistant supports explicit global memory:

- `remember that I prefer concise answers`;
- `remember that I use frame numbers`;
- explicit memory items are stored in `data/assistant_memory.json`;
- memory is injected into RAG, planning, and future requests;
- memory can be viewed, added, or cleared from the popup API.

Only explicit memory requests are stored. Normal conversations are not silently
converted into permanent memory.

## 9. Removed or simplified features

- Removed the old popup chat-thread browser from the primary UI.
- Removed the need for persistent popup conversation history.
- Removed the redundant separate Stop button; Send changes to Stop while busy.
- Removed reliance on the old activity-log/sidebar model for the popup workflow.
- Kept the browser UI and legacy WebSocket chat as a development fallback.
- Kept complex provider/MCP/RAG logic in the backend instead of duplicating it
  in Resolve's embedded Python.

## 10. Testing performed

### Automated tests

The current test suite result is:

```text
45 passed, 6 skipped, 3 warnings
```

Important regression coverage includes:

- provider error classification and API-key redaction;
- Resolve bridge connectivity probing;
- executor timeline-result aliases;
- planner schema injection and argument validation;
- confirmation behavior and duplicate-confirmation protection;
- popup state reset and stale-confirmation prevention;
- memory round trips;
- model catalog fallback behavior;
- web-search fallback ordering and source injection;
- cancellation and concurrency behavior;
- popup Markdown rendering and HTML escaping;
- case-insensitive timeline-name normalization;
- live-tool integration tests where Resolve is available.

### GUI testing performed in DaVinci Resolve

The Resolve-native popup was tested with these interactions:

| Test | Result |
|---|---|
| Fusion clip vs. compound clip | Passed |
| Current project name | Passed after backend restart |
| Current Resolve page | Passed; returned Edit |
| Video/audio track count | Initially exposed timeline alias bug; fixed and passed |
| Create timeline confirmation | Passed |
| Cancel create timeline | Passed |
| List timelines after cancellation | Passed; temporary timeline absent |
| Provider/model controls visible | Passed |
| Popup startup status | Passed after UI-thread health fix |

The latest live GUI smoke test also verified:

| Test | Result |
|---|---|
| Assistant response containing bullets, links, bold text, and inline code | Passed; rendered as formatted popup text |
| Timeline query using lowercase `test run` | Passed; resolved to `Test Run` |
| Switch to Color page | Passed |
| Switch back to Edit page | Passed |

### Popup Markdown rendering

The Resolve popup now renders the common Markdown emitted by providers instead
of displaying raw `**bold**`, backticks, or bullet markers. The lightweight
renderer supports headings, bold, italics, inline code, links, unordered lists,
and ordered lists. It escapes raw HTML before formatting, so model output cannot
inject arbitrary markup into the UI. This is intentionally a small renderer,
not a full CommonMark implementation, which keeps the embedded Resolve script
simple and dependency-free.

## 11. Known limitations

- Web search depends on network access and the search endpoint remaining
  available.
- The keyless HTML search fallback is intentionally lightweight and may need a
  provider API for production reliability.
- RAG answers still depend on the quality and coverage of the indexed Resolve
  documentation.
- LLM calls are not streamed, so responses may take several seconds.
- Some MCP plans use redundant intermediate reads, which is functional but can
  be optimized later.
- The current MCP activation set is still an MVP subset; Color Grading,
  Fairlight, Fusion-specific, and AI Studio domains are not activated by
  default.
- Ollama tool behavior varies by model.
- Resolve integration still requires Resolve to be running with external
  scripting enabled.
- Color grading, Fairlight, Fusion-specific, and AI Studio domains remain
  outside the currently activated MVP domain set.

## 12. Operational checklist

1. Keep the backend `.env` populated with the selected provider credentials.
2. Ensure Resolve external scripting is enabled.
3. Keep `resolve_scripts/ResolveAssistant.py` synchronized with the installed
   Utility-folder copy.
4. Restart the backend or reopen the popup after backend code changes.
5. Re-run `python -m backend.rag.ingest` when documentation changes.
6. Run `.venv/bin/python -m pytest -q` before handing changes to teammates.
7. If the popup reports startup failure, inspect `data/popup_backend.log`.
8. If code was changed while an older backend is running, stop that backend or
   reopen the popup so the new Python modules are loaded.
9. Confirm the MCP server checkout used by the backend contains the updated
   `resolve_client.py`; this project depends on the sibling
   `mcp-candidates/hoyt2` checkout for the Resolve MCP bridge.

## 13. College-project simplification pass — 2026-09-20

The project was reduced to the smallest set of components needed for the
current demonstration:

- Removed unused PostgreSQL configuration, Docker compose setup, and
  PostgreSQL checkpoint dependencies. The project uses one local SQLite
  checkpoint database instead.
- Removed unused direct dependencies for the old LangChain provider stack,
  PDF library, async file helper, and provider discovery package. The active
  code uses `pypdf`, direct provider clients, LangGraph, and Chroma.
- Removed dead imports and an old duplicate module-level declaration in the
  FastAPI entrypoint.
- Removed generated Python/test caches and stale runtime logs from the working
  folder; these are recreated automatically when the app or tests run.
- Added the missing `integration` pytest marker declaration so test output no
  longer reports a project-owned marker warning.
- Kept the browser frontend and WebSocket routes because they are still a
  documented development fallback, while the Resolve popup remains the main
  user interface.
- Kept the Chroma index, Resolve scripting documentation, SQLite checkpoint
  store, MCP classification file, provider modules, global memory, and tests
  because each is used by the current application or its documented workflow.

This is a cleanup rather than a feature rewrite: the popup, model switching,
MCP actions, confirmation gate, documentation-first RAG, web fallback, memory,
and browser fallback remain available. The goal is that the project can be
explained as a small pipeline: popup/browser → FastAPI → agent → RAG or MCP →
response.

## 14. Resolve popup input and transcript behavior — 2026-09-22

- Replaced the multiline prompt control with a single-line Resolve `LineEdit`.
- Enter now uses the native `ReturnPressed` event to submit the prompt, while
  the Send button remains available.
- Added a transcript anchor and automatic scroll-to-latest behavior after each
  user or assistant message. A cursor-visibility fallback is used when a
  Resolve build does not support scrolling to the anchor.
- Synchronized the updated source to Resolve's Utility scripts folder.

## 15. Important security note

An API key was previously visible in a pasted provider error URL during testing.
That key should be revoked and regenerated. Provider keys belong only in the
backend `.env` file and must never be committed or displayed in logs.

---

Last updated: 2026-09-20
