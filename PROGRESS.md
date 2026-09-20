# Resolve AI Assistant — Progress & Context Document

> **Purpose:** This document tracks all implementation progress, design decisions, features added, issues discovered, and next steps. It is the primary context source for continuing development across sessions.

For the complete consolidated history, including the Resolve popup migration,
reliability fixes, global memory, web fallback, GUI testing, and current
limitations, see [`docs/IMPLEMENTATION_HISTORY.md`](docs/IMPLEMENTATION_HISTORY.md).

---

## Project Summary

A standalone macOS companion app for DaVinci Resolve that provides:
1. RAG-grounded Q&A over official Resolve documentation (PDFs → Chroma vector store)
2. 15 agentic actions (editing/timeline, media/project, render/export) via the Resolve Scripting API
3. LangGraph agent with router → RAG/planner → confirmation gate → executor → reporter
4. Flexible LLM backend (Anthropic Claude or Ollama) switchable live from the UI

**Stack:** Python 3.11, FastAPI, LangGraph, Chroma, Anthropic/Gemini/Ollama clients, HTML/CSS/JS

## Simple Resolve popup mode — 2026-09-07

The project now includes a deliberately small Resolve/Fusion Python client for
the college-project version of the assistant:

- `resolve_scripts/ResolveAssistant.py` creates a compact UI Manager popup.
- The popup has an explicit Close control, provider/model selectors, and an
  Apply action that updates both the popup request and the backend default.
- Startup checks `/health` and enables the popup as soon as FastAPI responds;
  Resolve/MCP connectivity is reported separately while background warm-up
  continues.
- The popup uses `/api/simple/query`, `/api/simple/confirm`, and
  `/api/simple/cancel` instead of exposing chat threads.
- Popup sessions are ephemeral; completed turns are removed from the LangGraph
  SQLite checkpoint tables.
- `backend/memory.py` stores only explicit `remember ...` items in
  `data/assistant_memory.json`.
- The browser client remains available as a development fallback.

The popup intentionally keeps the heavy agent, MCP client, RAG, provider keys,
and tool safety logic in the FastAPI sidecar. Resolve's embedded Python only
needs its built-in UI Manager and the standard-library HTTP client.

### Reliability and rendering pass — 2026-09-20

- The popup renders common Markdown safely, including lists, links, headings,
  bold, italics, and inline code. Raw HTML is escaped before rendering.
- Timeline tool arguments are matched to real Resolve timeline names without
  case sensitivity, preventing avoidable `Timeline not found` failures.
- The MCP client no longer treats macOS process-list inspection as a hard
  prerequisite. The Resolve scripting API is now the source of truth for
  connection status.
- Live smoke tests covered formatted responses, a lowercase timeline query,
  switching to Color, and switching back to Edit.
- Latest automated result: **45 passed, 6 skipped, 3 warnings**.
- The installed Resolve script was synchronized from
  `resolve_scripts/ResolveAssistant.py` to the Utility scripts folder.

---

## Implementation Log

### Session 1 — 2026-08-17

**Status:** Full Phase 1 MVP implemented ✅

#### Completed

| Component | File(s) | Notes |
|---|---|---|
| Project scaffold | `resolve-assistant/` directory tree | All directories created |
| Requirements | `requirements.txt` | Only packages used by the backend, popup, RAG index, and tests |
| Configuration | `config.yaml`, `backend/config.py`, `.env.example` | Hierarchical: YAML → .env override |
| Logging | `backend/logging_config.py` | Dual output: console (human) + JSON file (machine) |
| LLM Provider — Base | `backend/llm/base.py` | `LLMProvider` ABC with `generate()`, `set_model()`, `current_model` |
| LLM Provider — Anthropic | `backend/llm/anthropic_provider.py` | All Claude models; live model switching; normalized output |
| LLM Provider — Ollama | `backend/llm/ollama_provider.py` | Fetches model list from running Ollama; tool-call normalization |
| LLM Factory | `backend/llm/factory.py` | Only file that imports concrete providers; `get_model_catalog()` for UI |
| RAG Embeddings | `backend/rag/embeddings.py` | Chroma's built-in ONNX embedding function — always local |
| RAG Ingest | `backend/rag/ingest.py` | PDF/text → section-based chunking → Chroma; CLI entry point |
| RAG Retriever | `backend/rag/retriever.py` | Top-k query with source metadata |
| Resolve Connection | `backend/resolve/connection.py` | Auto-launches Resolve; guarded `current_project()` / `current_timeline()` |
| Tool Functions | `backend/resolve/tools.py` | All 15 tools fully implemented |
| Tool Registry | `backend/resolve/tool_registry.py` | Safe/destructive tags; auto-generated LLM schemas from docstrings |
| Agent State | `backend/agent/state.py` | `AgentState` TypedDict |
| Router Node | `backend/agent/nodes/router.py` | Classifies `qa` vs `action` |
| RAG Node | `backend/agent/nodes/rag.py` | Retrieves + generates grounded answer |
| Planner Node | `backend/agent/nodes/planner.py` | RAG-grounded, validates against registry |
| Confirmation Node | `backend/agent/nodes/confirmation.py` | Pauses on destructive tools |
| Executor Node | `backend/agent/nodes/executor.py` | Runs calls, stops early on failure |
| Reporter Node | `backend/agent/nodes/reporter.py` | Plain-language summary with partial failures |
| Agent Graph | `backend/agent/graph.py` | Full LangGraph StateGraph wiring |
| FastAPI Main | `backend/main.py` | WS chat, `/health`, `/api/models`, `/api/set-model` |
| Frontend HTML | `frontend/index.html` | Model selector, status panel, confirmation dialog, action log |
| Frontend CSS | `frontend/style.css` | Premium dark mode, glassmorphism, Inter font, animated typing dots |
| Frontend JS | `frontend/chat.js` | WS client, markdown rendering, live model switching, confirm/cancel |
| Unit Tests | `tests/unit/test_tools_mocked.py` | 6 tool classes, 18+ test cases |
| Confirmation Tests | `tests/unit/test_confirmation_gate.py` | Registry tag + gate behavior regression |
| Integration Tests | `tests/integration/test_tools_live.py` | Live Resolve session tests (requires Resolve running) |
| README | `README.md` | Full setup + usage instructions |
| Progress Doc | `PROGRESS.md` | This file |

#### Design Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| UI shell | Browser tab (FastAPI serves `frontend/`) | Fastest iteration; no extra deps; matches author workflow |
| Embeddings | Always local (`all-MiniLM-L6-v2`) | Index reusable regardless of LLM provider |
| PDF ingestion | `pypdf` at ingest time | Small dependency and sufficient text extraction for the project documentation |
| Model switching | Live via UI dropdown + `/api/set-model` | No restart required; user-friendly |
| LLM default | `claude-sonnet-4-5` (Anthropic) | Good balance of speed/quality; overridable via config/UI |
| Tool validation | Against `TOOL_REGISTRY` in planner | Prevents hallucinated tool names from reaching executor |
| Confirmation gate | Tag-based (`destructive: True/False` at definition) | Never inferred at runtime; regression-tested |
| Auto-launch Resolve | `open -a "DaVinci Resolve"` + 15s wait | User convenience; fails gracefully if already running |

---

## Known Issues / TODOs

| Priority | Issue | Notes |
|---|---|---|
| 🔴 High | Resolve API connection untested end-to-end | Requires Resolve running with scripting enabled |
| 🟡 Medium | Ollama tool-call support varies by model | Some models don't support structured tool calls; planner falls back to JSON parsing |
| 🟡 Medium | `trim_clip` uses `SetClipProperty` — not official API | Needs verification against real Resolve scripting API; may need different approach |
| 🟡 Medium | Confirmation graph resume path | After user confirms via UI, the graph re-invokes from executor. Session state stored in-memory — works for single-user v1, will need rework for multi-session |
| 🟢 Low | No streaming support | Responses arrive complete, not streamed token-by-token; adds latency feel |
| 🟢 Low | Action log in sidebar not auto-updated from WS responses | JS side needs to parse execution_results from WS response to add log entries |
| 🟢 Low | Popup Markdown is a supported subset | Full CommonMark is intentionally not included in the embedded script |

---

## Feature Backlog (Post-MVP)

### Phase 2
- [ ] Add user manual to the RAG index for richer feature Q&A
- [ ] Expand action set: subtitle track management, additional timeline operations
- [ ] Streaming token output for lower perceived latency

### Phase 3
- [ ] Evaluate Workflow Integration Plugin packaging (Studio-only; provides native menu entry)
- [ ] `pywebview` wrapper option for a more native macOS window
- [ ] Multi-session / persistent conversation history

### Phase 4
- [ ] Color-grading action support (LUT application, preset grades — no node-level manipulation)
- [ ] Fusion page actions (if API supports them)

---

## File Structure (as built)

```
resolve-assistant/
├── backend/
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py                  # Settings loader
│   ├── logging_config.py          # Structured logging
│   ├── agent/
│   │   ├── graph.py               # LangGraph graph
│   │   ├── state.py               # AgentState schema
│   │   └── nodes/
│   │       ├── router.py
│   │       ├── rag.py
│   │       ├── planner.py
│   │       ├── confirmation.py
│   │       ├── executor.py
│   │       └── reporter.py
│   ├── llm/
│   │   ├── base.py                # LLMProvider ABC
│   │   ├── anthropic_provider.py
│   │   ├── ollama_provider.py
│   │   └── factory.py
│   ├── rag/
│   │   ├── embeddings.py          # Local Chroma/ONNX embeddings
│   │   ├── ingest.py              # PDF → Chroma pipeline
│   │   └── retriever.py
│   └── resolve/
│       ├── connection.py          # DaVinciResolveScript bridge
│       ├── tools.py               # 15 tool functions
│       └── tool_registry.py       # Tags + LLM schemas
├── frontend/
│   ├── index.html                 # Chat UI
│   ├── style.css                  # Premium dark mode
│   └── chat.js                    # WS client + model selector
├── data/
│   └── scripting_api_docs/        # Converted text files (auto-created by ingest)
├── tests/
│   ├── unit/
│   │   ├── test_tools_mocked.py
│   │   └── test_confirmation_gate.py
│   └── integration/
│       └── test_tools_live.py
├── config.yaml
├── requirements.txt
├── .env.example
├── README.md
└── PROGRESS.md
```

---

## Next Steps (Immediate)

1. **Install dependencies:**
   ```bash
   cd resolve-assistant
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure `.env`:**
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY
   ```

3. **Run doc ingestion:**
   ```bash
   python -m backend.rag.ingest
   ```

4. **Start the server:**
   ```bash
   uvicorn backend.main:app --reload
   # Then open http://localhost:8000
   ```

5. **Verify end-to-end** with Resolve running:
   - Ask a Q&A question → verify RAG answer cites a real API method
   - Ask to create a timeline → verify it appears in Resolve
   - Ask to delete a clip → verify confirmation prompt appears

---

*Last updated: 2026-09-20*
