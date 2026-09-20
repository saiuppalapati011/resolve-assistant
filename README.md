# Resolve AI Assistant

A standalone macOS companion app for DaVinci Resolve — AI-powered Q&A grounded in the official documentation, plus real automation of 15 Resolve actions via natural language.

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.11
- DaVinci Resolve installed at `/Applications/DaVinci Resolve/`
- Resolve scripting enabled: **Preferences → General → External scripting → Local**
- Anthropic API key (for cloud LLM) **or** [Ollama](https://ollama.com) running locally

## Setup

```bash
# 1. Create virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Run document ingestion (once, and whenever docs change)
python -m backend.rag.ingest
```

## Running

```bash
# Start the backend (from resolve-assistant/)
uvicorn backend.main:app --reload

# Open the UI
open http://localhost:8000
```

## Running the Resolve popup

The project also includes a small Resolve/Fusion script client at
`resolve_scripts/ResolveAssistant.py`. Install it into Resolve's
`Fusion/Scripts/Utility` folder and run it from
**Workspace > Scripts > Utility**. The popup uses the lightweight `/api/simple/*`
endpoints and does not show old chat threads.

When the popup opens, it starts the local backend automatically when it is not
already running and enables the UI as soon as FastAPI is reachable. Resolve/MCP
connectivity is shown separately because it may finish warming in the
background. The popup includes Provider, Model, Apply, and Close controls. If
the project is stored somewhere other than the default development path, set
`RESOLVE_ASSISTANT_DIR` and optionally `RESOLVE_ASSISTANT_PYTHON` before
launching Resolve.
If startup fails, inspect `data/popup_backend.log` in that directory.

The popup keeps only one temporary conversation while it is open. Explicit
requests such as `remember that I prefer concise answers` are stored in
`data/assistant_memory.json` and reused in later requests. The full browser UI
and legacy chat endpoints remain available as a development fallback.

### Documentation-first web fallback

For conceptual questions, the assistant searches the local Resolve
documentation first. If the grounded answer says the indexed documentation
does not cover the question, it performs a small web search, prioritizing
Blackmagic Design pages, and asks the selected model to answer from the
returned snippets with source URLs. Disable this fallback with
`WEB_SEARCH_ENABLED=false` in `.env` if the machine should stay offline.

## Model Switching

Use the **Provider** and **Model** dropdowns in the sidebar to switch between:
- **Anthropic Claude** — Claude Opus 4.5, Sonnet 4.5, Haiku 4.5, and more
- **Google Gemini** — Gemini Flash and Pro models
- **Ollama (Local)** — any model pulled via `ollama pull <model>`

Click **Apply Model** to switch live — no restart needed.

## Documentation Index

Place additional PDF or Markdown documentation files in `../Documentations/` (or `data/scripting_api_docs/`) and re-run:

```bash
python -m backend.rag.ingest
```

## Running Tests

```bash
# Unit tests (no Resolve required)
pytest tests/unit/ -v

# Integration tests (requires live Resolve session)
pytest tests/integration/ -m integration -v
```

## Project Structure

See [`PROGRESS.md`](PROGRESS.md) for the original progress breakdown and
[`docs/IMPLEMENTATION_HISTORY.md`](docs/IMPLEMENTATION_HISTORY.md) for the
complete implementation history, fixes, testing evidence, and operational
notes.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-5` | Model ID override |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `RESOLVE_ASSISTANT_DIR` | development checkout path | Folder containing `backend/main.py` and the virtual environment |
| `RESOLVE_ASSISTANT_PYTHON` | `<assistant-dir>/.venv/bin/python` | Python executable used by the Resolve popup to start the backend |
| `RESOLVE_ASSISTANT_URL` | `http://127.0.0.1:8000` | Local backend URL used by the popup |
