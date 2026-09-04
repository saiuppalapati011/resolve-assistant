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

The backend will attempt to launch DaVinci Resolve automatically if it isn't running.

## Model Switching

Use the **Provider** and **Model** dropdowns in the sidebar to switch between:
- **Anthropic Claude** — Claude Opus 4.5, Sonnet 4.5, Haiku 4.5, and more
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

See [`PROGRESS.md`](PROGRESS.md) for a detailed breakdown of all files and design decisions.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-5` | Model ID override |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
