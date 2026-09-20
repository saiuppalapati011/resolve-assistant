# Resolve Assistant popup script

`ResolveAssistant.py` is the simple Resolve/Fusion UI client. It is intentionally
small: it creates one popup, sends requests to the local FastAPI server, and
does not load or display old chat threads.

## Run the backend

**The backend starts automatically.** When you open `ResolveAssistant.py` inside DaVinci Resolve, it checks the local health endpoint, launches FastAPI (`uvicorn`) if needed, and enables the popup as soon as FastAPI responds. Resolve/MCP connectivity is shown separately while it warms in the background. When you close the popup, it stops only the backend process that it started itself.

If you need to debug or run it manually (for example, to see the logs in your terminal), you can start it from the `resolve-assistant` directory:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
If the script detects the server is already running, it will safely attach to it instead of spawning a new process.

The script connects to `http://127.0.0.1:8000` by default. Set
`RESOLVE_ASSISTANT_URL` if the backend uses another address. Set
`RESOLVE_ASSISTANT_DIR` to the absolute folder containing this project's
`backend/` directory and `.venv` on your machine. You can override the Python
executable with `RESOLVE_ASSISTANT_PYTHON`.

## Install the script

Copy `ResolveAssistant.py` into Resolve's Utility script directory. On macOS
the active installation path is:

```text
/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/
```

Then open DaVinci Resolve and run it from **Workspace > Scripts > Utility >
ResolveAssistant**.

The exact folder can be confirmed from Resolve's **Help > Documentation >
Developer** examples for the installed version.

## What is intentionally not included

- No browser UI is required to run the popup.
- No visible chat/thread list.
- No chat title database for the popup.
- No LangGraph, MCP, or LLM dependencies inside Resolve's Python runtime.
- No automatic deletion of the existing browser fallback.

The FastAPI process still owns the agent, Resolve MCP calls, confirmation gate,
and provider keys. The popup only needs standard-library HTTP and Resolve's
built-in UI Manager. The popup header includes Provider, Model, Apply, and
Close controls. It passes the selected provider/model with each request and
also updates the backend's default selection when Apply is pressed.

Assistant responses are rendered with a small dependency-free Markdown
renderer. It supports the formatting normally returned by the LLM (lists,
links, headings, bold, italics, and inline code) and escapes raw HTML before it
is inserted into the Resolve UI.

If startup fails, the popup reports the child-process exit instead of waiting
silently. The backend launch log is written to
`data/popup_backend.log` inside the assistant directory.

## Global memory

The backend stores explicit `remember ...` requests in
`data/assistant_memory.json`. The popup's **Memory** button displays the saved
items. The current simple UI does not expose item-by-item editing yet; clear
the file or call `DELETE /api/simple/memory` to reset it during development.
