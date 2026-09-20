"""Small Resolve/Fusion popup client for the Resolve AI Assistant.

Install this file in Resolve's Scripts/Edit folder and run it from the
Workspace > Scripts menu. The heavy agent stays in the local FastAPI process;
this script only owns the small Resolve-native window.

Set RESOLVE_ASSISTANT_URL if the backend is not running on port 8000.
"""
from __future__ import print_function

import html
import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

API_ROOT = os.environ.get("RESOLVE_ASSISTANT_URL", "http://127.0.0.1:8000").rstrip("/")
SESSION_ID = str(uuid.uuid4())

# Backend location for auto-start. Set RESOLVE_ASSISTANT_DIR once on each
# machine where the script is copied into Resolve's user Scripts folder.
# The old absolute path remains only as a same-machine fallback.
BACKEND_DIR = os.environ.get(
    "RESOLVE_ASSISTANT_DIR",
    "/Users/saiuppalapati/Downloads/Resolver/resolve-assistant",
)
VENV_PYTHON = os.environ.get(
    "RESOLVE_ASSISTANT_PYTHON",
    os.path.join(BACKEND_DIR, ".venv", "bin", "python"),
)
backend_process = None
backend_log_handle = None

DEFAULT_MODEL_CATALOG = {
    "anthropic": [
        {"id": "claude-sonnet-4-5", "label": "Claude Sonnet 4.5"},
        {"id": "claude-haiku-4-5", "label": "Claude Haiku 4.5"},
    ],
    "gemini": [
        {"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash-Lite"},
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
    ],
    # The backend replaces this list with the models actually installed in
    # Ollama. Keeping one common local default prevents an empty dropdown while
    # the backend is starting.
    "ollama": [{"id": "llama3.1", "label": "llama3.1"}],
    "current_provider": "gemini",
    "current_model": "gemini-3.5-flash-lite",
}


def _backend_address():
    parsed = urlparse(API_ROOT)
    return parsed.hostname or "127.0.0.1", parsed.port or 8000


def start_backend():
    """Launch the FastAPI backend process asynchronously when needed."""
    global backend_process, backend_log_handle
    if backend_process is not None and backend_process.poll() is None:
        return True
    backend_process = None

    # A manually started backend is fine; attach to it instead of spawning a
    # second server process.
    try:
        urllib.request.urlopen(API_ROOT + "/health", timeout=1).close()
        print("Resolve Assistant: Backend is already running.")
        return True
    except urllib.error.HTTPError:
        # A responding HTTP service is already occupying the configured URL.
        # Do not launch a second Uvicorn process just because its health route
        # returned an application-level error.
        print("Resolve Assistant: Backend responded with an HTTP error; attaching to it.")
        return True
    except Exception:
        pass

    if not os.path.isdir(BACKEND_DIR):
        print("Resolve Assistant: Backend folder does not exist:", BACKEND_DIR)
        return False
    if not os.path.isfile(VENV_PYTHON):
        print("Resolve Assistant: Python executable does not exist:", VENV_PYTHON)
        return False

    host, port = _backend_address()
    print("Resolve Assistant: Starting backend...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Resolve injects its own Python paths which break the virtual environment. Strip them.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    
    try:
        log_dir = os.path.join(BACKEND_DIR, "data")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, "popup_backend.log")
        backend_log_handle = open(log_path, "a")
        backend_process = subprocess.Popen(
            [VENV_PYTHON, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", str(port)],
            cwd=BACKEND_DIR,
            env=env,
            stdout=backend_log_handle,
            stderr=subprocess.STDOUT,
        )
        return True
    except Exception as e:
        print("Resolve Assistant: Failed to start backend:", e)
        if backend_log_handle is not None:
            try:
                backend_log_handle.close()
            except Exception:
                pass
            backend_log_handle = None
        backend_process = None
        return False


def stop_backend():
    """Release this popup's handle without stopping the shared backend.

    More than one Resolve window can use the same local backend. Terminating
    the child when one popup closes leaves another popup stuck on
    "Waiting for backend...". The backend is lightweight and can stay alive;
    a later popup will reuse it through the health check.
    """
    global backend_process, backend_log_handle
    if backend_process is not None and backend_process.poll() is None:
        print("Resolve Assistant: Leaving shared backend running.")
    backend_process = None
    if backend_log_handle is not None:
        try:
            backend_log_handle.close()
        except Exception:
            pass
        backend_log_handle = None


def http_json(path, payload=None, method="GET", timeout=120):
    """Tiny standard-library HTTP client; Resolve does not need extra packages."""
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        API_ROOT + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))
        except Exception:
            detail = {"detail": str(error)}
        raise RuntimeError(detail.get("detail", "Assistant service error"))


def get_resolve():
    """Use Resolve's injected variable when available, otherwise connect locally."""
    if "resolve" in globals() and globals()["resolve"] is not None:
        return globals()["resolve"]
    try:
        import DaVinciResolveScript as dvr_script
        return dvr_script.scriptapp("Resolve")
    except Exception:
        return None


def get_bmd():
    if "bmd" in globals() and globals()["bmd"] is not None:
        return globals()["bmd"]
    try:
        import DaVinciResolveScript as dvr_script
        return dvr_script
    except Exception:
        return None


def _inline_markdown(value):
    """Render the small Markdown subset commonly returned by the LLM."""
    text = html.escape(str(value or ""), quote=False)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda match: '<a href="%s">%s</a>' % (
            html.escape(match.group(2), quote=True),
            match.group(1),
        ),
        text,
    )
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: "<b>%s</b>" % (m.group(1) or m.group(2)), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)", lambda m: "<i>%s</i>" % (m.group(1) or m.group(2)), text)
    return text


def markdown_to_html(value):
    """Convert common Markdown to safe QTextEdit HTML for the Resolve popup."""
    lines = str(value or "").splitlines()
    output = []
    list_type = None

    def close_list():
        nonlocal list_type
        if list_type:
            output.append("</%s>" % list_type)
            list_type = None

    for raw_line in lines:
        line = raw_line.strip()
        bullet = re.match(r"^[-*+]\s+(.+)$", line)
        numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
        heading = re.match(r"^#{1,3}\s+(.+)$", line)
        if bullet or numbered:
            desired = "ul" if bullet else "ol"
            if list_type != desired:
                close_list()
                output.append("<%s>" % desired)
                list_type = desired
            output.append("<li>%s</li>" % _inline_markdown((bullet or numbered).group(1)))
        else:
            close_list()
            if not line:
                output.append("<br>")
            elif heading:
                output.append("<b>%s</b><br>" % _inline_markdown(heading.group(1)))
            else:
                output.append("%s<br>" % _inline_markdown(line))
    close_list()
    rendered = "".join(output)
    return rendered[:-4] if rendered.endswith("<br>") else rendered


def safe_text(value):
    """Backward-compatible name used by status/error rendering."""
    return markdown_to_html(value)


def current_context(resolve):
    """Return a short status line without making the assistant depend on it."""
    if not resolve:
        return "Resolve unavailable"
    try:
        manager = resolve.GetProjectManager()
        project = manager.GetCurrentProject() if manager else None
        project_name = project.GetName() if project else "No project"
        timeline = project.GetCurrentTimeline() if project else None
        timeline_name = timeline.GetName() if timeline else "No timeline"
        page = ""
        get_page = getattr(resolve, "GetCurrentPage", None)
        if get_page:
            page = get_page() or ""
        return "%s · %s%s" % (
            timeline_name,
            project_name,
            (" · " + page.title()) if page else "",
        )
    except Exception:
        return "Resolve connected"


def main():
    resolve = get_resolve()
    bmd = get_bmd()
    if not resolve or not bmd:
        print("Resolve Assistant: could not connect to DaVinci Resolve")
        return

    fusion = resolve.Fusion()
    ui = fusion.UIManager
    dispatcher = bmd.UIDispatcher(ui)

    window = dispatcher.AddWindow(
        {
            "ID": "ResolveAssistantWindow",
            "WindowTitle": "Resolve Assistant",
            "Geometry": [200, 120, 500, 680],
            "Events": {"AssistantResult": True},
            "StyleSheet": """
                QWidget { background: #171717; color: #e8e8e8; }
                QLabel { color: #b8b8b8; }
                QTextEdit, QLineEdit { background: #242424; border: 1px solid #3a3a3a; border-radius: 10px; padding: 8px; color: #ededed; }
                QPushButton { background: #2d2d2d; border: 1px solid #494949; border-radius: 8px; padding: 7px 12px; color: #eeeeee; }
                QPushButton:hover { background: #3a3a3a; }
                QPushButton#Close { color: #e07a68; }
                QPushButton#Close:hover { background: #512d2a; }
                QComboBox { background: #242424; border: 1px solid #3a3a3a; border-radius: 6px; padding: 5px 8px; color: #ededed; }
            """,
        },
        [
            ui.VGroup(
                {"Spacing": 8},
                [
                    ui.HGroup(
                        {"Weight": 0, "Spacing": 6},
                        [
                            ui.Label({"ID": "Title", "Text": "Resolve Assistant", "Weight": 1}),
                            ui.Button({"ID": "Memory", "Text": "Memory"}),
                            ui.Button({"ID": "Clear", "Text": "Clear"}),
                            ui.Button({"ID": "Close", "Text": "Close"}),
                        ],
                    ),
                    ui.Label({"ID": "Context", "Text": current_context(resolve), "Weight": 0}),
                    ui.HGroup(
                        {"Weight": 0, "Spacing": 6},
                        [
                            ui.Label({"ID": "ProviderLabel", "Text": "Provider", "Weight": 0}),
                            ui.ComboBox({"ID": "Provider", "Weight": 1}),
                            ui.Label({"ID": "ModelLabel", "Text": "Model", "Weight": 0}),
                            ui.ComboBox({"ID": "Model", "Weight": 2}),
                            ui.Button({"ID": "Apply", "Text": "Apply", "Weight": 0}),
                        ],
                    ),
                    ui.TextEdit(
                        {
                            "ID": "Transcript",
                            "ReadOnly": True,
                            "HTML": "<p style='color:#929292'>Ask about the current Resolve project or request an action.</p>",
                            "Weight": 1,
                        }
                    ),
                    ui.TextEdit(
                        {
                            "ID": "Prompt",
                            "PlaceholderText": "Ask Resolve Assistant...",
                            "MinimumSize": [0, 70],
                            "MaximumSize": [10000, 100],
                            "Weight": 0,
                        }
                    ),
                    ui.HGroup(
                        {"Weight": 0, "Spacing": 6},
                        [
                            ui.Label({"ID": "Status", "Text": "Starting backend...", "Weight": 1}),
                            ui.Button({"ID": "Confirm", "Text": "Confirm", "Visible": False}),
                            ui.Button({"ID": "Cancel", "Text": "Cancel", "Visible": False}),
                            ui.Button({"ID": "Send", "Text": "Send"}),
                        ],
                    ),
                ],
            )
        ],
    )
    items = window.GetItems()
    state = {
        "working": False,
        "pending": False,
        "closed": False,
        "backend_ready": False,
        "resolve_connected": False,
        "catalog": {},
        "provider": "",
        "model": "",
        "handled_events": {},
        "updating_catalog": False,
        "last_message_key": None,
    }
    pending_events = queue.Queue()
    startup = {
        "started": False,
        "deadline": None,
        "models_requested": False,
        "health_in_flight": False,
        "next_health_check": 0,
    }

    # The popup and its input controls are usable immediately. Send reports a
    # clear status while startup is in progress instead of relying on native
    # disabled-button rendering, which varies across Resolve UIManager builds.
    for control_id in ("Memory", "Apply"):
        items[control_id].Enabled = False
    # Keep the input and single action button usable while startup finishes.
    # The handlers show a clear status when the backend is not ready instead
    # of relying on a disabled native button, which is inconsistent across
    # Resolve UIManager builds.
    items["Prompt"].Enabled = True
    items["Send"].Enabled = True
    items["Clear"].Enabled = True

    def set_status(text):
        items["Status"].Text = text

    def set_backend_state(ready, text, resolve_connected=False):
        state["backend_ready"] = bool(ready)
        state["resolve_connected"] = bool(resolve_connected)
        if not state["working"]:
            for control_id in ("Memory", "Apply"):
                items[control_id].Enabled = bool(ready)
            items["Prompt"].Enabled = True
            items["Send"].Enabled = True
            items["Clear"].Enabled = True
            items["Provider"].Enabled = bool(state["catalog"])
            items["Model"].Enabled = bool(state["catalog"])
        set_status(text)

    def combo_text(control_id):
        """Read ComboBox text across Resolve UI Manager versions."""
        item = items[control_id]
        index = getattr(item, "CurrentIndex", -1)
        if control_id == "Provider":
            text = getattr(item, "CurrentText", "")
            if text:
                return str(text)
            values = state["catalog"].get("providers", [])
        else:
            # The UI displays a friendly label, but the API needs the model id.
            values = state["catalog"].get(state.get("provider", ""), [])
        if control_id == "Model":
            if isinstance(index, int) and 0 <= index < len(values):
                value = values[index]
                return value.get("id", value) if isinstance(value, dict) else str(value)
            return state.get("model", "")
        if isinstance(index, int) and 0 <= index < len(values):
            value = values[index]
            return value.get("id", value) if isinstance(value, dict) else str(value)
        return ""

    def fill_models(provider, selected_model=None):
        models = state["catalog"].get(provider, [])
        state["provider"] = provider
        try:
            items["Model"].Clear()
        except Exception:
            pass
        for model in models:
            model_id = model.get("id", "") if isinstance(model, dict) else str(model)
            label = model.get("label", model_id) if isinstance(model, dict) else model_id
            if model_id:
                items["Model"].AddItem(label)
        if not models:
            items["Model"].AddItem("No models available")
            state["model"] = ""
            return
        desired = selected_model or state.get("model", "")
        selected_index = 0
        for index, model in enumerate(models):
            if model.get("id") == desired:
                selected_index = index
                break
        items["Model"].CurrentIndex = selected_index
        state["model"] = models[selected_index].get("id", "")

    def fill_catalog(catalog):
        catalog = catalog or {}
        state["updating_catalog"] = True
        state["catalog"] = catalog
        providers = ["anthropic", "gemini", "ollama"]
        state["catalog"]["providers"] = providers
        try:
            items["Provider"].Clear()
        except Exception:
            pass
        for provider in providers:
            items["Provider"].AddItem(provider.title())
        selected_provider = catalog.get("current_provider", "")
        if selected_provider not in providers:
            selected_provider = "gemini"
        items["Provider"].CurrentIndex = providers.index(selected_provider)
        state["provider"] = selected_provider
        fill_models(selected_provider, catalog.get("current_model"))
        state["updating_catalog"] = False

    fill_catalog(DEFAULT_MODEL_CATALOG)

    def append_message(role, text):
        label = "You" if role == "user" else "AI"
        color = "#e05b45" if role == "user" else "#78a8d8"
        message_key = (role, str(text))
        # Native Resolve events can occasionally deliver the same HTTP result
        # twice. Suppress only consecutive duplicate assistant messages; a
        # new user message resets the guard so repeating a question remains
        # valid.
        if role == "assistant" and state.get("last_message_key") == message_key:
            return
        if role == "user":
            state["last_message_key"] = None
        existing = getattr(items["Transcript"], "HTML", "") or ""
        addition = (
            "<p><b style='color:%s'>%s</b><br>%s</p>"
            % (color, label, markdown_to_html(text))
        )
        try:
            items["Transcript"].HTML = existing + addition
        except Exception:
            items["Transcript"].Text = existing + "\n%s: %s\n" % (label, text)
        state["last_message_key"] = message_key

    def read_prompt():
        """Read user input across Resolve TextEdit implementations."""
        candidates = []
        for attribute in ("Text", "PlainText", "HTML"):
            try:
                value = getattr(items["Prompt"], attribute, "")
            except Exception:
                value = ""
            if value:
                value = str(value)
                if attribute == "HTML":
                    value = re.sub(r"<[^>]+>", " ", value)
                    value = html.unescape(value)
                candidates.append(value)
        placeholder = "Ask Resolve Assistant..."
        for value in candidates:
            cleaned = value.replace("\u00a0", " ").strip()
            if cleaned and cleaned != placeholder:
                return cleaned
        return ""

    def clear_prompt():
        try:
            items["Prompt"].Clear()
        except Exception:
            try:
                items["Prompt"].Text = ""
            except Exception:
                pass

    def set_working(value):
        state["working"] = value
        items["Send"].Text = "Stop" if value else "Send"
        items["Send"].Enabled = True
        items["Prompt"].Enabled = not value
        for control_id in ("Memory", "Apply"):
            items[control_id].Enabled = False if value else state["backend_ready"]
        items["Clear"].Enabled = not value
        items["Provider"].Enabled = False if value else bool(state["catalog"])
        items["Model"].Enabled = False if value else bool(state["catalog"])
        items["Confirm"].Enabled = not value
        items["Cancel"].Enabled = not value

    def show_confirmation(value):
        state["pending"] = bool(value)
        items["Confirm"].Visible = bool(value)
        items["Cancel"].Visible = bool(value)
        items["Confirm"].Enabled = bool(value) and not state["working"]
        items["Cancel"].Enabled = bool(value) and not state["working"]

    def queue_ui_event(kind, payload):
        if state["closed"]:
            return
        event_id = str(uuid.uuid4())
        event = {"id": event_id, "kind": kind, "payload": payload}
        # Do not call Resolve UI methods from a worker thread. Some Resolve
        # builds expose QueueEvent but silently drop cross-thread events. A
        # standard queue plus a UITimer lets the dispatcher apply responses on
        # its own UI thread and works consistently on those builds.
        pending_events.put(event)

    def request_worker(kind, path, payload=None, method="POST", timeout=120):
        try:
            result = http_json(path, payload, method=method, timeout=timeout)
        except Exception as error:
            result = {"ok": False, "content": str(error), "details": []}
        queue_ui_event(kind, result)

    def provider_changed(ev):
        if state["updating_catalog"]:
            return
        provider = combo_text("Provider").lower()
        if provider in ("anthropic", "gemini", "ollama"):
            fill_models(provider)

    def apply_selection(ev):
        provider = combo_text("Provider").lower()
        model = combo_text("Model")
        if provider not in ("anthropic", "gemini", "ollama") or not model or model == "No models available":
            set_status("Choose an available provider and model")
            return
        state["provider"] = provider
        state["model"] = model
        items["Apply"].Enabled = False
        set_status("Applying model...")
        threading.Thread(
            target=request_worker,
            args=("apply", "/api/set-model", {"provider": provider, "model": model}),
            kwargs={"timeout": 15},
            daemon=True,
        ).start()

    def send(ev):
        if state["working"]:
            set_status("Stopping...")
            threading.Thread(
                target=request_worker,
                args=("cancel", "/api/simple/cancel", {"session_id": SESSION_ID}),
                daemon=True,
            ).start()
            return
        if not state["backend_ready"]:
            set_status("Backend is not ready")
            return
        prompt = read_prompt()
        if not prompt:
            set_status("Enter a prompt")
            return
        clear_prompt()
        append_message("user", prompt)
        show_confirmation(False)
        set_working(True)
        set_status("Working...")
        threading.Thread(
            target=request_worker,
            args=(
                "query",
                "/api/simple/query",
                {
                    "session_id": SESSION_ID,
                    "content": prompt,
                    "provider": state.get("provider"),
                    "model": state.get("model"),
                },
            ),
            daemon=True,
        ).start()

    def confirm(ev):
        if state["working"]:
            return
        if not state["backend_ready"]:
            set_status("Backend is not ready")
            return
        # Hide and disable the confirmation controls before starting the
        # request. This prevents a double-click from submitting two resumes.
        show_confirmation(False)
        set_working(True)
        set_status("Applying action...")
        threading.Thread(
            target=request_worker,
            args=(
                "confirm",
                "/api/simple/confirm",
                {"session_id": SESSION_ID, "confirmed": True},
            ),
            daemon=True,
        ).start()

    def cancel(ev):
        if state["working"]:
            return
        show_confirmation(False)
        set_working(False)
        set_status("Ready")
        threading.Thread(
            target=request_worker,
            args=(
                "confirm",
                "/api/simple/confirm",
                {"session_id": SESSION_ID, "confirmed": False},
            ),
            daemon=True,
        ).start()

    def clear_chat(ev):
        items["Transcript"].HTML = "<p style='color:#929292'>Conversation cleared.</p>"
        state["last_message_key"] = None
        show_confirmation(False)

    def show_memory(ev):
        if not state["backend_ready"]:
            set_status("Backend is not ready")
            return
        threading.Thread(
            target=request_worker,
            args=("memory", "/api/simple/memory"),
            kwargs={"method": "GET", "timeout": 15},
            daemon=True,
        ).start()

    def handle_result(event):
        event_id = event.get("id")
        if event_id:
            if event_id in state["handled_events"]:
                return
            state["handled_events"][event_id] = True
        kind = event.get("kind")
        result = event.get("payload", {})
        
        if kind == "status":
            set_backend_state(
                result.get("ready", False),
                result.get("text", ""),
                result.get("resolve_connected", False),
            )
            return

        if kind == "health":
            startup["health_in_flight"] = False
            if not result.get("ok", True):
                # Keep polling until the startup deadline. A failed request
                # is expected while the child Uvicorn process is booting.
                return

            connected = bool(result.get("resolve_connected"))
            set_backend_state(
                True,
                "Connected" if connected else "Backend ready · Resolve offline",
                connected,
            )

            if not startup["models_requested"]:
                startup["models_requested"] = True
                threading.Thread(
                    target=request_worker,
                    args=("models", "/api/models"),
                    kwargs={"method": "GET", "timeout": 15},
                    daemon=True,
                ).start()
            return

        if kind == "models":
            if not result.get("ok", True):
                set_status("Connected · using built-in model list")
            else:
                fill_catalog(result)
                # A successful catalog response proves that the local HTTP
                # backend is already accepting requests.  Do not make the
                # user wait for the separate health probe before allowing a
                # query; the probe only adds Resolve connection detail.
                set_backend_state(
                    True,
                    "Connected" if state["resolve_connected"] else "Backend ready",
                    state["resolve_connected"],
                )
            return

        if kind == "apply":
            if result.get("ok"):
                items["Apply"].Enabled = state["backend_ready"]
                set_status("Using %s · %s" % (state["provider"].title(), state["model"]))
            else:
                items["Apply"].Enabled = state["backend_ready"]
                append_message("assistant", "Could not apply model: %s" % result.get("content", "Unknown error"))
                set_status("Model selection failed")
            return

        if kind == "memory":
            if not result.get("ok", True):
                append_message("assistant", "Could not load memory: %s" % result.get("content", "Unknown error"))
                return
            memory_items = result.get("items", [])
            if not memory_items:
                append_message("assistant", "No saved preferences or terminology yet.")
            else:
                text = "Saved memory:\n" + "\n".join(
                    "- " + item.get("text", "") for item in memory_items
                )
                append_message("assistant", text)
            return
            
        if kind == "backend_error":
            set_backend_state(False, result.get("text", "Backend unavailable"), False)
            items["Transcript"].HTML = (
                "<p style='color:#e05b45'><b>Connection error</b><br>%s</p>"
                % safe_text(result.get("text", "Could not reach the local backend server."))
            )
            return

        if kind == "cancel":
            show_confirmation(False)
            set_working(False)
            set_status("Ready")
            return
        if not result.get("ok", True):
            append_message("assistant", result.get("content", "Assistant error"))
            show_confirmation(False)
            set_working(False)
            set_status("Error")
            return
        if result.get("content"):
            append_message("assistant", result["content"])
        if result.get("needs_confirmation"):
            show_confirmation(result["needs_confirmation"])
            set_working(False)
            set_status("Confirm the action")
        else:
            show_confirmation(False)
            set_working(False)
            set_status("Ready")

    def close(ev):
        state["closed"] = True
        if poll_timer is not None:
            try:
                poll_timer.Stop()
            except Exception:
                pass
        if state["working"]:
            threading.Thread(
                target=request_worker,
                args=("cancel", "/api/simple/cancel", {"session_id": SESSION_ID}),
                daemon=True,
            ).start()
        stop_backend()
        dispatcher.ExitLoop()

    window.On.ResolveAssistantWindow.Close = close
    window.On.Send.Clicked = send
    window.On.Confirm.Clicked = confirm
    window.On.Cancel.Clicked = cancel
    window.On.Clear.Clicked = clear_chat
    window.On.Memory.Clicked = show_memory
    window.On.Provider.CurrentIndexChanged = provider_changed
    window.On.Apply.Clicked = apply_selection
    window.On.Close.Clicked = close

    def poll_events(ev):
        # Keep the short startup state machine on the UI timer, but never do
        # network I/O on the Resolve UI thread. A synchronous health request
        # here made the popup look frozen and caused buttons to miss clicks.
        if not startup["started"]:
            startup["started"] = True
            startup["deadline"] = time.time() + 20
            if not start_backend():
                handle_result(
                    {
                        "kind": "backend_error",
                        "payload": {"text": "Backend path is not configured or could not be started."},
                    }
                )

        if startup["started"] and not state["backend_ready"] and startup["deadline"]:
            now = time.time()
            if now >= startup["deadline"]:
                startup["deadline"] = None
                handle_result(
                    {
                        "kind": "backend_error",
                        "payload": {
                            "text": (
                                "Could not reach the local backend after 20 seconds. "
                                "Check data/popup_backend.log."
                            )
                        },
                    }
                )
            elif not startup["health_in_flight"] and now >= startup["next_health_check"]:
                startup["health_in_flight"] = True
                startup["next_health_check"] = now + 0.5
                threading.Thread(
                    target=request_worker,
                    args=("health", "/health"),
                    kwargs={"method": "GET", "timeout": 1},
                    daemon=True,
                ).start()

        while True:
            try:
                event = pending_events.get_nowait()
            except queue.Empty:
                return
            handle_result(event)

    poll_timer = ui.Timer(
        {
            "ID": "ResolveAssistantPollTimer",
            "Interval": 100,
            "SingleShot": False,
        }
    )
    dispatcher.On.Timeout = poll_events

    window.Show()

    # Show the window before startup polling so the user gets instant feedback
    # even when the backend needs to be started.
    try:
        window.Raise()
    except Exception:
        pass

    poll_timer.Start()

    dispatcher.RunLoop()


if __name__ in ("__main__", "__builtin__", "builtins"):
    main()
