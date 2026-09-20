"""
Central settings loader.
Reads config.yaml first, then overlays .env values.
All code imports Settings from here — no concrete provider class is imported
outside backend/llm/factory.py.
"""
from __future__ import annotations

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Resolve .env relative to this file's parent (project root)
_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(_ROOT / ".env")


def _load_yaml() -> dict:
    config_path = _ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()


class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = os.getenv(
        "LLM_PROVIDER", _cfg.get("llm", {}).get("provider", "anthropic")
    )
    llm_model: str = os.getenv(
        "LLM_MODEL",
        _cfg.get("llm", {}).get(
            _cfg.get("llm", {}).get("provider", "anthropic"), {}
        ).get("default_model", "claude-sonnet-4-5"),
    )

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_default_model: str = (
        _cfg.get("llm", {}).get("anthropic", {}).get("default_model", "claude-sonnet-4-5")
    )
    anthropic_available_models: list[dict] = (
        _cfg.get("llm", {}).get("anthropic", {}).get("available_models", [])
    )
    anthropic_max_tokens: int = (
        _cfg.get("llm", {}).get("anthropic", {}).get("max_tokens", 2048)
    )

    # Ollama
    ollama_host: str = os.getenv(
        "OLLAMA_HOST", _cfg.get("llm", {}).get("ollama", {}).get("host", "http://localhost:11434")
    )
    ollama_default_model: str = (
        _cfg.get("llm", {}).get("ollama", {}).get("default_model", "llama3.1")
    )
    ollama_max_tokens: int = (
        _cfg.get("llm", {}).get("ollama", {}).get("max_tokens", 2048)
    )

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_default_model: str = (
        _cfg.get("llm", {}).get("gemini", {}).get("default_model", "gemini-2.5-flash")
    )
    gemini_max_tokens: int = (
        _cfg.get("llm", {}).get("gemini", {}).get("max_tokens", 2048)
    )

    # ── RAG ──────────────────────────────────────────────────────────────────
    chroma_path: str = str((_ROOT / _cfg.get("rag", {}).get("chroma_path", "./chroma_store")).resolve())
    collection_name: str = _cfg.get("rag", {}).get("collection_name", "resolve_scripting_api")
    rag_top_k: int = _cfg.get("rag", {}).get("top_k", 5)
    embeddings_model: str = (
        _cfg.get("rag", {}).get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
    )
    web_search_enabled: bool = os.getenv(
        "WEB_SEARCH_ENABLED",
        str(_cfg.get("rag", {}).get("web_search_enabled", True)),
    ).lower() in {"1", "true", "yes", "on"}
    web_search_timeout: int = int(
        os.getenv("WEB_SEARCH_TIMEOUT", _cfg.get("rag", {}).get("web_search_timeout", 8))
    )
    web_search_max_results: int = int(
        os.getenv("WEB_SEARCH_MAX_RESULTS", _cfg.get("rag", {}).get("web_search_max_results", 5))
    )

    # ── Docs ─────────────────────────────────────────────────────────────────
    pdf_dir: str = str((_ROOT / _cfg.get("docs", {}).get("pdf_dir", "../Documentations")).resolve())
    text_dir: str = str((_ROOT / _cfg.get("docs", {}).get("text_dir", "./data/scripting_api_docs")).resolve())

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = _cfg.get("server", {}).get("host", "127.0.0.1")
    port: int = _cfg.get("server", {}).get("port", 8000)
    log_level: str = _cfg.get("server", {}).get("log_level", "info")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file: str = str((_ROOT / _cfg.get("logging", {}).get("file", "resolve_assistant.log")).resolve())
    log_verbosity: str = _cfg.get("logging", {}).get("level", "INFO")

    # MCP server used by the assistant. Keep this local and reproducible;
    # packaging can override both values through the environment.
    mcp_server_path: str = os.getenv(
        "MCP_SERVER_PATH",
        str((_ROOT / _cfg.get("mcp", {}).get(
            "server_path", "../mcp-candidates/hoyt2/mcp_server.py"
        )).resolve()),
    )
    mcp_python: str = os.getenv("MCP_PYTHON", os.sys.executable)

settings = Settings()
