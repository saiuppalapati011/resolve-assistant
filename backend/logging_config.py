"""
Logging configuration for the Resolve AI Assistant.

Uses structlog for structured keyword-arg logging: logger.info("msg", key=val).
Outputs:
  - Console: human-readable output
  - File:    newline-delimited JSON for machine parsing / action audit trail
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_level_str: str = "INFO", log_file: str = "resolve_assistant.log") -> None:
    """Configure structlog. Call once at app startup."""
    # Lazy import to avoid circular dep at module load time
    from backend.config import settings
    log_level_str = settings.log_verbosity
    log_file = settings.log_file

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib for uvicorn / third-party libs
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        stream=sys.stdout,
    )


def get_logger(name: str):
    """Return a structlog logger that supports keyword args: logger.info('msg', key=val)."""
    return structlog.get_logger(name)
