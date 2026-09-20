"""
Router node: classifies user message as 'qa' (informational) or 'action'.

Classification strategy (in order):
  1. Deterministic ACTION patterns  — live-state queries, imperatives → always action
  2. Deterministic QA patterns      — how-to / conceptual → always qa
  3. LLM fallback                   — for genuinely ambiguous messages
  4. Hard fallback: 'action'        — better to try and fail gracefully than refuse
"""
from __future__ import annotations

import re
from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.logging_config import get_logger
from backend.llm.errors import redact_secrets

logger = get_logger(__name__)


# ── Patterns that are ALWAYS action ──────────────────────────────────────────
# Covers: live-state queries, imperatives, mutations, enumerations
_ACTION_PATTERNS = [
    # Live state queries  ─ "what is/are/was the current/active/open ..."
    r"\b(what\s+(is|are|was|'s)\s+(the\s+)?(current|active|open|my|this)\s+\w+)",
    r"\b(tell\s+me\s+(the|about\s+the|what)\s+(current|active|open|my))",
    r"\b(show\s+me\s+(the|my|all|current))",
    r"\b(get\s+the\s+(current|active|open|my))",
    r"\b(list\s+(all\s+)?(the\s+)?(clips?|timelines?|tracks?|markers?|projects?|folders?|bins?|renders?))",

    # Live-state nouns without explicit qualifier — asking about live data
    r"\b(current\s+(project|timeline|clip|track|marker|timecode|page|folder|database|frame\s+rate|fps|settings?))\b",
    r"\b(active\s+(project|timeline|clip|track))\b",
    r"\b(frame\s+rate|fps|resolution|timecode|start\s+frame|end\s+frame)\b",
    r"\b(how\s+many\s+(clips?|timelines?|tracks?|markers?|projects?|frames?|items?))",
    r"\b(name\s+of\s+(the\s+)?(current|active|open|my))",
    r"\b(project\s+name|timeline\s+name|clip\s+name)\b",

    # MCP capability and domain state are live agent state, not documentation
    # questions. Route these through the planner so inactive-domain policy is
    # enforced and the user gets an explicit capability response.
    r"\b(list|show|what|which)\b.*\b(mcp|domains?|tools?)\b",

    # Imperative action verbs — only when NOT preceded by "how do/to/can"
    # Uses a negative lookbehind simulation: check start-of-string
    r"(?<!\bhow\s)(^|\.\s*)(add|create|delete|remove|rename|move|set|export|import|render|duplicate|"
    r"archive|save|open|close|switch|insert|append|replace|link|unlink|lock|enable|disable|"
    r"clear|reset|assign|activate|deactivate|start|stop|update|apply|load|refresh)\b",

    # Marker / flag operations
    r"\b(add|create|delete|remove|set)\s+a?\s*(red|blue|green|yellow|purple|cyan|orange|grey)?\s*marker\b",
    r"\b(marker|flag)\s+(at|on|to|from)\s+(frame|timecode)",

    # Clip / track / timeline operations
    r"\b(delete|remove|move|duplicate|rename)\s+(the\s+)?(first|second|third|last|\d+\w*|this|that)?\s*(clip|track|timeline|item|folder)\b",
    r"\b(on\s+(video|audio)\s+track)",
    r"\b(clips?\s+on\s+(video|audio|track))",
]

# ── Patterns that are ALWAYS qa ───────────────────────────────────────────────
# Covers: conceptual questions, how-to, documentation lookups
_QA_PATTERNS = [
    r"\bhow\s+(do\s+I|to|can\s+I|would\s+I)\b",
    r"\bwhat\s+(does|is\s+the\s+purpose|is\s+a|are\s+the\s+(differences?|options?|settings?))\b",
    r"\bexplain\b",
    r"\bwhat\s+is\s+(color\s+grading|fusion|fairlight|the\s+api|davinci\s+resolve)\b",
    r"\b(difference\s+between|comparison\s+of|vs\.?)\b",
    r"\bcan\s+davinci\s+resolve\b",
    r"\bwhat\s+format\s+(does|should|can)\b",
    r"\b(what|which)\s+(codec|format|preset|setting)\s+(should|do\s+i|is\s+best)\b",
]

_ACTION_RE = [re.compile(p, re.IGNORECASE) for p in _ACTION_PATTERNS]
_QA_RE     = [re.compile(p, re.IGNORECASE) for p in _QA_PATTERNS]


# Strong QA prefixes — checked BEFORE action patterns so "how do I add X" → qa
_HOWTO_RE = re.compile(
    r"^(how\s+(do\s+I|to|can\s+I|would\s+I|should\s+I)|"
    r"(explain|what\s+is\s+a|what\s+are\s+the|what\s+does|"
    r"difference\s+between|can\s+davinci|what\s+(codec|format|preset|setting)\s+(should|is\s+best)))",
    re.IGNORECASE,
)


def _deterministic_classify(msg: str) -> str | None:
    """
    Returns 'action', 'qa', or None (needs LLM).

    Order:
      1. How-do-I / conceptual prefix  → qa  (prevents "how do I add" hitting action)
      2. Action patterns               → action
      3. Remaining qa patterns         → qa
      4. None                          → LLM fallback
    """
    # 1. Hard QA prefixes — always qa regardless of what follows
    if _HOWTO_RE.match(msg.strip()):
        return "qa"

    # 2. Action patterns
    for pattern in _ACTION_RE:
        if pattern.search(msg):
            return "action"

    # 3. Remaining QA patterns
    for pattern in _QA_RE:
        if pattern.search(msg):
            return "qa"

    return None


def _build_system() -> str:
    return """You are a router for a DaVinci Resolve AI assistant.
Classify the user's latest message into exactly one of:
  - "qa"     : The user is asking a CONCEPTUAL question about how Resolve works, what a feature means, or how to do something in general. These questions can be answered from documentation WITHOUT connecting to Resolve.
  - "action" : The user wants to READ LIVE DATA from Resolve (e.g. current project name, timeline clips, frame rate) OR perform any operation inside Resolve (add markers, create timelines, delete clips, render, etc.).

IMPORTANT: Any question about the CURRENT/ACTIVE state of Resolve is ALWAYS "action", not "qa".
Examples:
  "What is the current project name?" → action
  "List the clips on the timeline"    → action
  "What is the frame rate?"           → action
  "How do I add a marker in Resolve?" → qa
  "What is color grading?"            → qa

Reply with ONLY the word "qa" or "action". No explanation, no punctuation."""


async def run(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"].strip()

    # 1. Deterministic classification (fast, no LLM call)
    intent = _deterministic_classify(user_message)
    if intent:
        logger.info("Router classified message (deterministic)", intent=intent, message=user_message[:60])
        return {"intent": intent}

    # 2. LLM fallback for genuinely ambiguous messages
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]

    try:
        provider = get_provider(state.get("llm_provider"), state.get("llm_model"))
        response = provider.generate(
            messages=llm_messages,
            system=_build_system(),
        )
        intent_raw = response["content"].strip().lower()
        intent = "action" if "action" in intent_raw else "qa"
    except Exception as exc:
        logger.error("Router LLM call failed — defaulting to action", error=redact_secrets(exc))
        intent = "action"   # ← safer fallback: try to execute; planner will reject if truly out-of-scope

    logger.info("Router classified message (LLM)", intent=intent, message=user_message[:60])
    return {"intent": intent}
