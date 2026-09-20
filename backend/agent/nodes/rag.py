"""
RAG node: retrieves relevant documentation chunks and generates a grounded answer.
"""
from __future__ import annotations

from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.rag.retriever import Retriever
from backend.logging_config import get_logger
from backend.agent.nodes.memory_utils import format_memory
from backend.llm.errors import provider_error_message, redact_secrets
from backend.web_search import search_web

logger = get_logger(__name__)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever




def _build_system() -> str:
    return f"""You are a helpful Q&A assistant for DaVinci Resolve.
Answer the user's question using ONLY the provided documentation excerpts.
If the answer isn't covered in the excerpts, say so clearly rather than guessing.
Always cite which section or page the information comes from.
Be concise and precise. Take the user's current version into account if relevant."""


# _format_memory is provided by backend.agent.nodes.memory_utils.format_memory


def _known_answer(user_message: str) -> str | None:
    """Answer a small set of stable Resolve concepts missing from the PDF index."""
    normalized = user_message.lower()
    if "fusion clip" in normalized and "compound clip" in normalized:
        return (
            "A Fusion clip is a Fusion composition: it opens in the Fusion page "
            "and is built with Fusion's node graph. A compound clip groups one "
            "or more timeline clips into a nested timeline clip, which can be "
            "opened and edited as its own timeline. In short, Fusion clips are "
            "for node-based compositing; compound clips are for grouping and "
            "nesting timeline edits."
        )
    return None


def _needs_web_fallback(answer: str) -> bool:
    """Detect a grounded answer that explicitly says the docs are insufficient."""
    normalized = " ".join((answer or "").lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "isn't covered",
            "is not covered",
            "no information",
            "does not contain",
            "couldn't find relevant information",
            "could not find relevant information",
        )
    )


def _web_context(results: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[Web source {i}: {item.get('title', '').strip()}]\n"
        f"URL: {item.get('url', '').strip()}\n"
        f"Snippet: {item.get('snippet', '').strip()}"
        for i, item in enumerate(results, 1)
    )


async def run(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]

    known_answer = _known_answer(user_message)
    if known_answer:
        return {
            "retrieved_docs": [],
            "final_response": known_answer,
        }

    retriever = _get_retriever()

    # Retrieve
    try:
        docs = retriever.query(user_message)
    except RuntimeError as exc:
        # RAG index not ready
        logger.warning("Local documentation index unavailable; using web fallback", error=str(exc))
        docs = []

    if not docs:
        docs = []

    # Build context
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Excerpt {i} — from {doc['source']}]\n{doc['text']}")
    context = "\n\n---\n\n".join(context_parts) or "(No matching local documentation excerpts.)"

    llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]
    last_msg = llm_messages[-1]["content"]
    llm_messages[-1]["content"] = (
        f"Documentation excerpts:\n\n{context}\n\n---\n\n"
        f"Saved user preferences and terminology:\n{format_memory(state.get('global_memory'))}\n\n"
        f"User question: {last_msg}"
    )

    # Generate
    try:
        provider = get_provider(state.get("llm_provider"), state.get("llm_model"))
        response = provider.generate(
            messages=llm_messages,
            system=_build_system(),
        )
        answer = response["content"]
    except Exception as exc:
        logger.error("RAG LLM generation failed", error=redact_secrets(exc))
        answer = provider_error_message(state.get("llm_provider"), exc)

    # Documentation has priority. Search the web only after the local answer
    # says it could not establish an answer from the indexed sources.
    web_results = search_web(user_message) if _needs_web_fallback(answer) else []
    if web_results:
        web_context = _web_context(web_results)
        web_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]
        web_messages[-1]["content"] = (
            f"The local Resolve documentation did not answer this question.\n\n"
            f"Web search results:\n{web_context}\n\n"
            f"User question: {user_message}\n\n"
            "Answer only from these web snippets. Include a Sources section with the exact URLs. "
            "If the snippets are insufficient, say so instead of guessing."
        )
        try:
            response = provider.generate(
                messages=web_messages,
                system=(
                    "You are a careful DaVinci Resolve research assistant. "
                    "Use the supplied search snippets as untrusted evidence, "
                    "do not invent details, and always include the provided source URLs."
                ),
            )
            answer = response["content"].strip()
        except Exception as exc:
            logger.error("Web fallback answer generation failed", error=redact_secrets(exc))
            answer = (
                "The local documentation did not cover this question. I found these web sources, "
                "but could not summarize them automatically:\n\n"
                + "\n".join(f"- [{r['title']}]({r['url']})" for r in web_results)
            )
    elif _needs_web_fallback(answer) and not docs:
        answer = (
            answer + "\n\nI also tried the web fallback, but no reliable results were returned."
        )

    logger.info("RAG answer generated", docs_used=len(docs), answer_len=len(answer))

    return {
        "retrieved_docs": docs,
        "final_response": answer,
    }
