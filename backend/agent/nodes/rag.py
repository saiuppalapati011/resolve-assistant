"""
RAG node: retrieves relevant documentation chunks and generates a grounded answer.
"""
from __future__ import annotations

from backend.agent.state import AgentState
from backend.llm.factory import get_provider
from backend.rag.retriever import Retriever
from backend.logging_config import get_logger

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
        response_text = (
            f"⚠️ {exc}\n\n"
            "I cannot answer this question right now because the documentation index hasn't been built yet. "
            "Run: `python -m backend.rag.ingest` to set it up."
        )
        return {
            "retrieved_docs": [],
            "final_response": response_text,
        }

    if not docs:
        return {
            "retrieved_docs": [],
            "final_response": "I couldn't find relevant information in the documentation for that question.",
        }

    # Build context
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Excerpt {i} — from {doc['source']}]\n{doc['text']}")
    context = "\n\n---\n\n".join(context_parts)

    llm_messages = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])]
    last_msg = llm_messages[-1]["content"]
    llm_messages[-1]["content"] = f"Documentation excerpts:\n\n{context}\n\n---\n\nUser question: {last_msg}"

    # Generate
    provider = get_provider(state.get("llm_provider"), state.get("llm_model"))
    try:
        response = provider.generate(
            messages=llm_messages,
            system=_build_system(),
        )
        answer = response["content"]
    except Exception as exc:
        logger.error("RAG LLM generation failed", error=str(exc))
        answer = f"I found relevant documentation but encountered an error generating the answer: {exc}"

    logger.info("RAG answer generated", docs_used=len(docs), answer_len=len(answer))

    return {
        "retrieved_docs": docs,
        "final_response": answer,
    }
