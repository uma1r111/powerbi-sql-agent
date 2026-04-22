# rag/rag_agent.py
"""
RAG pipeline:
  1. Search company docs (include results even at low confidence).
  2. If docs score is below threshold, also search the web.
  3. Send all context to the LLM for synthesis.
  4. If neither source returned anything, the LLM answers from its own knowledge.
     The pipeline NEVER blocks — it always produces an answer.
"""

import logging
import os
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from duckduckgo_search import DDGS

from rag.document_store import document_store

logger = logging.getLogger(__name__)

# Chunks with score >= this are treated as "confident"; below it we also run web.
CONFIDENCE_THRESHOLD = 0.30
WEB_MAX_RESULTS = 4


# ── Search helpers ────────────────────────────────────────────────────────────

def _search_docs(question: str) -> tuple[str, bool]:
    """
    Returns (formatted_context, is_confident).
    Always returns whatever chunks exist, even low-scoring ones.
    is_confident=False means web search should also run.
    """
    if not document_store.has_documents():
        return "", False

    hits = document_store.search(question, k=4)
    if not hits:
        return "", False

    best_score = max(h["score"] for h in hits)
    is_confident = best_score >= CONFIDENCE_THRESHOLD

    lines = ["=== COMPANY DOCUMENT EXCERPTS ==="]
    for i, hit in enumerate(hits, 1):
        lines.append(
            f"[Doc {i}] {hit['source']} | page {hit['page']} | relevance {hit['score']:.2f}\n"
            f"{hit['content']}"
        )

    if not is_confident:
        lines.append(
            f"\n(Note: best relevance score was {best_score:.2f} — "
            "the above excerpts may be only partially relevant.)"
        )

    return "\n\n".join(lines), is_confident


def _search_web(question: str) -> str:
    """Returns formatted web snippets, or empty string on failure."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=WEB_MAX_RESULTS))

        if not results:
            return ""

        lines = ["=== WEB SEARCH RESULTS ==="]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Web {i}] {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"{r.get('body', '')}"
            )
        return "\n\n".join(lines)

    except Exception as e:
        logger.error(f"Web search error: {e}")
        return ""


# ── LLM synthesizer ───────────────────────────────────────────────────────────

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


SYNTHESIS_SYSTEM = """You are a helpful assistant. Answer the user's question using the context below when it is relevant.

Rules:
- If the context contains a clear answer, use it and cite the source (document name + page, or URL).
- If the context is partially relevant, use what you can and supplement with your own knowledge.
- If no context is provided or it is irrelevant, answer from your own knowledge and state that you are doing so.
- Never refuse to answer. Always give the most helpful response you can.
- Be concise and direct."""


def _synthesize(question: str, context: str) -> str:
    if context:
        prompt = f"{context}\n\n---\nQuestion: {question}"
    else:
        prompt = (
            f"No external context was found for this question. "
            f"Please answer from your own knowledge.\n\nQuestion: {question}"
        )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content=prompt),
    ]
    return _get_llm().invoke(messages).content


# ── Public interface ──────────────────────────────────────────────────────────

def query_rag_agent(question: str) -> Dict[str, Any]:
    """
    Always returns an answer. Sources used:
      "company_docs" — indexed PDFs/documents were consulted
      "web"          — DuckDuckGo was searched
      "llm_knowledge"— no external context found; LLM answered from training data
    """
    sources_used: List[str] = []
    context_parts: List[str] = []
    steps = 0

    try:
        # Step 1: company docs (always include if anything exists)
        doc_context, confident = _search_docs(question)
        steps += 1
        if doc_context:
            context_parts.append(doc_context)
            sources_used.append("company_docs")

        # Step 2: web search when docs are absent or low-confidence
        if not confident:
            web_context = _search_web(question)
            steps += 1
            if web_context:
                context_parts.append(web_context)
                sources_used.append("web")

        # Step 3: LLM synthesis — always runs, even with no context
        combined_context = "\n\n".join(context_parts)
        if not combined_context:
            sources_used.append("llm_knowledge")

        answer = _synthesize(question, combined_context)

        return {"answer": answer, "sources_used": sources_used, "steps": steps}

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources_used": [],
            "steps": steps,
        }
