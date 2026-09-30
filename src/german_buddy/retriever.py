"""Wraps the Chroma collection built by ingest.py as callable retrieval
functions — one per source (grammar vs. vocab) — and exposes each as a
LangChain @tool so an agent's LLM can decide to call it.
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

from german_buddy.config import VECTORSTORE_DIR, settings

_client = None
_collection = None


def _get_collection():
    """Lazy singleton: only opens the Chroma store the first time a
    retrieval tool is actually called, not at import time."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        _collection = _client.get_collection(
            name=settings.collection_name, embedding_function=embedder
        )
    return _collection


def _search(query: str, source: str | None, k: int) -> str:
    collection = _get_collection()
    where = {"source": source} if source else None
    results = collection.query(query_texts=[query], n_results=k, where=where)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "No relevant passages found."

    formatted = []
    for text, meta in zip(docs, metas):
        page = meta.get("page", "?")
        src = meta.get("source", "?")
        formatted.append(f"[{src}, p.{page}] {text}")
    return "\n\n".join(formatted)


@tool
def search_grammar(query: str) -> str:
    """Search the German grammar reference book for passages relevant to
    the query. Use this whenever you need to explain or verify a grammar
    rule (cases, articles, verb tenses, word order, etc.) instead of
    relying on what you already 'know', which may be wrong or imprecise."""
    return _search(query, source="grammar_book", k=settings.top_k)


@tool
def search_vocab(query: str) -> str:
    """Search the German vocabulary reference book for passages relevant
    to the query. Use this for word meanings, usage examples, or
    vocabulary organized by topic/level."""
    return _search(query, source="vocab_book", k=settings.top_k)
