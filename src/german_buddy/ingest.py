"""Data ingestion: grammar/vocab PDFs -> cleaned text chunks -> Chroma vector store.

Run directly to (re)build the vector store from scratch:

    python -m german_buddy.ingest

Sources handled:
  1. data/raw/grammar_book.pdf  — your German grammar reference
  2. data/raw/vocab_book.pdf    — your German vocabulary reference

Both are plain PDFs (no OCR) — pypdf's text extraction is good enough for
typeset text but will return "" for scanned/image-only pages, which we
skip rather than crash on.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
import pypdf
from chromadb.utils import embedding_functions
from tqdm import tqdm

from german_buddy.config import (
    DATA_PROCESSED_DIR,
    GRAMMAR_PDF,
    VECTORSTORE_DIR,
    VOCAB_PDF,
    settings,
)


@dataclass
class Document:
    """A single retrievable chunk plus its metadata."""

    text: str
    metadata: dict


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #

# Lines that are just a page number, or very short lines that repeat as
# running headers/footers, add noise rather than meaning to a chunk.
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")


def _clean_page_text(raw_text: str) -> str:
    """Strip obvious PDF-extraction noise: bare page-number lines and
    excess blank lines. Deliberately conservative — we'd rather leave some
    noise in than accidentally delete real content."""
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line and not _PAGE_NUMBER_RE.match(line)]
    return "\n".join(lines)


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, cleaned_text) for every page that has
    extractable text. Page numbers are 1-indexed, matching what a human
    would cite ("see page 42")."""
    if not path.exists():
        return []

    reader = pypdf.PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_page_text(raw_text)
        if cleaned.strip():
            pages.append((i, cleaned))
    return pages


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack paragraphs (or, failing that, lines) greedily up to chunk_size
    characters, carrying a small overlap tail so no chunk loses context at
    its boundary.

    PDF-extracted text often has no blank-line paragraph breaks the way
    Markdown does, so we fall back to splitting on single newlines when
    there are no double-newline paragraphs at all.
    """
    units = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(units) <= 1:
        units = [line.strip() for line in text.split("\n") if line.strip()]

    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(current) + len(unit) + 1 <= chunk_size:
            current = f"{current}\n{unit}" if current else unit
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n{unit}" if tail else unit
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_pdf_documents(path: Path, source_label: str) -> list[Document]:
    """Turn one PDF into a list of chunked, metadata-tagged Documents."""
    docs: list[Document] = []
    for page_number, page_text in extract_pdf_pages(path):
        for chunk in chunk_text(page_text, settings.chunk_size, settings.chunk_overlap):
            docs.append(
                Document(
                    text=chunk,
                    metadata={
                        "source": source_label,
                        "type": "knowledge",
                        "page": page_number,
                    },
                )
            )
    return docs


def load_all_documents() -> list[Document]:
    docs: list[Document] = []
    docs += load_pdf_documents(GRAMMAR_PDF, "grammar_book")
    docs += load_pdf_documents(VOCAB_PDF, "vocab_book")
    return docs


# --------------------------------------------------------------------------- #
# Vector store build
# --------------------------------------------------------------------------- #

def build_vectorstore(rebuild: bool = True) -> chromadb.Collection:
    if rebuild and VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    documents = load_all_documents()
    if not documents:
        raise RuntimeError(
            "No documents found to ingest. Put your PDFs at "
            f"{GRAMMAR_PDF} and {VOCAB_PDF} (or update config.py's paths) "
            "before running ingest."
        )

    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding + indexing"):
        batch = documents[i : i + batch_size]
        collection.add(
            ids=[f"doc_{i + j}" for j in range(len(batch))],
            documents=[d.text for d in batch],
            metadatas=[d.metadata for d in batch],
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED_DIR / "ingest_summary.json").write_text(
        json.dumps(
            {
                "total_documents": len(documents),
                "grammar_chunks": sum(1 for d in documents if d.metadata["source"] == "grammar_book"),
                "vocab_chunks": sum(1 for d in documents if d.metadata["source"] == "vocab_book"),
                "embedding_model": settings.embedding_model,
            },
            indent=2,
        )
    )

    print(f"Indexed {len(documents)} chunks into '{settings.collection_name}' at {VECTORSTORE_DIR}")
    return collection


if __name__ == "__main__":
    build_vectorstore(rebuild=True)
