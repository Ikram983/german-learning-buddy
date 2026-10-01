# src/german_buddy/config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read from st.secrets first, then env vars, then default."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# PDF source files — adjust these paths to match your actual data folder
GRAMMAR_PDF = DATA_DIR / "grammar.pdf"
VOCAB_PDF = DATA_DIR / "vocab.pdf"


# ============================================================
# Settings
# ============================================================
class Settings:
    # LLM
    llm_provider: str = _get("LLM_PROVIDER", "deepseek")
    deepseek_api_key: str = _get("DEEPSEEK_API_KEY", "")
    deepseek_model: str = _get("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")

    # Retrieval / RAG
    top_k: int = 4
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "german_buddy"
    chunk_size: int = 500
    chunk_overlap: int = 50
settings = Settings()