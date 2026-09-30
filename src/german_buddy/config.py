"""Central, env-driven settings for the German Learning Buddy.

Same pattern as the fitness-coach project: everything configurable lives
here, everything else imports from here instead of reading os.environ
directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env if python-dotenv is installed (optional but recommended).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# The two PDF source books. Rename your files to match, or change these.
GRAMMAR_PDF = DATA_RAW_DIR / "Basic german.pdf"
VOCAB_PDF = DATA_RAW_DIR / "Goethe-Zertifikat_A2_Wortliste (1).pdf"


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    collection_name: str = os.getenv("COLLECTION_NAME", "german_buddy")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))


settings = Settings()
