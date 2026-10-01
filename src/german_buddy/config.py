"""Central, env-driven settings for the German Learning Buddy.

Same pattern as the fitness-coach project: everything configurable lives
here, everything else imports from here instead of reading os.environ
directly.
"""
# src/german_buddy/config.py
# src/german_buddy/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """
    Read a setting from (in order):
    1. Streamlit secrets (cloud)
    2. Environment variables (.env locally)
    3. Default value
    """
    # 1. Try Streamlit secrets
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    # 2. Try environment variables
    return os.getenv(key, default)


class Settings:
    llm_provider: str = _get("LLM_PROVIDER", "deepseek")
    deepseek_api_key: str = _get("DEEPSEEK_API_KEY", "")
    deepseek_model: str = _get("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = _get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()