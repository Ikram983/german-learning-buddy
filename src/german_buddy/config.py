"""Central, env-driven settings for the German Learning Buddy.

Same pattern as the fitness-coach project: everything configurable lives
here, everything else imports from here instead of reading os.environ
directly.
"""
# src/german_buddy/config.py
from __future__ import annotations

import os
from pydantic_settings import BaseSettings


def _load_streamlit_secrets() -> dict:
    """Load secrets from Streamlit Cloud if available."""
    try:
        import streamlit as st
        # st.secrets behaves like a dict
        return dict(st.secrets)
    except Exception:
        return {}


class Settings(BaseSettings):
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 1. Try .env first (local)
settings = Settings()

# 2. Override with Streamlit secrets if present (cloud)
_secrets = _load_streamlit_secrets()
if _secrets:
    if "DEEPSEEK_API_KEY" in _secrets:
        settings.deepseek_api_key = _secrets["DEEPSEEK_API_KEY"]
    if "DEEPSEEK_MODEL" in _secrets:
        settings.deepseek_model = _secrets["DEEPSEEK_MODEL"]
    if "LLM_PROVIDER" in _secrets:
        settings.llm_provider = _secrets["LLM_PROVIDER"]