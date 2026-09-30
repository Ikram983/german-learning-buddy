"""One place that knows how to construct the chat model, so every agent
imports `get_llm()` instead of each wiring up its own client.

Uses langchain_openai.ChatOpenAI pointed at DeepSeek's OpenAI-compatible
endpoint. Swap the base_url/model to use plain OpenAI instead — the rest
of the codebase doesn't care which provider it is.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from german_buddy.config import settings


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (check your .env)")
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (check your .env)")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
