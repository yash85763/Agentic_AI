"""
llm_config.py — Configurable LLM backend (Anthropic Claude or OpenAI GPT)

Usage:
    from llm_config import get_llm

    llm = get_llm()                          # reads LLM_PROVIDER from env
    llm = get_llm(provider="anthropic")      # explicit
    llm = get_llm(provider="openai")
"""

from __future__ import annotations

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

LLMProvider = Literal["anthropic", "openai"]

# Model defaults per provider
DEFAULTS = {
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "openai": {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
}


def get_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Returns a LangChain BaseChatModel. Provider is resolved in this order:
      1. Explicit `provider` argument
      2. LLM_PROVIDER environment variable
      3. Falls back to "anthropic"
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider not in ("anthropic", "openai"):
        raise ValueError(f"Unsupported provider: '{provider}'. Use 'anthropic' or 'openai'.")

    defaults = DEFAULTS[provider]
    resolved_model = model or os.getenv("LLM_MODEL", defaults["model"])
    resolved_temp = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", defaults["temperature"]))
    resolved_max = max_tokens or int(os.getenv("LLM_MAX_TOKENS", defaults["max_tokens"]))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=resolved_max,
            anthropic_api_key=api_key,
        )

    else:  # openai
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=resolved_max,
            openai_api_key=api_key,
        )
