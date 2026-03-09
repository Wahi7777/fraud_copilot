from __future__ import annotations

import os
from functools import lru_cache
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger("fraud_api")


def get_llm_runtime_info() -> dict:
    """
    Runtime metadata used for debug visibility in API responses/logs.
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "provider": "openai",
        "model": model,
        "has_api_key": has_key,
    }


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Shared LLM provider for CrewAI agents.

    Uses ChatOpenAI with model gpt-4o-mini and temperature 0, configured via
    OPENAI_API_KEY loaded from .env using python-dotenv.
    """
    load_dotenv()
    info = get_llm_runtime_info()
    api_key = os.getenv("OPENAI_API_KEY")
    logger.warning(
        "[LLM INIT] provider=%s model=%s api_key_present=%s",
        info["provider"],
        info["model"],
        info["has_api_key"],
    )
    # Do not fail hard if the key is missing; this allows tests to run in
    # environments without external connectivity. Actual CrewAI runs will
    # require a valid key.
    return ChatOpenAI(
        model=info["model"],
        temperature=0,
        api_key=api_key,
    )

