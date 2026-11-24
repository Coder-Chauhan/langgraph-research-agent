"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
}


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    max_steps: int
    checkpoint_db: str
    tavily_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                f"Choose one of: {', '.join(DEFAULT_MODELS)}"
            )
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
            max_steps=int(os.getenv("MAX_STEPS", "8")),
            checkpoint_db=os.getenv("CHECKPOINT_DB", "checkpoints.sqlite"),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        )
