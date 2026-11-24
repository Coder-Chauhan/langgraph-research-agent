"""Chat model factory."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def get_chat_model(provider: str, model: str, temperature: float = 0.0) -> BaseChatModel:
    """Create a chat model for the given provider.

    Imports are local so that only the provider you use needs to be installed
    and configured.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=2048)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")
