"""Shared test fixtures: a scripted chat model that needs no API key."""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    """Returns pre-defined AIMessages in order and accepts bind_tools()."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def scripted_model():
    def _make(*responses: AIMessage) -> ScriptedChatModel:
        iterator: Iterator[AIMessage] = iter(responses)
        return ScriptedChatModel(messages=iterator)

    return _make
