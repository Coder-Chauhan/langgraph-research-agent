"""Graph state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages is a reducer: node outputs are appended (or merged by id)
    # instead of overwriting the list.
    messages: Annotated[list[AnyMessage], add_messages]
    # Number of LLM calls made in the current user turn. Reset by the CLI /
    # caller on each new user message and used to stop runaway tool loops.
    steps: int
