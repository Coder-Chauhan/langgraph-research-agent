"""The LangGraph workflow.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                   \\--(no tool calls / step limit hit)--> END

The graph is built explicitly with StateGraph (rather than the prebuilt
create_react_agent) so every node and edge is visible and easy to extend.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from research_agent.state import AgentState

SYSTEM_PROMPT = """You are a careful research assistant.

Guidelines:
- Use tools to verify facts, do arithmetic and fetch current information. Do not guess.
- Break multi-part questions into steps and call tools as needed.
- When you used a source, cite its URL in the final answer.
- If tools fail or return nothing useful, say so plainly rather than inventing an answer.
- Keep final answers concise and well structured.

Today's date is {today}."""

STEP_LIMIT_MESSAGE = (
    "I reached the maximum number of reasoning steps for this request. "
    "Here is what I could determine so far; try asking a narrower question."
)


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int = 8,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before_tools: bool = False,
):
    """Compile the agent graph.

    Args:
        llm: Any LangChain chat model that supports tool calling.
        tools: Tools the agent may call.
        max_steps: Maximum LLM calls per user turn before forcing a stop.
        checkpointer: Persists state per thread_id, enabling multi-turn memory.
        interrupt_before_tools: Pause before running tools (human approval).
    """
    llm_with_tools = llm.bind_tools(list(tools))

    def agent(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"messages": [AIMessage(content=STEP_LIMIT_MESSAGE)], "steps": steps}

        system = SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))
        response = llm_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response], "steps": steps + 1}

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    # handle_tool_errors=True turns tool exceptions into ToolMessages so the
    # LLM can see the error and recover instead of crashing the run.
    graph.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"] if interrupt_before_tools else None,
    )
