"""Interactive command-line chat with the research agent.

Usage:
    research-agent                      # new conversation
    research-agent --thread my-notes    # resume a saved conversation
    research-agent --approve            # confirm each tool call before it runs
    research-agent --graph              # print the graph as Mermaid and exit
"""

from __future__ import annotations

import argparse
import sys
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.config import Settings
from research_agent.graph import build_graph
from research_agent.llm import get_chat_model
from research_agent.tools import get_tools

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a message (content may be a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def truncate(text: str, limit: int = 300) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def print_updates(stream) -> None:
    """Render graph updates as they happen."""
    for update in stream:
        for node, payload in update.items():
            if node == "__interrupt__" or not payload:
                continue
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        print(f"{DIM}  -> {call['name']}({call['args']}){RESET}")
                    text = message_text(msg)
                    if text and not msg.tool_calls:
                        print(f"\n{BOLD}{CYAN}Agent:{RESET} {text}\n")
                elif isinstance(msg, ToolMessage):
                    print(f"{DIM}  <- {msg.name}: {truncate(message_text(msg))}{RESET}")


def handle_approval(graph, config: dict) -> None:
    """While the graph is paused before tools, ask the user to approve."""
    while graph.get_state(config).next == ("tools",):
        state = graph.get_state(config)
        pending = state.values["messages"][-1].tool_calls
        print(f"{YELLOW}The agent wants to run:{RESET}")
        for call in pending:
            print(f"{YELLOW}  - {call['name']}({call['args']}){RESET}")
        answer = input("Approve? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            print_updates(graph.stream(None, config, stream_mode="updates"))
        else:
            # Record a rejection as the tool result, then let the agent respond.
            rejections = [
                ToolMessage(
                    content="The user declined this tool call.",
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                for call in pending
            ]
            graph.update_state(config, {"messages": rejections}, as_node="tools")
            print_updates(graph.stream(None, config, stream_mode="updates"))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a LangGraph research agent.")
    parser.add_argument("--thread", help="Conversation id to create or resume.")
    parser.add_argument("--approve", action="store_true", help="Approve tool calls manually.")
    parser.add_argument("--graph", action="store_true", help="Print Mermaid diagram and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    tools = get_tools()

    try:
        llm = get_chat_model(settings.provider, settings.model)
    except Exception as exc:  # missing key, bad model name, etc.
        print(f"Could not create the chat model: {exc}", file=sys.stderr)
        return 1

    with SqliteSaver.from_conn_string(settings.checkpoint_db) as checkpointer:
        graph = build_graph(
            llm,
            tools,
            max_steps=settings.max_steps,
            checkpointer=checkpointer,
            interrupt_before_tools=args.approve,
        )

        if args.graph:
            print(graph.get_graph().draw_mermaid())
            return 0

        thread_id = args.thread or uuid.uuid4().hex[:8]
        config = {"configurable": {"thread_id": thread_id}}

        print(f"{BOLD}Research agent{RESET} ({settings.provider}:{settings.model})")
        print(f"Tools: {', '.join(t.name for t in tools)}")
        print(f"Thread: {thread_id}  (resume with --thread {thread_id})")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input(f"{BOLD}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            turn = {"messages": [HumanMessage(content=user_input)], "steps": 0}
            try:
                print_updates(graph.stream(turn, config, stream_mode="updates"))
                if args.approve:
                    handle_approval(graph, config)
            except Exception as exc:
                print(f"\nError while running the agent: {exc}\n", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
