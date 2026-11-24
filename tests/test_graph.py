from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.graph import STEP_LIMIT_MESSAGE, build_graph
from research_agent.tools import calculator, current_datetime

from conftest import tool_call

TOOLS = [calculator, current_datetime]


def run(graph, text, config=None):
    return graph.invoke({"messages": [HumanMessage(content=text)], "steps": 0}, config)


def test_answers_directly_without_tools(scripted_model):
    llm = scripted_model(AIMessage(content="Hello!"))
    result = run(build_graph(llm, TOOLS), "hi")
    assert result["messages"][-1].content == "Hello!"
    assert result["steps"] == 1


def test_tool_loop_executes_tool_and_returns_answer(scripted_model):
    llm = scripted_model(
        tool_call("calculator", {"expression": "6*7"}),
        AIMessage(content="The answer is 42."),
    )
    result = run(build_graph(llm, TOOLS), "what is 6*7?")

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and tool_msgs[0].content == "42"
    assert result["messages"][-1].content == "The answer is 42."
    assert result["steps"] == 2


def test_step_limit_stops_runaway_loops(scripted_model):
    llm = scripted_model(
        *[tool_call("current_datetime", {}, f"c{i}") for i in range(10)]
    )
    result = run(build_graph(llm, TOOLS, max_steps=3), "loop forever")
    assert result["messages"][-1].content == STEP_LIMIT_MESSAGE
    assert result["steps"] == 3


def test_unknown_tool_error_is_reported_not_raised(scripted_model):
    llm = scripted_model(
        tool_call("does_not_exist", {}),
        AIMessage(content="Sorry, that tool failed."),
    )
    result = run(build_graph(llm, TOOLS), "use a bad tool")
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "error" in tool_msg.content.lower() or tool_msg.status == "error"
    assert result["messages"][-1].content == "Sorry, that tool failed."


def test_checkpointer_keeps_memory_across_turns(scripted_model):
    llm = scripted_model(AIMessage(content="Nice to meet you."), AIMessage(content="Bhanu."))
    graph = build_graph(llm, TOOLS, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    run(graph, "My name is Bhanu.", config)
    result = run(graph, "What is my name?", config)

    humans = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["My name is Bhanu.", "What is my name?"]
    assert len(result["messages"]) == 4


def test_interrupt_before_tools_pauses_for_approval(scripted_model):
    llm = scripted_model(
        tool_call("calculator", {"expression": "2+2"}),
        AIMessage(content="4"),
    )
    graph = build_graph(llm, TOOLS, checkpointer=InMemorySaver(), interrupt_before_tools=True)
    config = {"configurable": {"thread_id": "t2"}}

    run(graph, "2+2?", config)
    assert graph.get_state(config).next == ("tools",)

    result = graph.invoke(None, config)  # approve and resume
    assert result["messages"][-1].content == "4"
    assert graph.get_state(config).next == ()
