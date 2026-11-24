from research_agent.tools import calculator, current_datetime, get_tools, web_search


def test_calculator_basic_arithmetic():
    assert calculator.invoke({"expression": "(12.5 * 4) / 2 + 2**3"}) == "33"


def test_calculator_rejects_code():
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert result.startswith("Error")


def test_calculator_division_by_zero():
    assert calculator.invoke({"expression": "1/0"}) == "Error: division by zero"


def test_calculator_blocks_huge_exponent():
    assert calculator.invoke({"expression": "9**999999"}).startswith("Error")


def test_current_datetime_is_iso():
    assert "T" in current_datetime.invoke({})


def test_web_search_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "not configured" in web_search.invoke({"query": "anything"})


def test_get_tools_toggles_web_search():
    assert "web_search" not in [t.name for t in get_tools(include_web_search=False)]
    assert "web_search" in [t.name for t in get_tools(include_web_search=True)]
