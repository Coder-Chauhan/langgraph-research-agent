"""Tools the agent can call.

Each tool is a plain function decorated with @tool. The docstring becomes the
description the LLM sees, so keep it specific about when to use the tool.
"""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from langchain_core.tools import BaseTool, tool

HTTP_TIMEOUT = 15.0
USER_AGENT = "research-agent/0.1 (LangGraph demo project)"

# ---------------------------------------------------------------------------
# Calculator: evaluates arithmetic via the AST, never via eval().
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12.5 * 4) / 3 + 2**8'.

    Supports + - * / // % ** and parentheses. Use this for any non-trivial math
    instead of computing in your head.
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError) as exc:
        return f"Error: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
@tool
def current_datetime() -> str:
    """Return the current date and time in UTC (ISO 8601).

    Use this whenever the answer depends on today's date.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Wikipedia (no API key required)
# ---------------------------------------------------------------------------
@tool
def wikipedia_summary(topic: str) -> str:
    """Look up a topic on English Wikipedia and return its summary paragraph.

    Best for stable background facts: people, places, concepts, history.
    Pass a concise page title like 'LangChain' or 'Transformer (deep learning)'.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers) as client:
            # Resolve the best-matching title first, then fetch its summary.
            search = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 1,
                    "format": "json",
                },
            )
            search.raise_for_status()
            titles = search.json()[1]
            if not titles:
                return f"No Wikipedia page found for '{topic}'."
            title = titles[0]
            resp = client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(title.replace(" ", "_"))
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        return f"Error contacting Wikipedia: {exc}"

    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"{data.get('title', title)}\n\n{data.get('extract', '')}\n\nSource: {url}"


# ---------------------------------------------------------------------------
# Web search via Tavily (requires TAVILY_API_KEY)
# ---------------------------------------------------------------------------
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web for current information (news, prices, recent events).

    Returns titles, URLs and short content snippets. Prefer this over
    wikipedia_summary when recency matters.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: web_search is not configured (TAVILY_API_KEY missing)."
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max(1, min(max_results, 10)),
                "search_depth": "basic",
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except httpx.HTTPError as exc:
        return f"Error contacting search API: {exc}"

    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}")
    return "\n\n".join(lines)


def get_tools(include_web_search: bool | None = None) -> list[BaseTool]:
    """Return the tool list. Web search is included only when a key is set."""
    if include_web_search is None:
        include_web_search = bool(os.getenv("TAVILY_API_KEY"))
    tools: list[BaseTool] = [calculator, current_datetime, wikipedia_summary]
    if include_web_search:
        tools.append(web_search)
    return tools
