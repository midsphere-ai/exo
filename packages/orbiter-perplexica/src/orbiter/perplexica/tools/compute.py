"""Computation tools — safe math evaluation and Wolfram Alpha queries.

Provides a sandboxed math expression evaluator and a Wolfram Alpha
Short Answers API wrapper.  Set ``WOLFRAM_APP_ID`` to enable live
Wolfram Alpha queries.

Usage:
    from examples.advanced.perplexica.tools.compute import calculate, wolfram_query
"""

from __future__ import annotations

import ast
import math
import os
from urllib.parse import quote_plus

from orbiter import tool

# Allowed function names and their implementations.
_SAFE_FUNCTIONS: dict[str, object] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "abs": abs,
    "round": round,
}

# Allowed constants.
_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

# AST node types that are permitted in expressions.
_SAFE_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    # Binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
    # Unary operators
    ast.UAdd,
    ast.USub,
)


def _validate_ast(node: ast.AST) -> None:
    """Walk the AST and reject any unsafe nodes.

    Args:
        node: AST node to validate.

    Raises:
        ValueError: If an unsafe node type is encountered.
    """
    if not isinstance(node, _SAFE_NODES):
        raise ValueError(
            f"Unsafe expression element: {type(node).__name__}"
        )
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _safe_eval(expression: str) -> float:
    """Parse and evaluate a math expression safely using the AST module.

    Args:
        expression: Mathematical expression string.

    Returns:
        The computed result as a float.

    Raises:
        ValueError: If the expression contains unsafe elements.
    """
    tree = ast.parse(expression, mode="eval")
    _validate_ast(tree)

    # Build a restricted namespace with only safe functions and constants.
    namespace: dict[str, object] = {}
    namespace.update(_SAFE_FUNCTIONS)
    namespace.update(_SAFE_CONSTANTS)

    code = compile(tree, "<expression>", "eval")
    result = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307
    return result


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic (+, -, *, /, **, //), math functions
    (sqrt, sin, cos, tan, log, abs, round), and constants (pi, e).

    Args:
        expression: Mathematical expression to evaluate
            (e.g. "sqrt(144) + 2 ** 3").
    """
    try:
        result = _safe_eval(expression)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
        return f"Error evaluating '{expression}': {exc}"
    except Exception as exc:
        return f"Unexpected error evaluating '{expression}': {exc}"

    # Format the result cleanly.
    if isinstance(result, float) and result.is_integer():
        return f"{expression} = {int(result)}"
    return f"{expression} = {result}"


@tool
async def wolfram_query(query: str) -> str:
    """Query Wolfram Alpha Short Answers API for computational results.

    Requires the ``WOLFRAM_APP_ID`` environment variable to be set.
    Returns a concise textual answer.

    Args:
        query: Natural language or computational query for Wolfram Alpha.
    """
    import urllib.request

    app_id = os.environ.get("WOLFRAM_APP_ID", "")
    if not app_id:
        return (
            f"[Wolfram Alpha stub] Result for '{query}' "
            "(set WOLFRAM_APP_ID for real results). "
            "Example: Try using the calculate tool for math expressions."
        )

    url = (
        "https://api.wolframalpha.com/v1/result"
        f"?appid={app_id}"
        f"&i={quote_plus(query)}"
    )
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Wolfram Alpha error for '{query}': {exc}"
