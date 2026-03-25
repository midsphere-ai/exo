"""Create and use local tools — Exo quickstart.

Demonstrates three ways to define tools:

1. ``@tool`` decorator on a plain function (simplest)
2. ``FunctionTool`` wrapper (explicit, allows name/description overrides)
3. ``Tool`` ABC subclass (full control, custom execution logic)

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/local_tool.py
"""

from typing import Any, ClassVar

from exo import Agent, FunctionTool, Tool, run, tool

# --- 1. @tool decorator (simplest) ------------------------------------------


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: Name of the city to look up.
    """
    # In a real app, call a weather API here.
    return f"The weather in {city} is sunny, 22 °C."


# --- 2. FunctionTool wrapper (explicit) -------------------------------------


def calculate(expression: str) -> str:
    """Evaluate a simple math expression.

    Args:
        expression: A math expression like '2 + 3 * 4'.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return "Error: only basic arithmetic is supported."
    try:
        return str(eval(expression))
    except Exception as exc:
        return f"Error: {exc}"


calculator = FunctionTool(calculate, name="calculator")


# --- 3. Tool ABC subclass (full control) ------------------------------------


class GreetTool(Tool):
    """A tool that greets a user by name."""

    name = "greet"
    description = "Greet someone by name."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person to greet."},
        },
        "required": ["name"],
    }

    async def execute(self, **kwargs: Any) -> str:
        person = kwargs.get("name", "World")
        return f"Hello, {person}! Welcome to Exo."


# --- Agent with all three tools ----------------------------------------------

agent = Agent(
    name="toolbox-bot",
    model="openai:gpt-4o-mini",
    instructions=(
        "You have three tools: get_weather, calculator, and greet. "
        "Use the appropriate tool to answer the user's question."
    ),
    tools=[get_weather, calculator, GreetTool()],
)


if __name__ == "__main__":
    result = run.sync(agent, "Greet Alice, then tell me the weather in Paris.")
    print(result.output)
