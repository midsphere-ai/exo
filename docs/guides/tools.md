# Tools

Tools give agents the ability to interact with the outside world -- query APIs, read files, perform calculations, and more. Exo provides three ways to define tools, ranging from a simple decorator to a full abstract base class.

## Basic Usage: The @tool Decorator

The fastest way to create a tool is the `@tool` decorator. It turns a regular Python function into a `FunctionTool` with auto-generated JSON Schema:

```python
from exo.tool import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    return f"22 degrees {units} in {city}"
```

The decorator inspects the function's signature, type hints, and Google-style docstring to produce a complete JSON Schema automatically.

## Decorator Overloads

The `@tool` decorator supports three usage patterns:

```python
# 1. Bare decorator — uses function name and docstring
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# 2. Empty parentheses — same as bare
@tool()
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# 3. With options — override name and/or description
@tool(name="web_search", description="Search the internet for information")
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"
```

**Decorator Signatures:**

```python
# Overload 1: bare @tool
def tool(fn: Callable[..., Any], /) -> FunctionTool: ...

# Overload 2: @tool() or @tool(name=..., description=...)
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...
```

## FunctionTool

`FunctionTool` is the class that `@tool` creates. You can also construct it directly:

```python
from exo.tool import FunctionTool

def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: The math expression to evaluate.
    """
    return str(eval(expression))

calc_tool = FunctionTool(
    calculate,
    name="calculator",        # optional: override fn.__name__
    description="Do math",    # optional: override docstring
)
```

**Constructor:**

```python
class FunctionTool(Tool):
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,         # defaults to fn.__name__
        description: str | None = None,   # defaults to first docstring line
    ) -> None: ...
```

## Sync vs Async Functions

Both sync and async functions work with `@tool`. Sync functions are automatically wrapped via `asyncio.to_thread()` so they do not block the event loop:

```python
@tool
def sync_lookup(key: str) -> str:
    """A blocking I/O call -- runs in a thread automatically."""
    import time
    time.sleep(1)
    return f"Value for {key}"

@tool
async def async_lookup(key: str) -> str:
    """A native async call."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/{key}") as resp:
            return await resp.text()
```

## Schema Generation from Type Hints

Exo automatically converts Python type annotations to JSON Schema:

| Python Type | JSON Schema |
|-------------|-------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[X]` | `{"type": "array", "items": <X schema>}` |
| `dict[K, V]` | `{"type": "object"}` |
| `X \| None` / `Optional[X]` | Unwraps to schema of `X` |
| Unannotated / `Any` | `{"type": "string"}` |

Parameters without defaults are marked as `required`. Parameter descriptions come from the Google-style `Args:` section in the docstring.

**Example:**

```python
@tool
def create_user(name: str, age: int, tags: list[str] = []) -> str:
    """Create a new user.

    Args:
        name: The user's full name.
        age: The user's age in years.
        tags: Optional tags for the user.
    """
    return f"Created {name}"
```

Produces the schema:

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "The user's full name."},
    "age": {"type": "integer", "description": "The user's age in years."},
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional tags for the user."
    }
  },
  "required": ["name", "age"]
}
```

## Tool ABC Subclassing

For full control, subclass the `Tool` abstract base class:

```python
from exo.tool import Tool
from typing import Any

class DatabaseQuery(Tool):
    """Query a database with SQL."""

    def __init__(self, connection_string: str) -> None:
        self.name = "db_query"
        self.description = "Execute a SQL query against the database."
        self.parameters = {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return.",
                },
            },
            "required": ["sql"],
        }
        self._conn_str = connection_string

    async def execute(self, **kwargs: Any) -> str:
        sql = kwargs["sql"]
        limit = kwargs.get("limit", 100)
        # ... run query ...
        return f"Query returned {limit} rows"
```

**Tool ABC interface:**

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        """Execute the tool with the given keyword arguments."""

    def to_schema(self) -> dict[str, Any]:
        """Return the tool schema in OpenAI function-calling format."""
```

The `to_schema()` method returns:

```python
{
    "type": "function",
    "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters,
    },
}
```

## ToolError

Raise `ToolError` to signal a tool failure that the LLM should see and potentially recover from:

```python
from exo.tool import tool, ToolError

@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers.

    Args:
        a: The numerator.
        b: The denominator.
    """
    if b == 0:
        raise ToolError("Cannot divide by zero")
    return str(a / b)
```

When a tool raises any exception (including `ToolError`), the error message is sent back to the LLM as a `ToolResult` with the `error` field set. The LLM can then decide to retry with different arguments or explain the failure to the user. `FunctionTool` wraps non-`ToolError` exceptions in a `ToolError` automatically.

## Advanced: Parallel Tool Execution

When the LLM requests multiple tool calls in a single response, the agent executes them concurrently using `asyncio.TaskGroup`:

```python
# If the LLM calls get_weather("Tokyo") and get_weather("London")
# in the same response, both run concurrently
agent = Agent(
    name="multi_weather",
    tools=[get_weather],
    max_steps=5,
)
```

Each tool call runs independently -- if one fails, others still complete, and the errors are reported individually.

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Tool` | `exo.tool` | Abstract base class for tools |
| `FunctionTool` | `exo.tool` | Wraps a sync or async function as a tool |
| `tool` | `exo.tool` | Decorator to create a `FunctionTool` |
| `ToolError` | `exo.tool` | Error raised during tool execution |
| `ToolResult` | `exo.types` | Result of a tool execution (sent back to LLM) |
| `ToolCall` | `exo.types` | LLM request to invoke a tool |
