# exo.tool

Tool system: abstract base class, decorator, schema generation, and execution.

**Module:** `exo.tool`

```python
from exo.tool import Tool, FunctionTool, tool, ToolError
```

---

## ToolError

```python
class ToolError(ExoError)
```

Raised when a tool execution fails. Inherits from `ExoError`.

Use this in custom tool implementations to signal tool-specific errors that the agent can recover from.

```python
from exo.tool import ToolError

raise ToolError("Failed to fetch data from API")
```

---

## Tool (ABC)

```python
class Tool(ABC)
```

Abstract base class for all tools. Subclasses must implement `execute()`. The `name`, `description`, and `parameters` attributes describe the tool for LLM function calling.

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Unique tool name. |
| `description` | `str` | Human-readable description. |
| `parameters` | `dict[str, Any]` | JSON Schema dict describing accepted arguments. |

### Methods

#### execute() *(abstract)*

```python
@abstractmethod
async def execute(self, **kwargs: Any) -> str | dict[str, Any]
```

Execute the tool with the given keyword arguments.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `**kwargs` | `Any` | | Tool-specific arguments. |

**Returns:** A string or dict result.

#### to_schema()

```python
def to_schema(self) -> dict[str, Any]
```

Return the tool schema in OpenAI function-calling format.

**Returns:** A dict with the structure:

```python
{
    "type": "function",
    "function": {
        "name": "...",
        "description": "...",
        "parameters": { ... }  # JSON Schema
    }
}
```

### Example: Custom Tool Subclass

```python
from exo.tool import Tool

class DatabaseQueryTool(Tool):
    def __init__(self):
        self.name = "query_db"
        self.description = "Execute a read-only SQL query."
        self.parameters = {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute."},
            },
            "required": ["sql"],
        }

    async def execute(self, **kwargs) -> str:
        sql = kwargs["sql"]
        # Execute query...
        return "result rows"
```

---

## FunctionTool

```python
class FunctionTool(Tool)
```

A tool that wraps a plain sync or async function. Sync functions are automatically wrapped via `asyncio.to_thread()` so all execution goes through the single async `execute()` interface.

### Constructor

```python
def __init__(
    self,
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `fn` | `Callable[..., Any]` | *(required)* | The function to wrap. |
| `name` | `str \| None` | `None` | Override the tool name. Defaults to `fn.__name__`. |
| `description` | `str \| None` | `None` | Override the description. Defaults to first line of docstring. |

### Methods

#### execute()

```python
async def execute(self, **kwargs: Any) -> str | dict[str, Any]
```

Execute the wrapped function.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `**kwargs` | `Any` | | Arguments forwarded to the wrapped function. |

**Returns:** The function's return value.

**Raises:** `ToolError` if the wrapped function raises any exception (except `ToolError` itself, which re-raises directly).

### Schema Generation

`FunctionTool` automatically generates a JSON Schema from the wrapped function's signature, type hints, and Google-style `Args:` docstring section. The schema generation:

- Inspects function parameters (excluding `self`, `cls`, `*args`, `**kwargs`)
- Converts Python type annotations to JSON Schema types:
  - `str` -> `"string"`
  - `int` -> `"integer"`
  - `float` -> `"number"`
  - `bool` -> `"boolean"`
  - `list[X]` -> `{"type": "array", "items": ...}`
  - `dict[K, V]` -> `{"type": "object"}`
  - `X | None` / `Optional[X]` -> unwraps to the type of `X`
  - Unrecognized types default to `"string"`
- Extracts parameter descriptions from Google-style docstrings
- Marks parameters without defaults as `required`

### Example

```python
from exo.tool import FunctionTool

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b

tool = FunctionTool(calculate_sum)
print(tool.name)        # "calculate_sum"
print(tool.description) # "Calculate the sum of two numbers."
print(tool.parameters)
# {
#     "type": "object",
#     "properties": {
#         "a": {"type": "integer", "description": "The first number."},
#         "b": {"type": "integer", "description": "The second number."},
#     },
#     "required": ["a", "b"],
# }
```

---

## @tool Decorator

```python
@overload
def tool(fn: Callable[..., Any], /) -> FunctionTool: ...

@overload
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...
```

Decorator to turn a function into a `FunctionTool`. Supports three forms:

### Form 1: Bare decorator

```python
@tool
def my_tool(x: str) -> str:
    """Do something."""
    return x
```

### Form 2: Empty call

```python
@tool()
def my_tool(x: str) -> str:
    """Do something."""
    return x
```

### Form 3: With options

```python
@tool(name="custom_name", description="Custom description")
def my_tool(x: str) -> str:
    """Original docstring (overridden by description param)."""
    return x
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `fn` | `Callable[..., Any] \| None` | `None` | The function (when used as bare `@tool`). |
| `name` | `str \| None` | `None` | Override tool name. |
| `description` | `str \| None` | `None` | Override tool description. |

### Returns

A `FunctionTool` instance, or a decorator that produces one.

### Full Example

```python
from exo.tool import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to look up.
    """
    return f"Sunny in {city}"

# get_weather is now a FunctionTool instance
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # "Get the current weather for a city."

# Async tools are supported too
@tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL.

    Args:
        url: The URL to fetch.
    """
    return f"Data from {url}"

# Use with an Agent
from exo.agent import Agent
agent = Agent(name="bot", tools=[get_weather, fetch_data])
```
