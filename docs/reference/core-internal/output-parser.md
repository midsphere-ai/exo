# exo._internal.output_parser

Parse LLM responses into agent-level output types. The output parser bridges the model layer (`ModelResponse`) to the agent layer (`AgentOutput`, `ActionModel`). It extracts text and tool calls, parses JSON-encoded tool arguments, and optionally validates structured output against a Pydantic model.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.output_parser`

```python
from exo._internal.output_parser import (
    OutputParseError,
    parse_response,
    parse_tool_arguments,
    parse_structured_output,
)
```

---

## OutputParseError

```python
class OutputParseError(ExoError)
```

Raised when LLM output cannot be parsed as expected (invalid JSON, failed validation, non-object arguments). Inherits from `ExoError`.

---

## parse_response()

```python
def parse_response(
    *,
    content: str,
    tool_calls: list[ToolCall],
    usage: Usage,
) -> AgentOutput
```

Convert raw model fields into an `AgentOutput`. This is a lightweight mapping -- it copies text, tool calls, and usage without transforming the data. Use `parse_tool_arguments()` and `parse_structured_output()` for deeper parsing.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str` | *(required)* | Text output from the model. |
| `tool_calls` | `list[ToolCall]` | *(required)* | Tool invocations from the model. |
| `usage` | `Usage` | *(required)* | Token usage statistics. |

### Returns

An `AgentOutput` with the same data.

### Example

```python
from exo._internal.output_parser import parse_response
from exo.types import ToolCall, Usage

output = parse_response(
    content="Here is the answer.",
    tool_calls=[],
    usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
)
print(output.text)  # "Here is the answer."
```

---

## parse_tool_arguments()

```python
def parse_tool_arguments(tool_calls: list[ToolCall]) -> list[ActionModel]
```

Parse JSON-encoded arguments from tool calls into `ActionModel` objects. Each `ToolCall.arguments` is a JSON string. This function decodes it into a dict and wraps the result in an `ActionModel` ready for tool execution.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tool_calls` | `list[ToolCall]` | *(required)* | Tool calls with JSON-encoded arguments. |

### Returns

A list of `ActionModel` objects with parsed arguments.

### Raises

`OutputParseError` -- if any tool call has invalid JSON arguments or if the parsed JSON is not an object.

### Example

```python
from exo._internal.output_parser import parse_tool_arguments
from exo.types import ToolCall

tool_calls = [
    ToolCall(id="call_1", name="search", arguments='{"query": "AI safety"}'),
    ToolCall(id="call_2", name="calculate", arguments='{"x": 42}'),
]

actions = parse_tool_arguments(tool_calls)
print(actions[0].tool_name)    # "search"
print(actions[0].arguments)    # {"query": "AI safety"}
print(actions[1].tool_call_id) # "call_2"
```

### Empty Arguments

Empty argument strings are treated as no arguments:

```python
tc = ToolCall(id="call_1", name="get_time", arguments="")
actions = parse_tool_arguments([tc])
print(actions[0].arguments)  # {}
```

---

## parse_structured_output()

```python
def parse_structured_output(
    text: str,
    output_type: type[_T],
) -> _T
```

Validate LLM text output against a Pydantic model. Attempts to parse `text` as JSON and validate it against `output_type`.

### Type Parameters

- `_T` -- A Pydantic `BaseModel` subclass.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Raw text from the LLM (expected to be JSON). |
| `output_type` | `type[_T]` | *(required)* | The Pydantic model class to validate against. |

### Returns

A validated instance of `output_type`.

### Raises

- `OutputParseError` -- if `text` is not valid JSON.
- `OutputParseError` -- if the parsed JSON fails Pydantic validation.

### Example

```python
from pydantic import BaseModel
from exo._internal.output_parser import parse_structured_output

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str

text = '{"city": "Tokyo", "temperature": 25.5, "conditions": "sunny"}'
report = parse_structured_output(text, WeatherReport)
print(report.city)         # "Tokyo"
print(report.temperature)  # 25.5
```
