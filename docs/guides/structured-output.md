# Structured Output

Structured output constrains an agent's response to conform to a Pydantic model. Instead of receiving free-form text, you get a validated, typed Python object. This is useful for extracting structured data, enforcing response formats, and building reliable pipelines.

## Basic Usage

Define a Pydantic model and pass it as the `output_type` parameter:

```python
from pydantic import BaseModel
from exo.agent import Agent
from exo.runner import run

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    recommend: bool

agent = Agent(
    name="reviewer",
    model="openai:gpt-4o",
    instructions="You review movies. Always respond with valid JSON matching the requested format.",
    output_type=MovieReview,
)

result = await run(agent, "Review the movie 'Inception'")
```

## How It Works

When `output_type` is set on an agent:

1. The LLM is prompted to produce its response as JSON
2. The agent's final text output is parsed as JSON
3. The JSON is validated against the Pydantic model using `model_validate()`
4. If validation succeeds, the structured data is available

The output parsing is handled by `parse_structured_output()` in the output parser module.

## The Output Parser

The `parse_structured_output()` function performs two-step validation:

```python
from exo._internal.output_parser import parse_structured_output, OutputParseError

class WeatherData(BaseModel):
    city: str
    temperature: float
    condition: str

# Successful parse
data = parse_structured_output(
    '{"city": "Tokyo", "temperature": 22.5, "condition": "sunny"}',
    WeatherData,
)
print(data.city)         # "Tokyo"
print(data.temperature)  # 22.5

# Invalid JSON
try:
    parse_structured_output("not json", WeatherData)
except OutputParseError as e:
    print(e)  # "Structured output is not valid JSON: ..."

# Valid JSON, invalid schema
try:
    parse_structured_output('{"city": "Tokyo"}', WeatherData)
except OutputParseError as e:
    print(e)  # "Structured output failed WeatherData validation: ..."
```

### Function Signature

```python
def parse_structured_output(
    text: str,
    output_type: type[T],   # T bound to BaseModel
) -> T: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Raw text from the LLM (expected to be JSON) |
| `output_type` | `type[BaseModel]` | The Pydantic model class to validate against |

**Returns:** A validated instance of `output_type`.

**Raises:** `OutputParseError` if the text is not valid JSON or fails Pydantic validation.

## Nested Models

Complex nested structures work naturally with Pydantic:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    addresses: list[Address]

agent = Agent(
    name="extractor",
    instructions="Extract person information as JSON.",
    output_type=Person,
)
```

## Optional Fields

Use `Optional` or default values for fields that may not always be present:

```python
class SearchResult(BaseModel):
    query: str
    results: list[str]
    total_count: int
    next_page: str | None = None  # optional field
```

## Combining with Tools

Structured output works alongside tools. The agent uses tools during its loop, and the final text response is validated against the output type:

```python
from exo.tool import tool

@tool
def lookup_price(ticker: str) -> str:
    """Look up a stock price."""
    return f"{ticker}: $150.00"

class StockAnalysis(BaseModel):
    ticker: str
    price: float
    recommendation: str

agent = Agent(
    name="analyst",
    instructions="Analyze the stock and respond with JSON.",
    tools=[lookup_price],
    output_type=StockAnalysis,
)

result = await run(agent, "Analyze AAPL")
```

## Agent.describe() with output_type

The `describe()` method includes the output type name:

```python
agent = Agent(name="bot", output_type=MovieReview)
desc = agent.describe()
print(desc["output_type"])  # "MovieReview"

# Without output_type
agent2 = Agent(name="bot2")
print(agent2.describe()["output_type"])  # None
```

## Error Handling

Structured output errors are raised as `OutputParseError` (a subclass of `ExoError`):

```python
from exo._internal.output_parser import OutputParseError

try:
    result = parse_structured_output(llm_text, MyModel)
except OutputParseError as e:
    # Two possible causes:
    # 1. "Structured output is not valid JSON: ..."
    # 2. "Structured output failed MyModel validation: ..."
    print(f"Parse error: {e}")
```

## Best Practices

1. **Be explicit in instructions:** Tell the LLM to respond with JSON matching your schema. Include field descriptions.

2. **Use field descriptions:** Pydantic's `Field(description=...)` helps the LLM understand what each field should contain.

3. **Keep models simple:** Simpler models have higher success rates. Avoid deeply nested optional structures.

4. **Handle parse failures:** Not every LLM response will be valid JSON. Build retry logic or fallback handling.

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `parse_structured_output()` | `exo._internal.output_parser` | Validate LLM text against a Pydantic model |
| `parse_response()` | `exo._internal.output_parser` | Convert raw model fields to `AgentOutput` |
| `parse_tool_arguments()` | `exo._internal.output_parser` | Parse JSON tool call arguments to `ActionModel` |
| `OutputParseError` | `exo._internal.output_parser` | Parsing/validation error |
| `AgentOutput` | `exo.types` | Output from a single LLM call |
| `ActionModel` | `exo.types` | Parsed tool action ready for execution |
