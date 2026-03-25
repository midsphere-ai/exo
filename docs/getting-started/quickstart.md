# Quickstart

Build and run your first Exo agent in under 5 minutes.

## Prerequisites

- Exo installed (`pip install exo` or `pip install exo-core exo-models`)
- `OPENAI_API_KEY` environment variable set

## Hello, Weather Agent

Create a file called `weather_agent.py`:

```python
from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: The city to get weather for.
    """
    # In a real app, you would call a weather API here
    return f"Sunny, 22 C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
    tools=[get_weather],
)

result = run.sync(agent, "What's the weather in Tokyo?")
print(result.output)
```

Run it:

```bash
python weather_agent.py
```

You should see output like:

```
The weather in Tokyo is currently sunny and 22°C.
```

### What Just Happened?

1. **`@tool`** turned `get_weather` into a `FunctionTool` with an auto-generated JSON schema derived from the function signature and docstring.
2. **`Agent(...)`** created an agent with the name `"weather-bot"`, connected to the `openai:gpt-4o-mini` model, with system instructions and one tool.
3. **`run.sync(agent, "...")`** executed the agent synchronously. Under the hood, it:
   - Built the message list (system instruction + user message)
   - Called the LLM, which decided to invoke `get_weather(city="Tokyo")`
   - Executed the tool and fed the result back to the LLM
   - The LLM produced a final text response
4. **`result.output`** is the final text from the agent. The full `RunResult` also contains `messages` (the complete conversation history), `usage` (token counts), and `steps` (how many LLM calls were made).

## Streaming

For real-time output, use `run.stream()`:

```python
import asyncio
from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: The city to get weather for.
    """
    return f"Sunny, 22 C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)


async def main():
    async for event in run.stream(agent, "What's the weather in Tokyo?"):
        if event.type == "text":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[Calling tool: {event.tool_name}]")
    print()  # final newline


asyncio.run(main())
```

The stream yields two event types:

- **`TextEvent`** -- a chunk of text from the LLM (`event.type == "text"`, `event.text` contains the delta)
- **`ToolCallEvent`** -- notification that a tool is being called (`event.type == "tool_call"`, `event.tool_name` contains the tool name)

The streaming loop handles the full tool cycle automatically: when the LLM requests a tool call, the tool is executed and the LLM is re-streamed with the result.

## Multi-Turn Conversations

To continue a conversation, pass the previous result's messages back:

```python
from exo import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: The city to get weather for.
    """
    return f"Sunny, 22 C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

# First turn
result = run.sync(agent, "What's the weather in Tokyo?")
print(f"Turn 1: {result.output}")

# Second turn -- pass the message history from the first turn
result = run.sync(agent, "How about London?", messages=result.messages)
print(f"Turn 2: {result.output}")

# Third turn -- the agent remembers the whole conversation
result = run.sync(agent, "Which city was warmer?", messages=result.messages)
print(f"Turn 3: {result.output}")
```

The `messages` parameter accepts any sequence of `Message` objects. The `result.messages` list contains the full conversation history from that run, including system messages, user messages, assistant responses, tool calls, and tool results.

## Inspecting the RunResult

The `RunResult` object gives you full visibility into what happened:

```python
result = run.sync(agent, "What's the weather in Tokyo?")

# Final text output
print(result.output)

# Token usage
print(f"Input tokens:  {result.usage.input_tokens}")
print(f"Output tokens: {result.usage.output_tokens}")
print(f"Total tokens:  {result.usage.total_tokens}")

# Number of LLM call steps
print(f"Steps: {result.steps}")

# Full message history
for msg in result.messages:
    print(f"  [{msg.role}] {getattr(msg, 'content', '')[:80]}")
```

## Using a Different Model

Change the model by updating the `model` string:

```python
# OpenAI models
agent = Agent(name="bot", model="openai:gpt-4o")
agent = Agent(name="bot", model="openai:gpt-4o-mini")

# Anthropic models (requires ANTHROPIC_API_KEY)
agent = Agent(name="bot", model="anthropic:claude-sonnet-4-20250514")

# If no provider prefix is given, OpenAI is assumed
agent = Agent(name="bot", model="gpt-4o")  # same as "openai:gpt-4o"
```

## Next Steps

- **[Core Concepts](concepts.md)** -- Understand all the building blocks in detail
- **[Your First Agent](first-agent.md)** -- Build a multi-agent system step by step
