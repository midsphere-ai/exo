# Processors

Processors are event-driven hooks that run at specific points in the agent lifecycle. They can modify context state, transform payloads, or trigger side effects like summarization and result offloading. Processors are managed by the `ProcessorPipeline`, which fires all registered processors for a given event.

## Basic Usage

```python
from exo.context import ContextProcessor, ProcessorPipeline, Context, ContextConfig

# Create a pipeline
pipeline = ProcessorPipeline()

# Register a processor
class LoggingProcessor(ContextProcessor):
    event = "pre_llm_call"
    name = "logger"

    async def process(self, ctx: Context, payload: dict) -> dict:
        print(f"About to call LLM with {len(payload.get('messages', []))} messages")
        return payload

pipeline.register(LoggingProcessor())

# Fire the event
ctx = Context(task_id="t1", config=ContextConfig())
result = await pipeline.fire("pre_llm_call", ctx, {"messages": [...]})
```

## Event Points

The pipeline supports four lifecycle events:

| Event | When It Fires | Typical Use |
|-------|--------------|-------------|
| `pre_llm_call` | Before sending messages to the LLM | Summarize history, inject context, trim tokens |
| `post_llm_call` | After receiving the LLM response | Extract entities, update state, log response |
| `pre_tool_call` | Before executing a tool | Validate arguments, add metadata |
| `post_tool_call` | After a tool returns its result | Offload large results, update workspace |

## Writing a Processor

Subclass `ContextProcessor` and implement the `process` method:

```python
from exo.context import ContextProcessor

class EntityExtractor(ContextProcessor):
    event = "post_llm_call"
    name = "entity_extractor"

    async def process(self, ctx, payload: dict) -> dict:
        """Extract entities from the LLM response and store in state."""
        response_text = payload.get("response", "")

        # Extract entities (simplified)
        entities = self._extract(response_text)
        existing = ctx.state.get("entities", [])
        ctx.state.set("entities", existing + entities)

        return payload  # pass payload through unchanged

    def _extract(self, text: str) -> list[str]:
        # ... entity extraction logic ...
        return []
```

Key rules:

- **`event`** -- string matching one of the four lifecycle events.
- **`name`** -- unique identifier for the processor.
- **`process(ctx, payload)`** -- async method that receives the context and event payload. Must return the (possibly modified) payload dict.

## Built-in Processors

### SummarizeProcessor

Fires on `pre_llm_call`. Checks if the conversation history exceeds the `summary_threshold` from `ContextConfig` and, if so, summarizes older messages to reduce token usage:

```python
from exo.context import SummarizeProcessor

processor = SummarizeProcessor()
# event = "pre_llm_call"
# name = "summarize"
```

The processor reads `config.summary_threshold` from the context to decide when to trigger summarization.

### ToolResultOffloader

Fires on `post_tool_call`. When a tool result exceeds `max_size` characters, it stores the full result in the workspace and replaces the message content with a reference:

```python
from exo.context import ToolResultOffloader

processor = ToolResultOffloader(max_size=5000)
# event = "post_tool_call"
# name = "tool_result_offloader"
```

This prevents large tool outputs from consuming excessive tokens in subsequent LLM calls.

## Pipeline Management

```python
pipeline = ProcessorPipeline()

# Register processors
pipeline.register(SummarizeProcessor())
pipeline.register(ToolResultOffloader(max_size=5000))

# Check what's registered
pipeline.has_processors("pre_llm_call")   # True
pipeline.has_processors("pre_tool_call")  # False

# List processors for an event
for name in pipeline.list_processors("pre_llm_call"):
    print(name)  # "summarize"

# Unregister by name
pipeline.unregister("summarize")
```

## Firing Events

When firing an event, the pipeline runs all registered processors for that event in registration order. Each processor receives the payload returned by the previous one, forming a chain:

```python
# Payload flows through the chain:
# original_payload -> Processor1 -> modified_payload -> Processor2 -> final_payload
result = await pipeline.fire("post_tool_call", ctx, {
    "tool_name": "search",
    "result": "very long text...",
})
```

## Advanced Patterns

### Conditional Processing

Skip processing based on context state:

```python
class ConditionalProcessor(ContextProcessor):
    event = "pre_llm_call"
    name = "conditional"

    async def process(self, ctx, payload: dict) -> dict:
        if ctx.config.mode == "pilot":
            # Skip summarization in pilot mode
            return payload
        # ... do work ...
        return payload
```

### Multiple Processors per Event

Register multiple processors for the same event -- they chain in registration order:

```python
pipeline = ProcessorPipeline()
pipeline.register(EntityExtractor())       # runs first
pipeline.register(SummarizeProcessor())    # runs second

# Both fire on pre_llm_call; payload flows through both
```

### Token-Aware Processors

Use the `TokenTracker` to make decisions based on token budgets:

```python
class TokenBudgetProcessor(ContextProcessor):
    event = "pre_llm_call"
    name = "token_budget"

    async def process(self, ctx, payload: dict) -> dict:
        usage = ctx.token_usage.total_usage()
        if usage.total_tokens > 50000:
            # Aggressively trim context
            payload["trim_history"] = True
        return payload
```

### Workspace Integration

Store processed artifacts in the workspace:

```python
class ResultArchiver(ContextProcessor):
    event = "post_tool_call"
    name = "archiver"

    def __init__(self, workspace):
        self._workspace = workspace

    async def process(self, ctx, payload: dict) -> dict:
        tool_name = payload.get("tool_name", "unknown")
        result = payload.get("result", "")
        if len(result) > 1000:
            self._workspace.write(
                f"tool_results/{tool_name}.txt",
                result,
            )
            payload["result"] = f"[Archived to workspace: tool_results/{tool_name}.txt]"
        return payload
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `ContextProcessor` | `exo.context` | ABC with `event`, `name`, `process(ctx, payload)` |
| `ProcessorPipeline` | `exo.context` | Manages and fires processors by event |
| `ProcessorPipeline.register(processor)` | | Add a processor to the pipeline |
| `ProcessorPipeline.unregister(name)` | | Remove a processor by name |
| `ProcessorPipeline.fire(event, ctx, payload)` | | Run all processors for an event |
| `ProcessorPipeline.has_processors(event)` | | Check if any processors are registered for an event |
| `ProcessorPipeline.list_processors(event)` | | List processor names for an event |
| `SummarizeProcessor` | `exo.context` | Built-in: summarizes history on `pre_llm_call` |
| `ToolResultOffloader` | `exo.context` | Built-in: offloads large tool results on `post_tool_call` |
