# exo.models.anthropic

Anthropic LLM provider implementation. Wraps the `anthropic` SDK to implement `ModelProvider.complete()` and `ModelProvider.stream()` with normalized response types.

## Module Path

```python
from exo.models.anthropic import AnthropicProvider
```

## Auto-Registration

On import, `AnthropicProvider` is registered in `model_registry` under the name `"anthropic"`:

```python
model_registry.register("anthropic", AnthropicProvider)
```

---

## AnthropicProvider

Wraps the `anthropic.AsyncAnthropic` client for message completions.

**Inherits:** `ModelProvider`

### Constructor

```python
AnthropicProvider(config: ModelConfig)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ModelConfig` | *(required)* | Provider connection configuration |

The constructor creates an `AsyncAnthropic` client using:

- `api_key` from `config.api_key` (falls back to `"dummy"` if `None`)
- `base_url` from `config.base_url`
- `max_retries` from `config.max_retries`
- `timeout` from `config.timeout`

### Methods

#### complete()

```python
async def complete(
    self,
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelResponse
```

Send a message completion request to Anthropic.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions (OpenAI format, auto-converted) |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override (default: 4096) |

**Returns:** `ModelResponse`

**Raises:** `ModelError` if the API call fails.

#### stream()

```python
async def stream(
    self,
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[StreamChunk]
```

Stream a message completion from Anthropic.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions (OpenAI format, auto-converted) |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override (default: 4096) |

**Yields:** `StreamChunk`

**Raises:** `ModelError` if the API call fails.

### Default Max Tokens

Anthropic requires an explicit `max_tokens` parameter. If not specified by the caller, the provider defaults to **4096**.

### Message Conversion

Exo messages are converted to Anthropic format with these transformations:

| Exo Type | Anthropic Format | Notes |
|---|---|---|
| `SystemMessage` | Extracted to `system=` kwarg | Multiple system messages are joined with newlines |
| `UserMessage` | `{"role": "user", "content": ...}` | Direct content mapping |
| `AssistantMessage` | `{"role": "assistant", "content": [...]}` | Content blocks: `text` + `tool_use` |
| `ToolResult` | `{"role": "user", "content": [tool_result_block]}` | Merged into preceding user message for alternation |

Consecutive `ToolResult` messages are merged into a single `user` message with multiple `tool_result` content blocks to maintain Anthropic's strict role alternation requirement.

### Tool Schema Conversion

Tools are provided in OpenAI format and automatically converted to Anthropic format:

```python
# OpenAI format (input)
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

# Anthropic format (output)
{"name": "...", "description": "...", "input_schema": {...}}
```

### Stop Reason Mapping

| Anthropic Value | Exo FinishReason |
|---|---|
| `"end_turn"` | `"stop"` |
| `"stop_sequence"` | `"stop"` |
| `"tool_use"` | `"tool_calls"` |
| `"max_tokens"` | `"length"` |
| `None` | `"stop"` |

### Streaming Events

The Anthropic streaming API uses a different event model than OpenAI. The provider handles these event types:

| Event Type | Behavior |
|---|---|
| `message_start` | Captures input token count |
| `content_block_start` | Emits initial `ToolCallDelta` for `tool_use` blocks |
| `content_block_delta` | Emits `StreamChunk` with text delta or tool argument delta |
| `message_delta` | Emits final `StreamChunk` with finish reason and usage |

### Thinking/Reasoning Support

For Claude models with extended thinking enabled, `thinking` content blocks are extracted into the `reasoning_content` field of `ModelResponse`.

### Example

```python
import asyncio
from exo.models import get_provider
from exo.types import SystemMessage, UserMessage

async def main():
    provider = get_provider(
        "anthropic:claude-sonnet-4-20250514",
        api_key="sk-ant-...",
    )

    response = await provider.complete(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        temperature=0.0,
        max_tokens=100,
    )

    print(response.content)  # "The capital of France is Paris."
    print(response.usage)    # Usage(input_tokens=25, output_tokens=8, total_tokens=33)

asyncio.run(main())
```
