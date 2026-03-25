# exo.models.openai

OpenAI LLM provider implementation. Wraps the `openai` SDK to implement `ModelProvider.complete()` and `ModelProvider.stream()` with normalized response types.

## Module Path

```python
from exo.models.openai import OpenAIProvider
```

## Auto-Registration

On import, `OpenAIProvider` is registered in `model_registry` under the name `"openai"`:

```python
model_registry.register("openai", OpenAIProvider)
```

---

## OpenAIProvider

Wraps the `openai.AsyncOpenAI` client for chat completions.

**Inherits:** `ModelProvider`

### Constructor

```python
OpenAIProvider(config: ModelConfig)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ModelConfig` | *(required)* | Provider connection configuration |

The constructor creates an `AsyncOpenAI` client using:

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

Send a chat completion request to OpenAI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

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

Stream a chat completion from OpenAI. Automatically sets `stream=True` and `stream_options={"include_usage": True}` to get token usage on the final chunk.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

**Yields:** `StreamChunk`

**Raises:** `ModelError` if the API call fails.

### Message Conversion

Exo messages are converted to OpenAI format internally:

| Exo Type | OpenAI Role | Notes |
|---|---|---|
| `SystemMessage` | `"system"` | Direct content mapping |
| `UserMessage` | `"user"` | Direct content mapping |
| `AssistantMessage` | `"assistant"` | Includes `tool_calls` array if present |
| `ToolResult` | `"tool"` | Maps `tool_call_id`, uses `error` if present |

### Finish Reason Mapping

| OpenAI Value | Exo FinishReason |
|---|---|
| `"stop"` | `"stop"` |
| `"tool_calls"` | `"tool_calls"` |
| `"length"` | `"length"` |
| `"content_filter"` | `"content_filter"` |
| `None` | `"stop"` |

### Reasoning Models

For o1/o3 reasoning models, the `reasoning_content` field in `ModelResponse` is populated from the response's `model_extra["reasoning_content"]` if present.

### Example

```python
import asyncio
from exo.models import get_provider
from exo.types import SystemMessage, UserMessage

async def main():
    provider = get_provider("openai:gpt-4o", api_key="sk-...")

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

### Compatible Services

Since `OpenAIProvider` wraps the standard OpenAI SDK, it works with any OpenAI-compatible API by setting `base_url`:

- **OpenAI** (default)
- **Azure OpenAI** (set `base_url` to your Azure endpoint)
- **vLLM** (`base_url="http://localhost:8000/v1"`)
- **Ollama** (`base_url="http://localhost:11434/v1"`)
- **LiteLLM proxy**
