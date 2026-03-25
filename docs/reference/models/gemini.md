# exo.models.gemini

Google Gemini LLM provider implementation. Wraps the `google-genai` SDK to implement `ModelProvider.complete()` and `ModelProvider.stream()` with normalized response types.

## Module Path

```python
from exo.models.gemini import GeminiProvider
```

## Auto-Registration

On import, `GeminiProvider` is registered in `model_registry` under the name `"gemini"`:

```python
model_registry.register("gemini", GeminiProvider)
```

---

## GeminiProvider

Wraps the `google.genai.Client` for Gemini API completions using API key authentication.

**Inherits:** `ModelProvider`

### Constructor

```python
GeminiProvider(config: ModelConfig)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ModelConfig` | *(required)* | Provider connection configuration |

The constructor creates a `genai.Client` using:

- `api_key` from `config.api_key` (falls back to `"dummy"` if `None`)

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

Send a completion request to Google Gemini.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions (OpenAI format, auto-converted) |
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

Stream a completion from Google Gemini. Uses `generate_content_stream()` internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions (OpenAI format, auto-converted) |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

**Yields:** `StreamChunk`

**Raises:** `ModelError` if the API call fails.

### Message Conversion

Exo messages are converted to Google API format internally:

| Exo Type | Google Format | Notes |
|---|---|---|
| `SystemMessage` | Extracted to `system_instruction` config | Multiple system messages are joined with newlines |
| `UserMessage` | `{"role": "user", "parts": [{"text": ...}]}` | Direct content mapping |
| `AssistantMessage` | `{"role": "model", "parts": [...]}` | Text as `{"text": ...}`, tool calls as `{"function_call": ...}` |
| `ToolResult` | `{"role": "user", "parts": [{"function_response": ...}]}` | Uses `tool_name` for the function name |

### Tool Schema Conversion

Tools are provided in OpenAI format and automatically converted to Google's `function_declarations` format:

```python
# OpenAI format (input)
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

# Google format (output)
{"function_declarations": [{"name": "...", "description": "...", "parameters": {...}}]}
```

Multiple tools are grouped into a single `function_declarations` list.

### Finish Reason Mapping

| Google Value | Exo FinishReason |
|---|---|
| `"STOP"` | `"stop"` |
| `"MAX_TOKENS"` | `"length"` |
| `"SAFETY"` | `"content_filter"` |
| `"RECITATION"` | `"content_filter"` |
| `"BLOCKLIST"` | `"content_filter"` |
| `"MALFORMED_FUNCTION_CALL"` | `"stop"` |
| `"OTHER"` | `"stop"` |
| `None` | `"stop"` |

### Tool Call IDs

Google's API does not always return tool call IDs. When `function_call.id` is not present, the provider generates a synthetic ID using the pattern `call_{index}` where `index` is the part's position in the response.

### Example

```python
import asyncio
from exo.models import get_provider
from exo.types import SystemMessage, UserMessage

async def main():
    provider = get_provider(
        "gemini:gemini-2.0-flash",
        api_key="AIza...",
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

### Streaming Example

```python
import asyncio
from exo.models import get_provider
from exo.types import UserMessage

async def main():
    provider = get_provider("gemini:gemini-2.0-flash", api_key="AIza...")

    async for chunk in provider.stream(
        [UserMessage(content="Tell me a story")],
    ):
        if chunk.delta:
            print(chunk.delta, end="")
        if chunk.finish_reason:
            print(f"\n[Done: {chunk.finish_reason}]")

asyncio.run(main())
```

### Supported Models

Any model available through the Gemini API, including:

- `gemini-2.0-flash` -- fast, general-purpose
- `gemini-2.0-flash-lite` -- lightweight variant
- `gemini-2.5-pro` -- most capable, with thinking
- `gemini-2.5-flash` -- fast with thinking

### Authentication

The Gemini provider uses **API key authentication**. Pass your Google AI API key via:

```python
provider = get_provider("gemini:gemini-2.0-flash", api_key="AIza...")
```

Or set the `GOOGLE_API_KEY` environment variable.

For GCP-authenticated access (Vertex AI), use the [Vertex provider](vertex.md) instead.
