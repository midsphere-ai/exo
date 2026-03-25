# exo.models.vertex

Google Vertex AI LLM provider implementation. Wraps the `google-genai` SDK with Vertex AI (GCP Application Default Credentials) authentication to implement `ModelProvider.complete()` and `ModelProvider.stream()` with normalized response types.

## Module Path

```python
from exo.models.vertex import VertexProvider
```

## Auto-Registration

On import, `VertexProvider` is registered in `model_registry` under the name `"vertex"`:

```python
model_registry.register("vertex", VertexProvider)
```

---

## VertexProvider

Wraps the `google.genai.Client` with Vertex AI authentication for GCP-hosted model access. Supports Gemini models and Model Garden models available through Vertex AI.

**Inherits:** `ModelProvider`

### Constructor

```python
VertexProvider(config: ModelConfig)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ModelConfig` | *(required)* | Provider connection configuration |

The constructor creates a `genai.Client` with `vertexai=True` using:

- `project` from the `GOOGLE_CLOUD_PROJECT` environment variable (defaults to `""`)
- `location` from the `GOOGLE_CLOUD_LOCATION` environment variable (defaults to `"us-central1"`)

Authentication is handled via **Application Default Credentials (ADC)** -- no API key is needed.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `""` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | `"us-central1"` | GCP region for Vertex AI |

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

Send a completion request to Vertex AI.

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

Stream a completion from Vertex AI. Uses `generate_content_stream()` internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions (OpenAI format, auto-converted) |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

**Yields:** `StreamChunk`

**Raises:** `ModelError` if the API call fails.

### Message Conversion

Identical to the [Gemini provider](gemini.md) -- messages are converted to Google API format:

| Exo Type | Google Format | Notes |
|---|---|---|
| `SystemMessage` | Extracted to `system_instruction` config | Multiple system messages are joined with newlines |
| `UserMessage` | `{"role": "user", "parts": [{"text": ...}]}` | Direct content mapping |
| `AssistantMessage` | `{"role": "model", "parts": [...]}` | Text as `{"text": ...}`, tool calls as `{"function_call": ...}` |
| `ToolResult` | `{"role": "user", "parts": [{"function_response": ...}]}` | Uses `tool_name` for the function name |

### Tool Schema Conversion

Same as the [Gemini provider](gemini.md) -- tools are converted from OpenAI format to Google's `function_declarations` format.

### Finish Reason Mapping

Same as the [Gemini provider](gemini.md).

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

### Error Prefix

Vertex AI errors use the `"vertex:"` prefix in `ModelError` messages (e.g., `"vertex:gemini-2.0-flash"`), distinguishing them from Gemini API errors which use the `"gemini:"` prefix.

### Example

```python
import asyncio
from exo.models import get_provider
from exo.types import SystemMessage, UserMessage

async def main():
    # No API key needed -- uses Application Default Credentials
    provider = get_provider("vertex:gemini-2.0-flash")

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

### Authentication Setup

Vertex AI uses **Application Default Credentials (ADC)**. Set up before using the provider:

```bash
# Option 1: gcloud CLI (development)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Option 2: Service account (production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="my-project-id"
```

### Supported Models

Any model available through Vertex AI, including:

- **Gemini models:** `gemini-2.0-flash`, `gemini-2.5-pro`, `gemini-2.5-flash`
- **Model Garden:** Third-party and open models hosted on Vertex AI

### Gemini API vs Vertex AI

| Feature | Gemini (`"gemini:"`) | Vertex AI (`"vertex:"`) |
|---|---|---|
| **Authentication** | API key | GCP Application Default Credentials |
| **Provider string** | `"gemini:model-name"` | `"vertex:model-name"` |
| **Model access** | Google AI models only | Google AI + Model Garden |
| **Billing** | Google AI billing | GCP project billing |
| **VPC / Private** | No | Yes (VPC Service Controls) |
| **Region control** | No | Yes (`GOOGLE_CLOUD_LOCATION`) |
| **Best for** | Prototyping, personal projects | Production, enterprise |

For API-key-based access, use the [Gemini provider](gemini.md) instead.
