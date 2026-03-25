# Model Providers

Model providers are the bridge between Exo's agent system and LLM APIs. Exo ships with providers for OpenAI, Anthropic, Google Gemini, and Google Vertex AI, and supports custom providers via the `ModelProvider` abstract base class.

## Basic Usage

Providers are typically resolved automatically from the agent's model string:

```python
from exo.agent import Agent
from exo.runner import run

# Provider auto-resolved from "openai:gpt-4o"
agent = Agent(name="bot", model="openai:gpt-4o")
result = await run(agent, "Hello!")

# Or with Anthropic
agent = Agent(name="bot", model="anthropic:claude-sonnet-4-20250514")
result = await run(agent, "Hello!")

# Or with Google Gemini (API key)
agent = Agent(name="bot", model="gemini:gemini-2.0-flash")
result = await run(agent, "Hello!")

# Or with Google Vertex AI (GCP ADC)
agent = Agent(name="bot", model="vertex:gemini-2.0-flash")
result = await run(agent, "Hello!")
```

You can also create providers explicitly:

```python
from exo.models.provider import get_provider

provider = get_provider("openai:gpt-4o", api_key="sk-...")
result = await run(agent, "Hello!", provider=provider)
```

## Model String Format

Model strings follow the `"provider:model_name"` convention:

```
openai:gpt-4o
openai:gpt-4o-mini
anthropic:claude-sonnet-4-20250514
anthropic:claude-opus-4-20250514
gemini:gemini-2.0-flash
gemini:gemini-2.5-pro
vertex:gemini-2.0-flash
vertex:gemini-2.5-pro
```

If no colon is present, the provider defaults to `"openai"`:

```python
from exo.config import parse_model_string

parse_model_string("gpt-4o")                    # ("openai", "gpt-4o")
parse_model_string("openai:gpt-4o")             # ("openai", "gpt-4o")
parse_model_string("anthropic:claude-sonnet-4-20250514")  # ("anthropic", "claude-sonnet-4-20250514")
parse_model_string("gemini:gemini-2.0-flash")   # ("gemini", "gemini-2.0-flash")
parse_model_string("vertex:gemini-2.0-flash")   # ("vertex", "gemini-2.0-flash")
```

## get_provider()

The factory function for creating provider instances:

```python
def get_provider(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelProvider: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model string, e.g. `"openai:gpt-4o"` |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `**kwargs` | `Any` | -- | Extra fields forwarded to `ModelConfig` |

```python
from exo.models.provider import get_provider

# Basic usage
provider = get_provider("openai:gpt-4o")

# With explicit API key
provider = get_provider("openai:gpt-4o", api_key="sk-...")

# With custom base URL (for proxies or compatible APIs)
provider = get_provider("openai:gpt-4o", base_url="https://my-proxy.example.com/v1")
```

Raises `ModelError` if the provider name is not registered.

## ModelConfig

Configuration for an LLM provider connection:

```python
class ModelConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 3          # min: 0
    timeout: float = 30.0         # in seconds, must be > 0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"openai"` | Provider name |
| `model_name` | `str` | `"gpt-4o"` | Model identifier |
| `api_key` | `str \| None` | `None` | API key |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `max_retries` | `int` | `3` | Maximum retries for transient failures |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

## ModelProvider ABC

The abstract base class that all providers implement:

```python
class ModelProvider(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

### complete()

Sends a completion request and returns the full response:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[Message]` | *required* | Conversation history |
| `tools` | `list[dict] \| None` | `None` | JSON-schema tool definitions |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

Returns a `ModelResponse`.

### stream()

Streams a completion request, yielding chunks incrementally:

Same parameters as `complete()`. Yields `StreamChunk` objects.

## ModelResponse

Returned by `complete()`:

```python
class ModelResponse(BaseModel):
    id: str = ""                                    # provider correlation ID
    model: str = ""                                 # which model produced this
    content: str = ""                               # text output
    tool_calls: list[ToolCall] = []                 # tool invocations
    usage: Usage = Usage()                          # token usage
    finish_reason: FinishReason = "stop"            # why the model stopped
    reasoning_content: str = ""                     # chain-of-thought (o1/o3, Claude thinking)
```

### FinishReason

```python
FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]
```

| Value | Meaning | OpenAI | Anthropic | Gemini / Vertex |
|-------|---------|--------|-----------|-----------------|
| `"stop"` | Natural completion | `"stop"` | `"end_turn"`, `"stop_sequence"` | `"STOP"`, `"MALFORMED_FUNCTION_CALL"`, `"OTHER"` |
| `"tool_calls"` | Model wants to call tools | `"tool_calls"` | `"tool_use"` | -- (inferred from parts) |
| `"length"` | Hit max tokens | `"length"` | `"max_tokens"` | `"MAX_TOKENS"` |
| `"content_filter"` | Content filtered | `"content_filter"` | -- | `"SAFETY"`, `"RECITATION"`, `"BLOCKLIST"` |

## StreamChunk

Yielded by `stream()`:

```python
class StreamChunk(BaseModel):
    delta: str = ""                                 # incremental text
    tool_call_deltas: list[ToolCallDelta] = []      # incremental tool call fragments
    finish_reason: FinishReason | None = None       # non-None on final chunk
    usage: Usage = Usage()                          # typically only on final chunk
```

### ToolCallDelta

```python
class ToolCallDelta(BaseModel):
    index: int = 0              # position in multi-tool-call responses
    id: str | None = None       # tool call ID (first chunk only)
    name: str | None = None     # tool name (first chunk only)
    arguments: str = ""         # incremental JSON fragment
```

## OpenAI Provider

The `OpenAIProvider` wraps the `openai.AsyncOpenAI` client:

```python
from exo.models.openai import OpenAIProvider
from exo.config import ModelConfig

config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="sk-...",
)
provider = OpenAIProvider(config)
```

The OpenAI provider:
- Converts Exo messages to OpenAI chat format (`role`, `content`, `tool_calls`)
- Converts tool results to `role: "tool"` messages
- Handles `tool_calls` in responses as `ToolCall` objects
- Extracts `reasoning_content` from o1/o3 models
- Maps OpenAI finish reasons to the normalized `FinishReason` type

Registered automatically as `"openai"` in the model registry.

## Anthropic Provider

The `AnthropicProvider` wraps the `anthropic.AsyncAnthropic` client:

```python
from exo.models.anthropic import AnthropicProvider
from exo.config import ModelConfig

config = ModelConfig(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    api_key="sk-ant-...",
)
provider = AnthropicProvider(config)
```

The Anthropic provider:
- Extracts system messages into the `system=` parameter (Anthropic's format)
- Converts tool results to `tool_result` content blocks within `user` messages
- Merges consecutive tool results into a single user message (required by Anthropic's strict alternation)
- Converts OpenAI-format tool schemas to Anthropic format (`input_schema`)
- Uses a default `max_tokens` of `4096` when none is specified
- Extracts `reasoning_content` from `thinking` blocks
- Maps Anthropic stop reasons to the normalized `FinishReason` type

Registered automatically as `"anthropic"` in the model registry.

## Gemini Provider

The `GeminiProvider` wraps the `google.genai.Client` with API key authentication:

```python
from exo.models.gemini import GeminiProvider
from exo.config import ModelConfig

config = ModelConfig(
    provider="gemini",
    model_name="gemini-2.0-flash",
    api_key="AIza...",
)
provider = GeminiProvider(config)
```

The Gemini provider:
- Extracts system messages into the `system_instruction` config parameter
- Converts Exo messages to Google `Content` format (`"user"` / `"model"` roles)
- Converts OpenAI-format tool schemas to Google `function_declarations` format
- Tool results are sent as `function_response` parts within `"user"` messages
- Generates synthetic tool call IDs (`call_0`, `call_1`, ...) when the API omits them
- Maps Google finish reasons to the normalized `FinishReason` type

Registered automatically as `"gemini"` in the model registry.

### Authentication

Pass your Google AI API key via `api_key`:

```python
provider = get_provider("gemini:gemini-2.0-flash", api_key="AIza...")
```

Or set the `GOOGLE_API_KEY` environment variable.

### Supported Models

- `gemini-2.0-flash` -- fast, general-purpose
- `gemini-2.0-flash-lite` -- lightweight variant
- `gemini-2.5-pro` -- most capable, with thinking
- `gemini-2.5-flash` -- fast with thinking

## Vertex AI Provider

The `VertexProvider` wraps the `google.genai.Client` with GCP Application Default Credentials (ADC) for Vertex AI:

```python
from exo.models.vertex import VertexProvider
from exo.config import ModelConfig

config = ModelConfig(
    provider="vertex",
    model_name="gemini-2.0-flash",
)
provider = VertexProvider(config)
```

The Vertex AI provider:
- Uses the same message/tool conversion as the Gemini provider
- Authenticates via GCP Application Default Credentials (no API key needed)
- Reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (default: `us-central1`) from environment
- Error messages use the `"vertex:"` prefix to distinguish from Gemini API errors
- Supports all Vertex AI models including Gemini and Model Garden models

Registered automatically as `"vertex"` in the model registry.

### Authentication

Vertex AI uses Application Default Credentials. Set up before using:

```bash
# Option 1: gcloud CLI (development)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Option 2: Service account (production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="my-project-id"
```

### Gemini API vs Vertex AI

| Feature | Gemini (`"gemini:"`) | Vertex AI (`"vertex:"`) |
|---|---|---|
| **Authentication** | API key | GCP Application Default Credentials |
| **Model access** | Google AI models only | Google AI + Model Garden |
| **Billing** | Google AI billing | GCP project billing |
| **Region control** | No | Yes (`GOOGLE_CLOUD_LOCATION`) |
| **Best for** | Prototyping, personal projects | Production, enterprise |

## Custom Provider Implementation

To add support for a new LLM provider:

```python
from exo.models.provider import ModelProvider, model_registry
from exo.models.types import ModelResponse, StreamChunk
from exo.config import ModelConfig
from exo.types import Message, ToolCall, Usage

class MyProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # Initialize your client
        self._client = MySDKClient(api_key=config.api_key)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        # Convert messages to your provider's format
        converted = self._convert_messages(messages)

        # Call your provider's API
        response = await self._client.chat(
            model=self.config.model_name,
            messages=converted,
            temperature=temperature,
        )

        # Return normalized ModelResponse
        return ModelResponse(
            id=response.id,
            model=self.config.model_name,
            content=response.text,
            tool_calls=[],  # parse tool calls if supported
            usage=Usage(
                input_tokens=response.usage.input,
                output_tokens=response.usage.output,
                total_tokens=response.usage.total,
            ),
            finish_reason="stop",
        )

    async def stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        # Implement streaming
        converted = self._convert_messages(messages)
        async for chunk in self._client.chat_stream(
            model=self.config.model_name,
            messages=converted,
        ):
            yield StreamChunk(delta=chunk.text)

    def _convert_messages(self, messages):
        # Convert Exo Message types to your format
        ...

# Register with the model registry
model_registry.register("myprovider", MyProvider)
```

Now use it with agents:

```python
agent = Agent(name="bot", model="myprovider:my-model-v1")
result = await run(agent, "Hello!")
```

## Model Registry

The global `model_registry` maps provider names to `ModelProvider` subclasses:

```python
from exo.models.provider import model_registry

# List registered providers
print(model_registry.list_all())  # ["openai", "anthropic", "gemini", "vertex"]

# Register a new provider
model_registry.register("custom", MyProvider)

# Get a provider class
cls = model_registry.get("openai")  # returns OpenAIProvider class
```

## ModelError

Provider errors are raised as `ModelError`:

```python
from exo.models.types import ModelError

try:
    response = await provider.complete(messages)
except ModelError as e:
    print(e.model)  # "openai:gpt-4o"
    print(e.code)   # "context_length", "rate_limit", etc.
```

`ModelError` has three attributes:
- `message` (str): Human-readable error description
- `model` (str): The model identifier that caused the error
- `code` (str): Error classification code (e.g., `"context_length"`, `"rate_limit"`)

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `ModelProvider` | `exo.models.provider` | Abstract base class for LLM providers |
| `model_registry` | `exo.models.provider` | Global provider registry |
| `get_provider()` | `exo.models.provider` | Factory to build a provider from a model string |
| `OpenAIProvider` | `exo.models.openai` | OpenAI provider implementation |
| `AnthropicProvider` | `exo.models.anthropic` | Anthropic provider implementation |
| `GeminiProvider` | `exo.models.gemini` | Google Gemini provider implementation |
| `VertexProvider` | `exo.models.vertex` | Google Vertex AI provider implementation |
| `ModelResponse` | `exo.models.types` | Response from `complete()` |
| `StreamChunk` | `exo.models.types` | Chunk from `stream()` |
| `ToolCallDelta` | `exo.models.types` | Incremental tool call fragment |
| `ModelError` | `exo.models.types` | Provider error |
| `FinishReason` | `exo.models.types` | Why the model stopped generating |
| `ModelConfig` | `exo.config` | Provider connection configuration |
