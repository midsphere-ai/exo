# exo.models.provider

Abstract base class for LLM provider implementations and the factory function for building providers from model strings.

## Module Path

```python
from exo.models.provider import ModelProvider, model_registry, get_provider
```

---

## model_registry

```python
model_registry: Registry[type[ModelProvider]]
```

Global registry mapping provider name strings to `ModelProvider` subclasses. Built-in registrations:

| Name | Class |
|---|---|
| `"openai"` | `OpenAIProvider` |
| `"anthropic"` | `AnthropicProvider` |
| `"gemini"` | `GeminiProvider` |
| `"vertex"` | `VertexProvider` |

### Methods

| Method | Signature | Description |
|---|---|---|
| `register` | `(name: str, item: type[ModelProvider]) -> None` | Register a provider class |
| `get` | `(name: str) -> type[ModelProvider]` | Retrieve a provider class by name |
| `list_all` | `() -> list[str]` | List all registered provider names |
| `has` | `(name: str) -> bool` | Check if a name is registered |

### Example: Custom Provider Registration

```python
from exo.models.provider import ModelProvider, model_registry

class MyProvider(ModelProvider):
    async def complete(self, messages, **kwargs):
        ...
    async def stream(self, messages, **kwargs):
        ...

model_registry.register("my_provider", MyProvider)
```

---

## ModelProvider

Abstract base class for LLM providers. Subclasses implement `complete()` for single-shot calls and `stream()` for incremental token delivery.

### Constructor

```python
ModelProvider(config: ModelConfig)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ModelConfig` | *(required)* | Provider connection configuration |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `config` | `ModelConfig` | The provider's connection configuration |

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

Send a completion request and return the full response.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions for the provider |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

**Returns:** `ModelResponse`

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

Stream a completion request, yielding chunks incrementally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[Message]` | *(required)* | Conversation history |
| `tools` | `list[dict[str, Any]] \| None` | `None` | JSON-schema tool definitions for the provider |
| `temperature` | `float \| None` | `None` | Sampling temperature override |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens override |

**Yields:** `StreamChunk`

---

## get_provider()

Factory function that builds a `ModelProvider` from a `"provider:model_name"` string.

```python
def get_provider(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ModelProvider
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *(required)* | Model string, e.g. `"openai:gpt-4o"` |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `**kwargs` | `Any` | -- | Extra fields forwarded to `ModelConfig` |

**Returns:** A configured `ModelProvider` instance.

**Raises:** `ModelError` if the provider is not registered.

### How It Works

1. Parses `"provider:model_name"` using `parse_model_string()` (defaults provider to `"openai"` if no colon)
2. Looks up the provider class in `model_registry`
3. Constructs a `ModelConfig` with the parsed values and extra kwargs
4. Returns `ProviderClass(config)`

### Example

```python
from exo.models import get_provider

# OpenAI
provider = get_provider("openai:gpt-4o", api_key="sk-...")

# Anthropic
provider = get_provider("anthropic:claude-sonnet-4-20250514", api_key="sk-ant-...")

# Gemini (API key auth)
provider = get_provider("gemini:gemini-2.0-flash", api_key="AIza...")

# Vertex AI (GCP Application Default Credentials)
provider = get_provider("vertex:gemini-2.0-flash")

# Default provider (openai) when no colon
provider = get_provider("gpt-4o", api_key="sk-...")

# Custom base URL (e.g., local vLLM)
provider = get_provider(
    "openai:my-model",
    api_key="dummy",
    base_url="http://localhost:8000/v1",
)
```

---

## ModelConfig (from exo.config)

Configuration for an LLM provider connection.

```python
ModelConfig(
    provider: str = "openai",
    model_name: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int = 3,      # ge=0
    timeout: float = 30.0,     # gt=0
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"openai"` | Provider name |
| `model_name` | `str` | `"gpt-4o"` | Model identifier within the provider |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `max_retries` | `int` | `3` | Maximum retries on transient failures |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
