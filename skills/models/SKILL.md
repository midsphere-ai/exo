---
name: exo:models
description: "Use when configuring Exo agent models and providers — model strings, provider selection, API keys, base URLs, ModelConfig, custom providers, get_provider(), model_registry, context windows, token counting, media generation tools, multimodal content, provider-specific options. Triggers on: model, provider, openai, anthropic, gemini, vertex, api_key, base_url, ModelConfig, get_provider, model_registry, MODEL_CONTEXT_WINDOWS, ModelResponse, StreamChunk, ModelError, custom provider, media tools, dalle, imagen, veo, context window, temperature, max_tokens, TokenCounter, count_tokens."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Models — Providers & Model Configuration

## When To Use This Skill

Use this skill when the developer needs to:
- Configure which LLM model and provider an agent uses
- Understand the `"provider:model"` string format
- Set API keys, base URLs, timeouts, or retry policies
- Register a custom provider
- Use provider-specific options (Vertex AI credentials, OpenAI proxies)
- Work with `ModelResponse`, `StreamChunk`, or `ModelError` types
- Add media generation tools (DALL-E, Imagen, Veo)
- Understand context window sizes and token limits
- Handle multimodal content across providers

## Decision Guide

1. **Which model string do I use?** → `"provider:model_name"` (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-20250514"`)
2. **No colon in model string?** → Defaults to OpenAI (e.g., `"gpt-4o"` → `"openai:gpt-4o"`)
3. **Need a provider instance directly?** → `get_provider("openai:gpt-4o", api_key=..., base_url=...)`
4. **Need to customize retries/timeouts?** → Pass `max_retries` and `timeout` kwargs to `get_provider()`
5. **Using Vertex AI?** → Set `GOOGLE_CLOUD_PROJECT` env var + either ADC or `google_service_account_base64`
6. **Building a custom provider?** → Subclass `ModelProvider`, implement `complete()` + `stream()`, register with `model_registry`
7. **Need image/video generation?** → Use `dalle_generate_image`, `imagen_generate_image`, or `veo_generate_video` tools
8. **Need context window size?** → Check `MODEL_CONTEXT_WINDOWS` dict
9. **Need to count tokens before sending?** → `count_tokens(text, model="openai:gpt-4o")` or `TokenCounter(model)` for bulk counting

## Reference

### Model String Format

Format: `"provider:model_name"`

```python
# Explicit provider
agent = Agent(name="bot", model="openai:gpt-4o")
agent = Agent(name="bot", model="anthropic:claude-sonnet-4-20250514")
agent = Agent(name="bot", model="gemini:gemini-2.0-flash")
agent = Agent(name="bot", model="vertex:gemini-2.5-pro")

# Shorthand — no colon defaults to OpenAI
agent = Agent(name="bot", model="gpt-4o")  # equivalent to "openai:gpt-4o"
```

**Parsing** (`exo.config.parse_model_string`):
```python
from exo.config import parse_model_string

parse_model_string("openai:gpt-4o")       # → ("openai", "gpt-4o")
parse_model_string("gpt-4o")              # → ("openai", "gpt-4o")
parse_model_string("anthropic:claude-sonnet-4-20250514")  # → ("anthropic", "claude-sonnet-4-20250514")
```

### Built-in Providers

| Provider String | Class | SDK | Env Var for API Key |
|----------------|-------|-----|---------------------|
| `"openai"` | `OpenAIProvider` | `openai` (AsyncOpenAI) | `OPENAI_API_KEY` |
| `"anthropic"` | `AnthropicProvider` | `anthropic` (AsyncAnthropic) | `ANTHROPIC_API_KEY` |
| `"gemini"` | `GeminiProvider` | `google-genai` | `GEMINI_API_KEY` |
| `"vertex"` | `VertexProvider` | `google-genai` (vertexai=True) | GCP ADC or service account |

### Agent Model Parameters

```python
from exo import Agent

agent = Agent(
    name="assistant",
    model="openai:gpt-4o",           # Model string (default: "openai:gpt-4o")
    temperature=0.7,                   # Sampling temperature 0.0-2.0 (default: 1.0)
    max_tokens=4096,                   # Max output tokens (default: None → provider default)
    planning_model="anthropic:claude-sonnet-4-20250514",  # Optional separate model for planning phase
)
```

The Agent does **not** instantiate a provider at init time — it stores the model string and lazily resolves the provider during `run()` via `get_provider()`.

### get_provider() — Direct Provider Instantiation

```python
from exo.models import get_provider

# Basic
provider = get_provider("openai:gpt-4o")

# With explicit API key and custom base URL (e.g., proxy)
provider = get_provider(
    "openai:gpt-4o",
    api_key="sk-...",
    base_url="https://my-proxy.com/v1",
)

# With retry/timeout customization
provider = get_provider(
    "anthropic:claude-sonnet-4-20250514",
    api_key="sk-ant-...",
    max_retries=5,
    timeout=60.0,
)

# Vertex AI with service account
provider = get_provider(
    "vertex:gemini-2.5-pro",
    google_project="my-gcp-project",
    google_location="us-central1",
    google_service_account_base64="eyJ0eXBlIjoi...",
)
```

**What happens internally:**
1. Parses model string → `(provider_name, model_name)`
2. Looks up provider class in `model_registry`
3. Looks up context window from `MODEL_CONTEXT_WINDOWS` (optional)
4. Creates `ModelConfig` with all provided parameters
5. Instantiates and returns the provider

**Raises:** `ModelError` if provider not registered (error message includes available providers).

### ModelConfig

```python
from exo.config import ModelConfig

config = ModelConfig(
    provider="openai",                     # Provider name
    model_name="gpt-4o",                  # Model identifier
    api_key="sk-...",                      # API key (None → SDK reads env var)
    base_url="https://proxy.com/v1",       # Custom base URL (None → provider default)
    max_retries=3,                         # Retries on transient failures (default: 3)
    timeout=30.0,                          # Request timeout seconds (default: 30.0)
    context_window_tokens=128000,          # Context window override (None → auto from registry)
)
```

`ModelConfig` has `extra = "allow"` — provider-specific kwargs are stored as attributes:
```python
config = ModelConfig(
    provider="vertex",
    model_name="gemini-2.5-pro",
    google_project="my-project",           # Vertex-specific
    google_location="europe-west1",        # Vertex-specific
    google_service_account_base64="...",   # Vertex-specific
)
```

### ModelProvider ABC

All providers implement this interface:

```python
from exo.models.provider import ModelProvider
from exo.models.types import ModelResponse, StreamChunk
from exo.types import Message

class ModelProvider:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

### Response Types

**ModelResponse** (from `complete()`):
```python
class ModelResponse(BaseModel):
    id: str = ""                                # Provider correlation ID
    model: str = ""                             # Model that produced this
    content: str = ""                           # Text output
    tool_calls: list[ToolCall] = []             # Tool invocations requested
    usage: Usage = Usage()                      # Token usage (input, output, total)
    finish_reason: FinishReason = "stop"        # "stop" | "tool_calls" | "length" | "content_filter"
    reasoning_content: str = ""                 # Chain-of-thought (o1/o3, Claude thinking)
```

**StreamChunk** (from `stream()`):
```python
class StreamChunk(BaseModel):
    delta: str = ""                             # Incremental text
    tool_call_deltas: list[ToolCallDelta] = []  # Incremental tool call fragments
    finish_reason: FinishReason | None = None   # Non-None only on final chunk
    usage: Usage = Usage()                      # Typically only on final chunk
```

**ToolCallDelta:**
```python
class ToolCallDelta(BaseModel):
    index: int = 0                  # Position in multi-tool-call responses
    id: str | None = None           # Tool call ID (first chunk only)
    name: str | None = None         # Tool name (first chunk only)
    arguments: str = ""             # Incremental JSON fragment
```

**FinishReason normalization** — all providers map to: `"stop" | "tool_calls" | "length" | "content_filter"`

### ModelError

```python
from exo.models.types import ModelError

# Raised by providers on failure
class ModelError(ExoError):
    def __init__(self, message: str, *, model: str = "", code: str = "") -> None:
        self.model = model     # e.g., "openai:gpt-4o"
        self.code = code       # e.g., "context_length", "rate_limit"
```

### Environment Variables

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI | API authentication (auto-read by SDK) |
| `ANTHROPIC_API_KEY` | Anthropic | API authentication (auto-read by SDK) |
| `GEMINI_API_KEY` | Gemini | API authentication (fallback if `api_key` not passed) |
| `GOOGLE_API_KEY` | Gemini media tools | Used by `imagen_generate_image` |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI | GCP project ID (required) |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI | GCP region (default: `"us-central1"`) |
| `GOOGLE_SERVICE_ACCOUNT_BASE64` | Vertex AI | Base64-encoded service account JSON |

### Context Windows

```python
from exo.models import MODEL_CONTEXT_WINDOWS

# Known models and their context window sizes (tokens):
# gpt-4o:                    128,000
# gpt-4o-mini:               128,000
# o1:                        200,000
# claude-sonnet-4-6:         200,000
# claude-opus-4-6:           200,000
# claude-haiku-4-5-20251001: 200,000
# gemini-2.0-flash:        1,048,576
# gemini-1.5-pro:          2,097,152
```

Used automatically by `get_provider()` to populate `ModelConfig.context_window_tokens`.

### Token Counting

For pre-call token estimation, use `TokenCounter` or the `count_tokens()` convenience function. These use tiktoken with provider-aware encoding selection.

```python
from exo import TokenCounter, count_tokens

# Quick count (caches counter per model string)
n = count_tokens("Hello, world!", model="anthropic:claude-sonnet-4-6")

# Reusable counter
counter = TokenCounter("openai:gpt-4o")
n = counter.count("Some text to count")

# Count chat messages (includes per-message overhead)
total = counter.count_messages([
    {"role": "user", "content": "Hello"},
])

# Token ↔ character conversion (encoding-aware ratios)
chars = counter.tokens_to_chars(4096)
tokens = counter.chars_to_tokens(10000)
```

**Accuracy:** Exact for OpenAI models (tiktoken is their tokenizer). ~95% for Anthropic, ~85-90% for Gemini/Vertex (best available local approximation).

**Post-call usage:** For actual tokens consumed after an LLM call, use `ModelResponse.usage` or `RunResult.usage` — these are provider-reported and always exact.

### model_registry — Provider Registration

```python
from exo.models.provider import model_registry

# List all registered providers
model_registry.list_all()  # ["openai", "anthropic", "gemini", "vertex"]

# Get a provider class
provider_cls = model_registry.get("openai")  # → OpenAIProvider

# Register a custom provider
model_registry.register("my_provider", MyProviderClass)
```

### Media Generation Tools

Pre-built `@tool` functions for image/video generation:

```python
from exo.models import dalle_generate_image, imagen_generate_image, veo_generate_video

agent = Agent(
    name="creative",
    tools=[dalle_generate_image, imagen_generate_image, veo_generate_video],
)
```

**dalle_generate_image(prompt, size="1024x1024", quality="standard", style="vivid")**
- Requires: `OPENAI_API_KEY`
- Returns: `list[ImageURLBlock]`

**imagen_generate_image(prompt, aspect_ratio="1:1", number_of_images=1)**
- Requires: `GOOGLE_API_KEY`
- Returns: `list[ImageDataBlock]` (base64 PNG)

**veo_generate_video(prompt, duration_seconds=5, aspect_ratio="16:9")**
- Requires: `GOOGLE_CLOUD_PROJECT` + GCP auth
- Returns: `list[VideoBlock]`

### Multimodal Content Support

Exo uses `ContentBlock` types for multimodal messages. Provider support varies:

| Content Type | OpenAI | Anthropic | Gemini/Vertex |
|-------------|--------|-----------|---------------|
| `TextBlock` | Yes | Yes | Yes |
| `ImageURLBlock` | Yes | Yes | Yes |
| `ImageDataBlock` | Yes | Yes | Yes |
| `AudioBlock` | Yes | No (warn) | Yes |
| `VideoBlock` | No (warn) | No (warn) | Yes |
| `DocumentBlock` | No (warn) | Yes | Yes |

Unsupported types log a warning and are skipped — they don't raise errors.

## Patterns

### Basic Agent with Specific Model

```python
from exo import Agent, run

agent = Agent(
    name="assistant",
    model="anthropic:claude-sonnet-4-20250514",
    temperature=0.7,
    instructions="You are a helpful assistant.",
)

result = await run(agent, "Hello!")
print(result.output)
```

### Direct Provider Usage (No Agent)

```python
from exo.models import get_provider
from exo.types import SystemMessage, UserMessage

provider = get_provider("openai:gpt-4o", api_key="sk-...")

response = await provider.complete([
    SystemMessage(content="You are helpful."),
    UserMessage(content="What is 2+2?"),
])
print(response.content)        # "4"
print(response.usage)          # Usage(input_tokens=..., output_tokens=..., total_tokens=...)
print(response.finish_reason)  # "stop"
```

### Streaming with Direct Provider

```python
from exo.models import get_provider
from exo.types import UserMessage

provider = get_provider("anthropic:claude-sonnet-4-20250514")

async for chunk in provider.stream([UserMessage(content="Tell me a story")]):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    if chunk.finish_reason:
        print(f"\n[Finished: {chunk.finish_reason}]")
```

### OpenAI-Compatible Proxy (e.g., Azure, LiteLLM, vLLM)

```python
agent = Agent(
    name="bot",
    model="openai:my-deployment",  # model name as the deployment expects
)

provider = get_provider(
    "openai:my-deployment",
    api_key="your-proxy-key",
    base_url="https://your-proxy.com/v1",
)

result = await run(agent, "Hello", provider=provider)
```

### Vertex AI with Service Account

```python
import base64, json

service_account = json.dumps({
    "type": "service_account",
    "project_id": "my-project",
    # ... rest of service account JSON
})

provider = get_provider(
    "vertex:gemini-2.5-pro",
    google_project="my-project",
    google_location="europe-west1",
    google_service_account_base64=base64.b64encode(service_account.encode()).decode(),
)
```

### Custom Provider Registration

```python
from exo.models.provider import ModelProvider, model_registry
from exo.models.types import ModelResponse, StreamChunk
from exo.config import ModelConfig

class MyProvider(ModelProvider):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # Initialize your client here

    async def complete(self, messages, *, tools=None, temperature=None, max_tokens=None):
        # Convert messages, call your API, return ModelResponse
        return ModelResponse(
            content="response text",
            model=self.config.model_name,
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def stream(self, messages, *, tools=None, temperature=None, max_tokens=None):
        # Call streaming API, yield StreamChunk objects
        yield StreamChunk(delta="Hello ")
        yield StreamChunk(delta="world!", finish_reason="stop")

# Register it
model_registry.register("my_llm", MyProvider)

# Use it
agent = Agent(name="bot", model="my_llm:my-model-v1")
```

### Different Models for Planning vs Execution

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",                           # Main execution model
    planning_enabled=True,
    planning_model="anthropic:claude-sonnet-4-20250514",  # Cheaper model for planning
    planning_instructions="Create a research plan with clear steps.",
)
```

### Error Handling

```python
from exo.models.types import ModelError

try:
    result = await run(agent, "Hello", provider=provider)
except ModelError as e:
    print(f"Model: {e.model}")    # "openai:gpt-4o"
    print(f"Code: {e.code}")      # "rate_limit", "context_length", etc.
    print(f"Message: {e}")        # "[openai:gpt-4o] Rate limit exceeded"
```

## Provider-Specific Notes

### OpenAI
- Default max_tokens: `None` (OpenAI applies its own default)
- Streaming includes usage via `stream_options={"include_usage": True}`
- Extracts `reasoning_content` from o1/o3 model responses via `model_extra`
- `base_url` enables proxies, Azure OpenAI, vLLM, LiteLLM, etc.

### Anthropic
- Default max_tokens: **4096** (hardcoded — Anthropic requires this field)
- System messages extracted to `system=` kwarg (not in messages list)
- Consecutive `ToolResult` messages are merged into a single user message (Anthropic API requirement)
- Extracts `reasoning_content` from `thinking` blocks
- Tool format uses `input_schema` instead of OpenAI's `parameters`

### Gemini
- Falls back to `GEMINI_API_KEY` env var when `api_key` not provided
- Generates synthetic tool call IDs (`call_0`, `call_1`) when the API omits them
- System messages passed via `system_instruction` config, not in messages
- Safety filter reasons mapped to `"content_filter"` finish reason

### Vertex AI
- Same message/tool format as Gemini (shared Google SDK)
- Requires `GOOGLE_CLOUD_PROJECT` env var or `google_project` kwarg
- Location defaults to `"us-central1"` — override with `GOOGLE_CLOUD_LOCATION` or `google_location`
- Credentials: service account base64 → `google.oauth2.service_account.Credentials`, or GCP ADC fallback

## Gotchas

- **No colon in model string → OpenAI assumed** — `"gpt-4o"` works but `"claude-sonnet-4-20250514"` without `"anthropic:"` prefix will try to find it as an OpenAI model and fail
- **API keys default to env vars** — if `api_key=None`, each provider's SDK reads its own env var automatically. Only pass `api_key` explicitly when you need to override.
- **Anthropic hardcodes max_tokens=4096** — unlike OpenAI/Gemini, Anthropic requires `max_tokens`. The provider defaults to 4096 if you don't specify. Pass `max_tokens` on Agent or in `complete()`/`stream()` to override.
- **Vertex AI needs GCP auth** — either set up ADC (`gcloud auth application-default login`) or pass `google_service_account_base64`
- **`ModelConfig.extra = "allow"`** — provider-specific kwargs (like `google_project`) are silently stored. Typos won't raise errors.
- **Context window is informational** — `MODEL_CONTEXT_WINDOWS` is used by context management (summarization, windowing) but providers don't enforce it. Exceeding it causes a provider-side error.
- **Provider instantiation is lazy** — `Agent(model="openai:gpt-4o")` doesn't create a provider until `run()` is called. Invalid model strings won't fail at init time.
- **Temperature default is 1.0 on Agent** — but `None` is passed to providers when calling `complete()`/`stream()` directly, which uses the provider's own default
- **Media tools are standalone** — they create their own SDK clients internally and don't use the agent's provider. They read env vars directly.
- **Unsupported content types don't error** — if you send `AudioBlock` to Anthropic, it logs a warning and skips the block silently
- **`FinishReason` is normalized** — all providers map to 4 values: `"stop"`, `"tool_calls"`, `"length"`, `"content_filter"`. Provider-specific reasons are lost.
