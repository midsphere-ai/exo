# exo-models

LLM provider abstractions for the [Exo](../../README.md) multi-agent framework.

## Installation

```bash
pip install exo-models
```

Requires Python 3.11+. Installs `exo-core`, `openai>=1.0`, `anthropic>=0.39`, and `google-genai>=1.0`.

## Supported Providers

| Provider | Model String | SDK |
|----------|-------------|-----|
| OpenAI | `"openai:gpt-4o"`, `"openai:gpt-4o-mini"` | `openai` |
| Anthropic | `"anthropic:claude-sonnet-4-20250514"` | `anthropic` |
| Gemini | `"gemini:gemini-2.0-flash"` | `google-genai` |
| Vertex AI | `"vertex:gemini-2.0-flash"` | `google-genai` |

## Quick Example

```python
from exo import Agent, run

# Models are resolved automatically from the model string
agent = Agent(name="bot", model="openai:gpt-4o-mini")
result = run.sync(agent, "Hello!")
print(result.output)

# Switch providers by changing the model string
agent = Agent(name="bot", model="anthropic:claude-sonnet-4-20250514")
```

## Direct Provider Usage

```python
from exo.models.provider import get_provider

provider = get_provider("openai:gpt-4o")

# Or with an explicit API key
provider = get_provider("openai:gpt-4o", api_key="sk-...")
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."                # For Gemini
export GOOGLE_CLOUD_PROJECT="my-project"   # For Vertex AI
```

## Public API

```python
from exo.models import (
    ModelProvider,    # Abstract base class for providers
    ModelResponse,    # Response from a complete() call
    StreamChunk,      # Chunk from a stream() call
    ModelError,       # Provider error
    FinishReason,     # Why the model stopped
    ToolCallDelta,    # Partial tool call in a stream
    get_provider,     # Factory: model string -> provider instance
    model_registry,   # Registry of provider classes
)
```

## Documentation

- [Model Providers Guide](../../docs/guides/models.md)
- [API Reference](../../docs/reference/models/)
