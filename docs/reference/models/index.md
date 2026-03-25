# exo-models

Provider-agnostic LLM abstraction layer for the Exo framework.

## Module Path

```
exo.models
```

## Installation

```bash
pip install exo-models
```

## Overview

The `exo-models` package provides a unified interface for calling LLM providers. It defines a `ModelProvider` abstract base class with `complete()` and `stream()` methods, normalized response types (`ModelResponse`, `StreamChunk`), and a registry-based factory for building providers from `"provider:model_name"` strings.

Four built-in providers are included: `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, and `VertexProvider`. All auto-register on import.

## Exports

| Export | Type | Description |
|---|---|---|
| `ModelProvider` | ABC | Abstract base for LLM provider implementations |
| `OpenAIProvider` | Class | OpenAI chat completions provider |
| `AnthropicProvider` | Class | Anthropic messages API provider |
| `GeminiProvider` | Class | Google Gemini API provider (API key auth) |
| `VertexProvider` | Class | Google Vertex AI provider (GCP ADC auth) |
| `ModelResponse` | Pydantic model | Non-streaming LLM response |
| `StreamChunk` | Pydantic model | Single streaming chunk |
| `ToolCallDelta` | Pydantic model | Incremental tool call fragment |
| `FinishReason` | Literal type | Why the model stopped generating |
| `ModelError` | Exception | LLM provider call failure |
| `model_registry` | `Registry[type[ModelProvider]]` | Global provider class registry |
| `get_provider` | Function | Factory: build a provider from a model string |

## Import Patterns

```python
# Import everything from the package
from exo.models import (
    ModelProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    VertexProvider,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
    FinishReason,
    ModelError,
    model_registry,
    get_provider,
)

# Common usage: build a provider via factory
from exo.models import get_provider

provider = get_provider("openai:gpt-4o", api_key="sk-...")
provider = get_provider("gemini:gemini-2.0-flash", api_key="AIza...")
provider = get_provider("vertex:gemini-2.0-flash")  # uses GCP ADC
```

## Quick Example

```python
import asyncio
from exo.models import get_provider
from exo.types import UserMessage

async def main():
    provider = get_provider("openai:gpt-4o", api_key="sk-...")

    # Non-streaming completion
    response = await provider.complete(
        [UserMessage(content="What is 2+2?")],
        temperature=0.0,
    )
    print(response.content)       # "4"
    print(response.usage)         # Usage(input_tokens=12, output_tokens=1, ...)
    print(response.finish_reason) # "stop"

    # Streaming completion
    async for chunk in await provider.stream(
        [UserMessage(content="Tell me a story")],
    ):
        if chunk.delta:
            print(chunk.delta, end="")
        if chunk.finish_reason:
            print(f"\n[Done: {chunk.finish_reason}]")

asyncio.run(main())
```

## Architecture

```
get_provider("openai:gpt-4o")
    |
    v
model_registry.get("openai")  -->  OpenAIProvider class
    |
    v
OpenAIProvider(ModelConfig(...))  -->  provider instance
    |
    v
provider.complete(messages)  -->  ModelResponse
provider.stream(messages)    -->  AsyncIterator[StreamChunk]
```

## See Also

- [types](types.md) -- Response and error types
- [provider](provider.md) -- ModelProvider ABC and factory
- [openai](openai.md) -- OpenAI provider details
- [anthropic](anthropic.md) -- Anthropic provider details
- [gemini](gemini.md) -- Google Gemini provider details
- [vertex](vertex.md) -- Google Vertex AI provider details
