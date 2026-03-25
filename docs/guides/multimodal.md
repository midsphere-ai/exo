# Multimodal Support

Exo supports multimodal inputs and outputs across all built-in LLM providers (OpenAI, Anthropic, Gemini, Vertex). Agents can receive images, audio, video, and documents from users, and tools can return media that the LLM processes directly.

---

## Content Block Types

Six frozen Pydantic models represent media content. Import them from `exo` (re-exported from `exo.types`):

```python
from exo import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    TextBlock,
    VideoBlock,
)
```

| Type | `type` field | Key fields |
|---|---|---|
| `TextBlock` | `"text"` | `text: str` |
| `ImageURLBlock` | `"image_url"` | `url: str`, `detail: "auto"\|"low"\|"high"` |
| `ImageDataBlock` | `"image_data"` | `data: str` (base64), `media_type: str` |
| `AudioBlock` | `"audio"` | `data: str` (base64), `format: str` |
| `VideoBlock` | `"video"` | `data: str \| None`, `url: str \| None`, `media_type: str` |
| `DocumentBlock` | `"document"` | `data: str` (base64), `media_type: str`, `title: str \| None` |

`ContentBlock` is a discriminated union of all six types. `MessageContent = str | list[ContentBlock]`.

---

## Sending Multimodal Input to an Agent

Pass a list of `ContentBlock` objects as the `input` to `run()`, or as a `UserMessage.content`:

```python
import base64
from exo import run, Agent, ImageURLBlock, TextBlock

agent = Agent(name="vision", model="openai:gpt-4o")

# Image from URL
result = await run(agent, [
    TextBlock(text="What is in this image?"),
    ImageURLBlock(url="https://example.com/photo.jpg", detail="high"),
])

# Image from bytes
with open("diagram.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

result = await run(agent, [
    TextBlock(text="Explain this diagram."),
    ImageDataBlock(data=b64, media_type="image/png"),
])
```

For audio:

```python
from exo import AudioBlock

with open("speech.mp3", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

result = await run(agent, [
    TextBlock(text="Transcribe this audio."),
    AudioBlock(data=b64, format="mp3"),
])
```

For PDF documents (Anthropic):

```python
from exo import DocumentBlock

with open("report.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

result = await run(agent, [
    TextBlock(text="Summarise this report."),
    DocumentBlock(data=b64, media_type="application/pdf", title="Q1 Report"),
])
```

---

## Tools That Return Media

A tool can return `list[ContentBlock]` — the agent automatically propagates the blocks to the LLM so it can reason about the generated media:

```python
from exo import tool, ImageDataBlock
import base64

@tool
async def capture_screenshot(url: str) -> list[ImageDataBlock]:
    """Capture a screenshot of a URL."""
    screenshot_bytes = await _take_screenshot(url)
    b64 = base64.b64encode(screenshot_bytes).decode()
    return [ImageDataBlock(data=b64, media_type="image/png")]

agent = Agent(
    name="auditor",
    model="anthropic:claude-opus-4-6",
    tools=[capture_screenshot],
)
```

The LLM receives the image alongside the tool result and can describe, analyze, or make decisions based on it.

---

## Built-in Generation Tools

Three ready-to-use tools are available in `exo.models`. Add them directly to any agent.

### DALL-E 3 Image Generation

```python
from exo import Agent
from exo.models import dalle_generate_image

agent = Agent(
    name="artist",
    model="openai:gpt-4o",
    tools=[dalle_generate_image],
)

result = await run(agent, "Create a painting of a sunset over Mount Fuji.")
```

Parameters: `prompt`, `size` (`"1024x1024"` / `"1792x1024"` / `"1024x1792"`), `quality` (`"standard"` / `"hd"`), `style` (`"vivid"` / `"natural"`).

Returns: `list[ImageURLBlock]`.

Requires: `OPENAI_API_KEY` environment variable.

### Imagen 3 Image Generation

```python
from exo.models import imagen_generate_image

agent = Agent(
    name="illustrator",
    model="gemini:gemini-2.0-flash",
    tools=[imagen_generate_image],
)
```

Parameters: `prompt`, `aspect_ratio` (`"1:1"` / `"16:9"` / `"9:16"` / `"4:3"` / `"3:4"`), `number_of_images` (1-4).

Returns: `list[ImageDataBlock]` (base64 PNG).

Requires: `GOOGLE_API_KEY` environment variable.

### Veo 2 Video Generation

```python
from exo.models import veo_generate_video

agent = Agent(
    name="director",
    model="vertex:gemini-2.0-flash",
    tools=[veo_generate_video],
)
```

Parameters: `prompt`, `duration_seconds` (5-8), `aspect_ratio` (`"16:9"` / `"9:16"`).

Returns: `list[VideoBlock]` (URL to generated video in Cloud Storage).

Requires: `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` environment variables.

---

## Provider Capability Matrix

| Block type | OpenAI | Anthropic | Gemini | Vertex |
|---|---|---|---|---|
| `TextBlock` | ✅ | ✅ | ✅ | ✅ |
| `ImageURLBlock` (https) | ✅ | ✅ | ✅ | ✅ |
| `ImageURLBlock` (data URI) | ✅ | ✅ | ✅ | ✅ |
| `ImageDataBlock` | ✅ (via data URI) | ✅ | ✅ | ✅ |
| `AudioBlock` | ✅ | ❌ (warning) | ✅ | ✅ |
| `VideoBlock` | ❌ (warning) | ❌ (warning) | ✅ | ✅ |
| `DocumentBlock` (PDF) | ❌ (warning) | ✅ | ✅ | ✅ |
| Tool result with media | synthetic user msg | native | native | native |

Unsupported blocks are silently skipped with a `WARNING` log. OpenAI tool results containing media blocks inject a synthetic `user` message after the tool message so the LLM can see the images.

---

## End-to-End Example: Vision Agent + Image Generation Swarm

```python
import asyncio
from exo import Agent, Swarm, run, ImageURLBlock, TextBlock
from exo.models import dalle_generate_image

# Agent 1: Analyze an input image
analyst = Agent(
    name="analyst",
    model="openai:gpt-4o",
    instructions="You describe images in detail, focusing on style and composition.",
)

# Agent 2: Generate a variation based on the description
generator = Agent(
    name="generator",
    model="openai:gpt-4o",
    instructions="You generate image prompts and call dalle_generate_image.",
    tools=[dalle_generate_image],
)

# Sequential swarm: analyst → generator
swarm = Swarm(agents=[analyst, generator], flow="analyst >> generator")

async def main() -> None:
    result = await run(
        swarm,
        [
            TextBlock(text="Analyze this painting and generate a modern variant."),
            ImageURLBlock(url="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
        ],
    )
    print(result.output)

asyncio.run(main())
```

The analyst describes the Mona Lisa; the generator calls `dalle_generate_image` with a modernized prompt; the final response includes the URL of the generated image.
