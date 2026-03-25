# exo.memory.summary

Summary trigger logic and multi-template summary generation for compressing long conversations.

## Module Path

```python
from exo.memory.summary import (
    Summarizer,
    SummaryConfig,
    SummaryResult,
    SummaryTemplate,
    TriggerResult,
    check_trigger,
    generate_summary,
)
```

---

## SummaryTemplate

Built-in summarization templates.

```python
class SummaryTemplate(StrEnum):
    CONVERSATION = "conversation"
    FACTS = "facts"
    PROFILES = "profiles"
```

| Value | Default Prompt Focus |
|---|---|
| `CONVERSATION` | Key decisions, action items, important context |
| `FACTS` | Factual statements and verified information |
| `PROFILES` | User preferences, personality traits, background |

---

## SummaryConfig

Configuration for summary triggers and generation.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
SummaryConfig(
    message_threshold: int = 20,
    token_threshold: int = 4000,
    templates: tuple[SummaryTemplate, ...] = (SummaryTemplate.CONVERSATION,),
    prompts: dict[SummaryTemplate, str] = {},
    keep_recent: int = 4,
    token_estimate_ratio: float = 4.0,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `message_threshold` | `int` | `20` | Trigger when message count exceeds this |
| `token_threshold` | `int` | `4000` | Trigger when estimated tokens exceed this |
| `templates` | `tuple[SummaryTemplate, ...]` | `(CONVERSATION,)` | Which templates to generate |
| `prompts` | `dict[SummaryTemplate, str]` | `{}` | Custom prompt overrides per template |
| `keep_recent` | `int` | `4` | Number of recent messages to preserve after compression |
| `token_estimate_ratio` | `float` | `4.0` | Characters-per-token ratio for estimation |

### Methods

#### get_prompt()

```python
def get_prompt(self, template: SummaryTemplate) -> str
```

Get the prompt for a template, falling back to built-in defaults.

---

## TriggerResult

Result of a trigger check.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Fields

| Field | Type | Description |
|---|---|---|
| `triggered` | `bool` | Whether a summary should be generated |
| `reason` | `str` | Description of the trigger condition |
| `message_count` | `int` | Current message count |
| `estimated_tokens` | `int` | Estimated total tokens |

---

## Summarizer (Protocol)

Protocol for LLM-powered summarization.

```python
@runtime_checkable
class Summarizer(Protocol):
    async def summarize(self, prompt: str) -> str: ...
```

---

## SummaryResult

Result of a summary generation run.

**Decorator:** `@dataclass(slots=True)`

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `summaries` | `dict[str, str]` | `{}` | Generated summaries keyed by template name |
| `compressed_items` | `list[MemoryItem]` | *(required)* | Items to keep after compression (recent tail) |
| `original_count` | `int` | *(required)* | Number of items before compression |

---

## check_trigger()

Check whether summary generation should be triggered.

```python
def check_trigger(
    items: Sequence[MemoryItem],
    config: SummaryConfig,
) -> TriggerResult
```

Triggers when EITHER message count or estimated token count exceeds the configured thresholds.

| Parameter | Type | Description |
|---|---|---|
| `items` | `Sequence[MemoryItem]` | Memory items to evaluate |
| `config` | `SummaryConfig` | Configuration with thresholds |

**Returns:** `TriggerResult`.

---

## generate_summary()

Generate summaries from memory items using configured templates.

```python
async def generate_summary(
    items: Sequence[MemoryItem],
    config: SummaryConfig,
    summarizer: Any,
) -> SummaryResult
```

| Parameter | Type | Description |
|---|---|---|
| `items` | `Sequence[MemoryItem]` | Memory items to summarize |
| `config` | `SummaryConfig` | Summary configuration |
| `summarizer` | `Any` | Object implementing the `Summarizer` protocol |

**Returns:** `SummaryResult` with generated summaries and compressed item list.

**Behavior:**
1. Splits items into "to compress" (older) and "recent" (kept as-is)
2. Formats compressed items into conversation text
3. Generates a summary for each configured template
4. Returns summaries + recent items

---

## Example

```python
import asyncio
from exo.memory import (
    HumanMemory, AIMemory,
    SummaryConfig, SummaryTemplate,
    check_trigger, generate_summary,
)

class MySummarizer:
    async def summarize(self, prompt: str) -> str:
        return "Summary: Discussion about Python programming."

async def main():
    items = [
        HumanMemory(content=f"Message {i}") for i in range(25)
    ]

    config = SummaryConfig(
        message_threshold=20,
        templates=(SummaryTemplate.CONVERSATION, SummaryTemplate.FACTS),
        keep_recent=4,
    )

    # Check trigger
    trigger = check_trigger(items, config)
    print(f"Triggered: {trigger.triggered}")  # True
    print(f"Reason: {trigger.reason}")

    if trigger.triggered:
        result = await generate_summary(items, config, MySummarizer())
        print(f"Summaries: {list(result.summaries.keys())}")
        print(f"Kept {len(result.compressed_items)} recent items")
        print(f"Original count: {result.original_count}")

asyncio.run(main())
```
