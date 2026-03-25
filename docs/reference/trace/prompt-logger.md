# exo.trace.prompt_logger

Structured LLM execution logging with token breakdown and context analysis.

```python
from exo.trace.prompt_logger import (
    ExecutionLogEntry,
    PromptLogger,
    TokenBreakdown,
    compute_token_breakdown,
    estimate_tokens,
)
```

---

## estimate_tokens

```python
def estimate_tokens(text: str, ratio: float = 4.0) -> int
```

Estimate token count from character length using a simple `chars / ratio` heuristic. Returns at least 1 for non-empty strings, 0 for empty strings.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | Text to estimate tokens for |
| `ratio` | `float` | `4.0` | Character-to-token ratio |

**Returns:** `int` -- Estimated token count.

### Constants

| Constant | Value | Description |
|---|---|---|
| `DEFAULT_CHAR_TOKEN_RATIO` | `4.0` | Default characters per token |

---

## TokenBreakdown

```python
@dataclass(frozen=True, slots=True)
class TokenBreakdown
```

Per-role token counts and context window analysis.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `system` | `int` | `0` | Tokens in system messages |
| `user` | `int` | `0` | Tokens in user messages |
| `assistant` | `int` | `0` | Tokens in assistant messages |
| `tool` | `int` | `0` | Tokens in tool messages |
| `other` | `int` | `0` | Tokens in other role messages |

### Properties

| Property | Type | Description |
|---|---|---|
| `total` | `int` | Sum of all role token counts |

### Methods

#### percentages

```python
def percentages(self, context_window: int) -> dict[str, float]
```

Return a `role -> percentage` mapping relative to *context_window*.

| Name | Type | Default | Description |
|---|---|---|---|
| `context_window` | `int` | *(required)* | Total context window size in tokens |

**Returns:** Dict with keys `"system"`, `"user"`, `"assistant"`, `"tool"`, `"other"`, `"free"`, each a percentage (0-100).

### Example

```python
bd = TokenBreakdown(system=100, user=200, assistant=150, tool=50)
print(bd.total)  # 500

pcts = bd.percentages(1000)
print(pcts["free"])  # 50.0 (500 tokens free)
```

---

## compute_token_breakdown

```python
def compute_token_breakdown(
    messages: Sequence[dict[str, Any]],
    *,
    ratio: float = 4.0,
) -> TokenBreakdown
```

Compute a `TokenBreakdown` from a list of message dicts. Each message is expected to have `role` and `content` keys. Multi-modal content (a list of content items) is handled by summing text items and counting images as a fixed 85-token estimate. `tool_calls` in assistant messages are included in the token count.

| Name | Type | Default | Description |
|---|---|---|---|
| `messages` | `Sequence[dict[str, Any]]` | *(required)* | List of OpenAI-style message dicts |
| `ratio` | `float` | `4.0` | Character-to-token ratio |

**Returns:** `TokenBreakdown`

### Example

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
]

bd = compute_token_breakdown(messages)
print(bd.system, bd.user, bd.assistant)
```

---

## ExecutionLogEntry

```python
@dataclass(slots=True)
class ExecutionLogEntry
```

Structured record of a single LLM execution.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_name` | `str` | `""` | Name of the agent |
| `model_name` | `str` | `""` | Model used |
| `message_count` | `int` | `0` | Number of messages in the conversation |
| `tool_names` | `list[str]` | `[]` | Tools available during the execution |
| `breakdown` | `TokenBreakdown` | `TokenBreakdown()` | Per-role token breakdown |
| `context_window` | `int` | `0` | Context window size in tokens |
| `duration_s` | `float` | `0.0` | Execution duration in seconds |

### Methods

#### format_summary

```python
def format_summary(self) -> str
```

Return a human-readable multi-line summary string including agent name, model, message count, duration, and token breakdown with percentages (when `context_window > 0`).

---

## PromptLogger

```python
class PromptLogger(
    *,
    log: logging.Logger | None = None,
    ratio: float = 4.0,
)
```

Structured LLM execution logger. Accepts message dicts (OpenAI-style), computes token breakdown, and writes structured log entries.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `log` | `logging.Logger \| None` | `None` | Logger instance. Defaults to `exo.trace.prompt` logger |
| `ratio` | `float` | `4.0` | Character-to-token ratio for estimation |

### Methods

#### log_execution

```python
def log_execution(
    self,
    messages: Sequence[dict[str, Any]],
    *,
    agent_name: str = "",
    model_name: str = "",
    context_window: int = 0,
    tool_names: Sequence[str] | None = None,
    duration_s: float = 0.0,
    level: int = logging.INFO,
) -> ExecutionLogEntry
```

Compute token breakdown and log a structured execution entry.

| Name | Type | Default | Description |
|---|---|---|---|
| `messages` | `Sequence[dict[str, Any]]` | *(required)* | OpenAI-style message dicts |
| `agent_name` | `str` | `""` | Agent name for the log entry |
| `model_name` | `str` | `""` | Model name for the log entry |
| `context_window` | `int` | `0` | Context window size in tokens (enables percentage display) |
| `tool_names` | `Sequence[str] \| None` | `None` | Tools available during execution |
| `duration_s` | `float` | `0.0` | Execution duration in seconds |
| `level` | `int` | `logging.INFO` | Log level |

**Returns:** `ExecutionLogEntry` for programmatic access.

### Example

```python
from exo.trace import PromptLogger

logger = PromptLogger()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain decorators in Python."},
    {"role": "assistant", "content": "Decorators are functions that..."},
]

entry = logger.log_execution(
    messages,
    agent_name="tutor",
    model_name="gpt-4o",
    context_window=128000,
    tool_names=["search", "code_runner"],
    duration_s=1.23,
)

print(entry.breakdown.total)
print(entry.format_summary())
```
