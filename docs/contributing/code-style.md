# Code Style

Exo enforces a consistent code style across all packages. This document describes the rules and conventions.

## Formatting

| Rule | Value | Enforced By |
|------|-------|-------------|
| Line length | 100 characters | ruff |
| Quotes | Double quotes | ruff |
| Indentation | 4 spaces | ruff |
| Import sorting | stdlib, third-party, local | ruff (isort rules) |
| Trailing commas | Yes, for multi-line | ruff |

## Type Hints

Type hints are **required on all public functions and class attributes**. They are checked by pyright in strict mode.

### Preferred Syntax

```python
# Use modern union syntax (Python 3.10+)
def process(value: str | None = None) -> str: ...

# Use lowercase generics (Python 3.9+)
def collect(items: list[str]) -> dict[str, int]: ...

# NOT the typing module equivalents
from typing import Optional, List, Dict, Union  # Avoid these
```

### Rules

- Use `X | Y` union syntax, not `Union[X, Y]`
- Use `X | None` instead of `Optional[X]`
- Use `list`, `dict`, `tuple` lowercase, not `List`, `Dict`, `Tuple`
- Use `Literal["a", "b"]` for string enums in function signatures
- Use `Any` sparingly -- prefer specific types

## Docstrings

Docstrings use **Google style**. They are required on all public classes and functions. They are NOT required on private/internal helpers or test functions.

### Function Docstring

```python
def build_messages(
    instructions: str,
    history: list[Message],
    tool_results: list[ToolResult] | None = None,
) -> list[Message]:
    """Build the message list for an LLM call.

    Constructs a correctly ordered message sequence from system instructions,
    conversation history, and any pending tool results.

    Args:
        instructions: The system prompt.
        history: Previous conversation messages.
        tool_results: Results from tool calls to include.

    Returns:
        Ordered list of messages ready for the LLM provider.
    """
```

### Class Docstring

```python
class Agent:
    """An autonomous LLM-powered agent with tools and lifecycle hooks.

    Agents are the core building block in Exo. Each agent wraps an LLM
    model, a set of tools, optional handoff targets, and lifecycle hooks.

    Args:
        name: Unique identifier for this agent.
        model: Model string in "provider:model_name" format.
        instructions: System prompt for the agent.
        tools: Tools available to this agent.
    """
```

### Docstring Sections

Use these sections in order when applicable:

1. **Summary line** -- One line, imperative mood ("Build the message list", not "Builds the message list")
2. **Extended description** -- Optional, separated from summary by a blank line
3. **Args:** -- Parameter descriptions (required if function has parameters)
4. **Returns:** -- Return value description (required if function returns something other than None)
5. **Raises:** -- Exception descriptions (required if function raises)
6. **Yields:** -- For generators

## Pydantic Model Conventions

All configuration and data classes use Pydantic v2 `BaseModel`:

```python
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for an Agent."""

    model_config = {"frozen": True}  # immutable after creation

    name: str
    model: str = "openai:gpt-4o"
    instructions: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = None
    max_steps: int = Field(default=10, ge=1)
```

### Rules

- **Use `model_config = {"frozen": True}`** for config and data classes (immutability prevents bugs)
- **Use `Field()`** for validation constraints, defaults with metadata
- **Use plain defaults** for simple values without constraints
- **No `@validator`** -- use `@field_validator` (Pydantic v2)
- **All models inherit `BaseModel` directly** -- no `BaseConfig` intermediary class
- **No `ConfigDict` subclass** -- all configuration is via proper Pydantic models

## Exception Style

```python
class ToolExecutionError(ExoError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, cause: Exception):
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' failed: {cause}")
```

### Rules

- All Exo exceptions inherit from `ExoError`
- Include the thing that failed (tool name, agent name, provider) in the message
- Use `from e` for exception chaining
- Never silently swallow exceptions -- log or re-raise

## Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Modules | `snake_case.py` | `message_builder.py` |
| Classes | `PascalCase` | `Agent`, `ModelProvider`, `RunResult` |
| Functions | `snake_case` | `run()`, `get_provider()`, `build_messages()` |
| Constants | `UPPER_SNAKE` | `DEFAULT_MAX_STEPS`, `HookPoint.PRE_LLM_CALL` |
| Type aliases | `PascalCase` | `Message = UserMessage \| AssistantMessage \| ...` |
| Private/internal | `_prefix` | `_internal/`, `_sync_run()`, `_parse_tool_calls()` |
| Test functions | `test_<what>_<scenario>` | `test_agent_calls_tool_and_returns_result` |

## Model String Convention

Provider and model are specified as a single string: `"provider:model_name"`.

```python
"openai:gpt-4o"
"openai:gpt-4o-mini"
"anthropic:claude-sonnet-4-20250514"
"anthropic:claude-haiku-3-20240307"
```

Parsed by `parse_model_string()` in `exo.config`. If no colon prefix, defaults to `"openai"`.

## File Size Guidelines

- **Max ~200 lines** per source file (not counting tests)
- If a file grows beyond 200 lines, split into `_internal/` submodules
- `__init__.py` files are for re-exports only -- no logic
- Test files can be longer (up to ~300 lines)

## Import Order

Imports are sorted by ruff into three groups, separated by blank lines:

```python
# 1. Standard library
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

# 2. Third-party packages
from pydantic import BaseModel, Field

# 3. Local imports
from exo.types import Message, ExoError, ToolCall
from exo._internal.message_builder import build_messages
```

Within each group, imports are sorted alphabetically.
