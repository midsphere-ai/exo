# Context Enhancement Design — Offloading, Compression, Windowing, Token Budgeting

**Status:** Proposed
**Epic:** 4 — Context Engine Enhancements
**Date:** 2026-03-10

---

## 1. Motivation

Orbiter's `orbiter-context` package provides a solid foundation:

- **ContextProcessor ABC** — event-driven processors with `process(ctx, payload)`.
- **ProcessorPipeline** — sequential dispatch by event type via `register()`.
- **SummarizeProcessor** — marks history for summarization at a configurable threshold.
- **ToolResultOffloader** — truncates large tool results and stores full content in state.
- **TokenTracker** — per-agent, per-step token usage tracking.
- **ContextConfig** — immutable config with `summary_threshold`, `offload_threshold`,
  `history_rounds`, and automation modes (pilot/copilot/navigator).

However, it lacks several context engineering capabilities found in agent-core
(`openjiuwen/core/context_engine/`):

1. **Message-level offloading** — replacing oversized messages with `[[OFFLOAD: handle=<id>]]`
   markers, with a tool to reload content on demand.
2. **Dialogue compression** — LLM-based compression of verbose tool call chains.
3. **Round-level windowing** — understanding dialogue structure (one round = user message
   through assistant response without tool calls) for smarter history trimming.
4. **Token budgeting** — accurate token counting via tiktoken encodings to enforce
   hard token limits before LLM calls.

This document proposes adding four new processors and one reload tool to
`orbiter-context`, all plugging into the existing `ProcessorPipeline` with zero
changes to the pipeline itself.

---

## 2. Key Decision: Extend, Don't Replace

### Option A — New pipeline or processor base (rejected)

Create a separate processor system for the new capabilities. This duplicates
dispatch logic, forces consumers to choose between pipelines, and breaks the
single `ProcessorPipeline` model.

### Option B — New processors in existing pipeline (chosen)

All four enhancements are implemented as new `ContextProcessor` subclasses that
register on existing pipeline events (`pre_llm_call`, `post_tool_call`). The
reload capability is a standard context tool.

**Why Option B:**

- Zero changes to `ProcessorPipeline`, `ContextProcessor`, or any existing code.
- New processors are opt-in — `pipeline.register(MessageOffloader())`.
- Existing `SummarizeProcessor` and `ToolResultOffloader` continue to work unchanged.
- Consumers compose processors freely: any combination, any order.

---

## 3. New Processor: MessageOffloader

**Event:** `pre_llm_call`
**Module:** `processor.py` (added to existing file)

Scans `ctx.state["history"]` for messages exceeding a character limit. Oversized
messages are replaced with a marker containing a unique handle ID. The original
content is stored in `ctx.state["offloaded_messages"]` keyed by handle.

### Behaviour

```
Input message:  { "role": "assistant", "content": "<very long content>" }
Output message: { "role": "assistant", "content": "[[OFFLOADED: handle=off_abc123]]" }
State:          ctx.state["offloaded_messages"]["off_abc123"] = "<original>"
```

### Rules

- Only offloads `user`, `assistant`, and `tool` messages — **never system messages**.
- Default `max_message_size = 10000` characters (constructor parameter).
- Handle IDs use `f"off_{uuid4().hex[:12]}"` for uniqueness.
- Messages within the limit are untouched.

### Interaction with existing processors

- **SummarizeProcessor** fires on `pre_llm_call` before or after MessageOffloader
  (registration order). Both read `ctx.state["history"]` independently. No conflict:
  SummarizeProcessor marks *which* messages are candidates; MessageOffloader replaces
  *content* of oversized ones.
- **ToolResultOffloader** fires on `post_tool_call` (different event), so no
  ordering concern with MessageOffloader.

---

## 4. Reload Tool for Offloaded Content

**Module:** `tools.py` (new tool function added to existing context tools)

A tool function that agents can call to retrieve offloaded message content. Looks
up a handle ID in `ctx.state["offloaded_messages"]` and returns the original content.

### Signature

```python
async def reload_offloaded(ctx: Context, handle: str) -> str:
    """Retrieve content previously offloaded by MessageOffloader."""
```

### Behaviour

- Returns full original content for valid handles.
- Raises `ContextError` for unknown handle IDs.
- Registered via `get_context_tools()` so agents automatically have access when
  context tools are enabled.

---

## 5. New Processor: DialogueCompressor

**Event:** `pre_llm_call`
**Module:** `processor.py`

Detects chains of tool call messages (assistant with `tool_calls` → tool results →
assistant with `tool_calls` → ...) in history and replaces them with LLM-generated
summaries. This dramatically reduces context size for multi-step tool use.

### Chain Detection

A "tool chain" is a contiguous sequence of:
1. Assistant message with `tool_calls` field
2. One or more tool result messages
3. Optionally followed by another assistant+tool sequence

The chain ends when an assistant message has **no** `tool_calls` — that message
and subsequent ones are preserved.

### Compression

```python
async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
    # 1. Scan history for tool chains
    # 2. For each chain, call summarizer (injected callable) to produce summary
    # 3. Replace chain in history with single assistant message containing summary
```

The summarizer is a constructor-injected async callable:
`summarizer: Callable[[list[dict]], Awaitable[str]]`. This keeps the processor
LLM-agnostic — the caller provides the summarization function.

### Interaction with existing processors

- Runs on `pre_llm_call` alongside `SummarizeProcessor`. Registration order matters:
  **DialogueCompressor should be registered before SummarizeProcessor** so that
  compression reduces history length before the threshold check.
- MessageOffloader can run before or after — compressed chains don't contain
  offloaded markers (they're replaced entirely).

---

## 6. New Processor: RoundWindowProcessor

**Event:** `pre_llm_call`
**Module:** `processor.py`

Trims history to the most recent N dialogue rounds, preserving complete round
boundaries. This prevents mid-conversation truncation that confuses the model.

### Round Definition

One dialogue round = a user message through the next assistant message that does
**not** contain `tool_calls`. Tool-calling sequences within a round are preserved
as a unit.

```
Round 1: [user] → [assistant+tool_calls] → [tool_result] → [assistant]
Round 2: [user] → [assistant]
Round 3: [user] → [assistant+tool_calls] → [tool_result] → [assistant+tool_calls] → [tool_result] → [assistant]
```

### Behaviour

- Uses `ctx.config.history_rounds` as the window size (already in ContextConfig).
- Counts rounds from the end of history backward.
- Trims all messages before the window boundary.
- If the current round is incomplete (no final assistant response yet), it is
  always preserved regardless of the window.

### Interaction with existing processors

- Complements `SummarizeProcessor` — windowing trims old rounds, summarization
  handles the threshold within the remaining window.
- Should be registered **after** `DialogueCompressor` (compressed history has fewer
  messages, so round counting is more accurate).

---

## 7. New Processor: TokenBudgetProcessor

**Event:** `pre_llm_call`
**Module:** `processor.py`

Enforces a hard token budget on the context before sending to the LLM. Uses
tiktoken for accurate token counting (cl100k_base for GPT-3.5/4, o200k_base for
GPT-4o/newer models).

### Behaviour

1. Count tokens for all messages in `ctx.state["history"]` using tiktoken.
2. If total exceeds `max_tokens` (constructor parameter), trim oldest messages
   (respecting round boundaries from RoundWindowProcessor if active) until the
   budget is met.
3. Store trimmed message count and token savings in `ctx.state["token_budget_trimmed"]`.

### Token Counting

```python
import tiktoken

class TokenBudgetProcessor(ContextProcessor):
    def __init__(self, *, max_tokens: int = 100_000, encoding: str = "cl100k_base"):
        super().__init__("pre_llm_call", name="token_budget")
        self._max_tokens = max_tokens
        self._enc = tiktoken.get_encoding(encoding)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))
```

### ContextConfig Integration

The `max_tokens` budget can be passed via `ctx.config.extra["token_budget"]`,
allowing per-agent configuration without changing the frozen ContextConfig schema:

```python
config = ContextConfig(extra={"token_budget": 80_000, "token_encoding": "o200k_base"})
```

### Interaction with existing processors

- Should be the **last** processor in the `pre_llm_call` chain (registered last).
  It acts as a final safety net after summarization, compression, and windowing.
- Reads the same `ctx.state["history"]` that other processors modify.

---

## 8. Recommended Processor Registration Order

For the `pre_llm_call` event, the recommended registration order is:

```python
pipeline = ProcessorPipeline()

# 1. Offload oversized individual messages first
pipeline.register(MessageOffloader(max_message_size=10_000))

# 2. Compress verbose tool chains into summaries
pipeline.register(DialogueCompressor(summarizer=my_summarizer))

# 3. Window to recent N rounds (uses config.history_rounds)
pipeline.register(RoundWindowProcessor())

# 4. Check if summarization is needed for remaining messages
pipeline.register(SummarizeProcessor())

# 5. Final token budget enforcement
pipeline.register(TokenBudgetProcessor(max_tokens=100_000))
```

For the `post_tool_call` event:

```python
# Existing — offload large tool results
pipeline.register(ToolResultOffloader(max_size=5000))
```

Each processor is independent and optional. Users can register any subset in any
order. The above order is a best-practice recommendation.

---

## 9. Backward Compatibility

### Existing APIs — no changes required

| Component | Impact |
|-----------|--------|
| `ContextProcessor` ABC | No changes — new processors extend it |
| `ProcessorPipeline` | No changes — `register()` works for new processors |
| `SummarizeProcessor` | No changes — continues to work as-is |
| `ToolResultOffloader` | No changes — continues to work as-is |
| `TokenTracker` | No changes — token budgeting is a separate concern |
| `ContextConfig` | No changes — uses existing `extra` dict for new config |
| `Context` | No changes — processors use `ctx.state` and `ctx.config` |

### New additions (purely additive)

| Addition | Location |
|----------|----------|
| `MessageOffloader` class | `processor.py` |
| `DialogueCompressor` class | `processor.py` |
| `RoundWindowProcessor` class | `processor.py` |
| `TokenBudgetProcessor` class | `processor.py` |
| `reload_offloaded()` tool | `tools.py` |
| New exports in `__init__.py` | `__init__.py` |

### Existing hook/processor calls — unchanged

All existing `pipeline.register(SummarizeProcessor())` and
`pipeline.register(ToolResultOffloader(...))` calls work without modification.
The pipeline's `fire()` method calls processors in registration order, and new
processors are only invoked if explicitly registered.

---

## 10. Dependencies

- **tiktoken** — new dependency for `TokenBudgetProcessor` only. Added as an
  optional dependency in `orbiter-context[tiktoken]` to avoid forcing it on users
  who don't need token budgeting. The processor raises `ImportError` with a clear
  message if tiktoken is not installed.
- No other new dependencies.

---

## 11. Test Strategy

Each new processor gets its own test class with the following coverage:

| Processor | Key test cases |
|-----------|---------------|
| `MessageOffloader` | Within-limit untouched, oversized replaced, system messages skipped, handle stored |
| `DialogueCompressor` | Chain detection, single-chain compression, multi-chain, mock summarizer |
| `RoundWindowProcessor` | Round boundary detection, incomplete round preserved, mixed sequences |
| `TokenBudgetProcessor` | Under-budget no-op, over-budget trimming, encoding selection |
| `reload_offloaded` | Valid handle returns content, unknown handle raises error |
| Integration | All processors composed in recommended order |
