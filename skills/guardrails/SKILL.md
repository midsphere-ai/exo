---
name: exo:guardrails
description: "Use when adding security guardrails to Exo agents — BaseGuardrail, UserInputGuardrail, PatternBackend, LLMGuardrailBackend, Rail ABC, RailManager, RailAction (CONTINUE/SKIP/RETRY/ABORT), prompt injection detection, content filtering. Triggers on: guardrail, security, prompt injection, jailbreak, content filter, rail, RailAction, safety, risk detection, BaseGuardrail."
---

# Exo Guardrails — Security Guardrails

## When To Use This Skill

Use this skill when the developer needs to:
- Detect and block prompt injection or jailbreak attempts
- Add content filtering to agent inputs or outputs
- Block dangerous tool calls
- Use pattern-based (regex) or LLM-based detection
- Control execution flow with Rails (CONTINUE/SKIP/RETRY/ABORT)
- Combine multiple detection strategies

## Decision Guide

1. **Quick regex-based injection detection?** → `UserInputGuardrail()` — zero API calls, built-in patterns
2. **Sophisticated content analysis?** → `LLMGuardrailBackend(model="openai:gpt-4o-mini")` — uses a cheap model for each check
3. **Need to control execution flow (skip, retry, abort)?** → `Rail` ABC with `RailAction`
4. **Need both pattern + LLM detection?** → Attach both: `UserInputGuardrail` for fast check, `LLMGuardrailBackend`-backed guardrail for deep analysis
5. **Need to block specific tools?** → Subclass `Rail`, check `ToolCallInputs.tool_name`

**Two systems available (can coexist):**
- **BaseGuardrail** (exo-guardrail package): Hook-based, attach/detach, pluggable backends
- **Rail** (exo-core): Priority-ordered lifecycle guards with CONTINUE/SKIP/RETRY/ABORT actions

## Reference

### System 1: BaseGuardrail + Backends

#### Risk Levels

```python
from exo.guardrail.types import RiskLevel

RiskLevel.SAFE       # No risk detected
RiskLevel.LOW        # Minor concern — passes through
RiskLevel.MEDIUM     # Moderate concern — passes through
RiskLevel.HIGH       # Blocked automatically
RiskLevel.CRITICAL   # Blocked automatically
```

**Blocking behavior:** Only `HIGH` and `CRITICAL` trigger automatic blocking (raise `GuardrailError`). `LOW` and `MEDIUM` are detected but allowed through.

#### UserInputGuardrail (Pattern-Based)

Built-in regex-based detection for common injection patterns:

```python
from exo.guardrail.user_input import UserInputGuardrail

guard = UserInputGuardrail()
guard.attach(agent)
# Agent now blocks prompt injection attempts at PRE_LLM_CALL

# Detach when no longer needed
guard.detach(agent)
```

**Default patterns detect:**
- Instruction overrides: "ignore all previous instructions", "disregard prior rules"
- Role impersonation: "you are now in DAN mode", "act as unrestricted"
- System prompt extraction: "reveal your system prompt", "what are your instructions"
- Delimiter attacks: `[INST]`, `<<SYS>>`, `` ```system ``
- Encoded injection: `base64:decode`, `eval()`, `exec()`

**Custom patterns:**

```python
from exo.guardrail.user_input import UserInputGuardrail, PatternBackend
from exo.guardrail.types import RiskLevel

# Add extra patterns on top of defaults
guard = UserInputGuardrail(
    extra_patterns=[
        (r"bypass\s+safety", RiskLevel.HIGH, "safety_bypass"),
        (r"sudo\s+mode", RiskLevel.CRITICAL, "privilege_escalation"),
    ],
)

# Or replace defaults entirely
custom_backend = PatternBackend(
    patterns=[
        (r"my_custom_pattern", RiskLevel.HIGH, "custom_risk"),
    ],
)
guard = UserInputGuardrail(backend=custom_backend)
```

**Pattern tuple format:** `(regex_str, RiskLevel, description)`

#### LLMGuardrailBackend

Uses an LLM to analyze messages for sophisticated threats:

```python
from exo.guardrail.llm_backend import LLMGuardrailBackend
from exo.guardrail.base import BaseGuardrail

backend = LLMGuardrailBackend(
    model="openai:gpt-4o-mini",          # Cheap model for analysis
    api_key=None,                          # Uses env var if None
    prompt_template=None,                  # Uses built-in template if None
)

guard = BaseGuardrail(
    backend=backend,
    events=["pre_llm_call"],               # Which hook points to monitor
)
guard.attach(agent)
```

**Built-in prompt template detects:**
- Prompt injection
- Jailbreak attempts
- PII leakage requests
- Harmful content requests

**Custom prompt template:**

```python
custom_template = """\
Analyze this message for company policy violations:

Message: {user_message}

Respond with JSON: {{"has_risk": true/false, "risk_level": "safe"|"high", ...}}
"""

backend = LLMGuardrailBackend(
    model="openai:gpt-4o-mini",
    prompt_template=custom_template,
)
```

**Fail-safe:** If the LLM call fails or response can't be parsed, returns `RiskLevel.SAFE` (fails open).

#### Custom Backend

```python
from exo.guardrail.types import GuardrailBackend, RiskAssessment, RiskLevel
from typing import Any

class ContentPolicyBackend(GuardrailBackend):
    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        # data contains: {"event": "pre_llm_call", "messages": [...], ...}
        messages = data.get("messages", [])

        # Your detection logic here
        text = _get_latest_user_text(messages)
        if self._is_policy_violation(text):
            return RiskAssessment(
                has_risk=True,
                risk_level=RiskLevel.HIGH,
                risk_type="policy_violation",
                confidence=0.9,
                details={"reason": "Contains prohibited content"},
            )

        return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
```

#### GuardrailResult

```python
from exo.guardrail.types import GuardrailResult, RiskLevel

# Safe result
safe = GuardrailResult.safe()
# GuardrailResult(is_safe=True, risk_level=SAFE)

# Block result
blocked = GuardrailResult.block(
    risk_level=RiskLevel.HIGH,
    risk_type="prompt_injection",
    details={"matched_patterns": ["instruction_override"]},
)
# GuardrailResult(is_safe=False, risk_level=HIGH, risk_type="prompt_injection", ...)
```

#### Attach/Detach Lifecycle

```python
guard = UserInputGuardrail()

# Attach: registers hooks (idempotent — safe to call twice)
guard.attach(agent)

# Agent runs with protection
result = await run(agent, "ignore all previous instructions", provider=provider)
# Raises GuardrailError!

# Detach: removes exactly the hooks it registered
guard.detach(agent)
```

### System 2: Rails (exo-core)

#### Rail Actions

```python
from exo.rail import RailAction

RailAction.CONTINUE  # Pass — proceed to next rail or operation
RailAction.SKIP      # Skip the guarded operation entirely
RailAction.RETRY     # Retry the operation (with delay/max)
RailAction.ABORT     # Abort the agent run immediately (raises RailAbortError)
```

#### Rail ABC

```python
from exo.rail import Rail, RailAction, RetryRequest
from exo.rail_types import RailContext, ToolCallInputs, ModelCallInputs

class BlockDangerousTool(Rail):
    def __init__(self):
        super().__init__(name="block-dangerous", priority=10)  # Lower = runs first

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if isinstance(ctx.inputs, ToolCallInputs):
            if ctx.inputs.tool_name in ("rm_rf", "drop_table", "delete_all"):
                return RailAction.ABORT
        return RailAction.CONTINUE  # or return None (same as CONTINUE)

class RetryOnRateLimit(Rail):
    def __init__(self):
        super().__init__(name="retry-rate-limit", priority=20)

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if isinstance(ctx.inputs, ModelCallInputs):
            if ctx.inputs.response and "rate_limit" in str(ctx.inputs.response):
                ctx.extra["retry"] = RetryRequest(delay=1.0, max_retries=3)
                return RailAction.RETRY
        return RailAction.CONTINUE
```

#### RailContext

```python
from exo.rail_types import RailContext

# ctx fields:
ctx.agent      # The agent instance
ctx.event      # HookPoint enum value
ctx.inputs     # Typed inputs (see below)
ctx.extra      # Shared dict across all rails in one invocation
```

#### Typed Inputs per Hook Point

| Hook Points | Input Type | Fields |
|------------|-----------|--------|
| `PRE_TOOL_CALL`, `POST_TOOL_CALL` | `ToolCallInputs` | `tool_name`, `arguments`, `result`, `metadata` |
| `PRE_LLM_CALL`, `POST_LLM_CALL` | `ModelCallInputs` | `messages`, `tools`, `response`, `usage` |
| `START`, `FINISHED`, `ERROR` | `InvokeInputs` | `input`, `messages`, `result` |

#### Using Rails on Agent

```python
agent = Agent(
    name="bot",
    rails=[
        BlockDangerousTool(),
        RetryOnRateLimit(),
    ],
)
```

**What happens internally:**
1. Creates `RailManager()`, adds all rails
2. For each `HookPoint`, registers a hook via `rail_manager.hook_for(event)`
3. Rails run in ascending priority order (lower priority number = runs first)
4. If any rail returns `ABORT`, `RailAbortError` is raised
5. Shared `extra` dict enables cross-rail coordination

#### RailManager

```python
from exo.rail import RailManager

manager = RailManager()
manager.add(my_rail)
manager.remove(my_rail)
manager.clear()

# Run all rails for an event
action = await manager.run(HookPoint.PRE_TOOL_CALL, agent=agent, tool_name="search", arguments={})
```

## Patterns

### Combined Pattern + LLM Detection

```python
from exo.guardrail.user_input import UserInputGuardrail
from exo.guardrail.llm_backend import LLMGuardrailBackend
from exo.guardrail.base import BaseGuardrail

# Layer 1: Fast regex check
pattern_guard = UserInputGuardrail()
pattern_guard.attach(agent)

# Layer 2: Deep LLM analysis for what patterns miss
llm_guard = BaseGuardrail(
    backend=LLMGuardrailBackend(model="openai:gpt-4o-mini"),
    events=["pre_llm_call"],
)
llm_guard.attach(agent)

# Both fire at PRE_LLM_CALL — pattern check runs first (registered first)
```

### Tool Whitelist Rail

```python
class ToolWhitelist(Rail):
    def __init__(self, allowed: set[str]):
        super().__init__(name="tool-whitelist", priority=5)
        self._allowed = allowed

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if isinstance(ctx.inputs, ToolCallInputs):
            if ctx.inputs.tool_name not in self._allowed:
                return RailAction.ABORT
        return RailAction.CONTINUE

agent = Agent(
    name="bot",
    tools=[search, calculate, fetch],
    rails=[ToolWhitelist(allowed={"search", "calculate"})],
    # fetch will be blocked by the rail
)
```

### Output Content Filter

```python
class OutputFilter(Rail):
    def __init__(self, blocked_phrases: list[str]):
        super().__init__(name="output-filter", priority=50)
        self._blocked = blocked_phrases

    async def handle(self, ctx: RailContext) -> RailAction | None:
        if ctx.event == HookPoint.FINISHED:
            output = getattr(ctx.inputs, "result", "")
            if any(phrase in str(output).lower() for phrase in self._blocked):
                return RailAction.ABORT
        return RailAction.CONTINUE
```

## Gotchas

- **Only HIGH and CRITICAL block** in BaseGuardrail — LOW and MEDIUM are detected but pass through
- **LLMGuardrailBackend fails open** — parsing errors or API failures return `RiskLevel.SAFE`
- **Pattern confidence scales** — `min(1.0, matched_count * 0.5)`. Multiple pattern matches increase confidence.
- **All patterns are case-insensitive** (compiled with `re.IGNORECASE`)
- **Rails and BaseGuardrails coexist** — Rails via `Agent(rails=[...])`, BaseGuardrails via `.attach(agent)`. Both register as hooks.
- **Rail priority ordering** — lower numbers run first. Use 1-10 for critical safety, 50 for default, 90+ for logging.
- **Cross-rail state** — the `ctx.extra` dict is shared across all rails in a single hook invocation. One rail can pass data to a later rail.
- **`RailAbortError`** is raised when any rail returns `ABORT` — catch it if you need graceful handling.
- **Backend `data` dict** includes `data["event"]` (hook point name) plus all hook kwargs — use this to write event-specific detection logic.
