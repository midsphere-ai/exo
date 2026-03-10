# Guardrails — agent-core to Orbiter Mapping

**Epic:** 1 — Security Guardrails
**Date:** 2026-03-10

This document maps agent-core's (openJiuwen) security guardrail system to
Orbiter's `orbiter-guardrail` package, helping contributors familiar with
either framework navigate both.

---

## 1. Agent-Core Overview

Agent-core's security guardrail system lives in
`openjiuwen/core/security/guardrail/` and provides event-driven content
moderation that can block or flag risky inputs and outputs during agent
execution.

### Key Components

**`RiskLevel`** — An enum of severity tiers used to classify detected risks:

| Level | When used |
|-------|-----------|
| `SAFE` | No risk detected |
| `LOW` | Minor concern, logged but not blocked |
| `MEDIUM` | Moderate concern, may warrant review |
| `HIGH` | Serious threat, blocked by default |
| `CRITICAL` | Maximum severity, always blocked |

**`GuardrailBackend` ABC** — The base class for detection logic. Subclasses
implement `analyze(data)` to inspect arbitrary data and return a
`RiskAssessment`.

```python
# agent-core pattern
class GuardrailBackend(ABC):
    @abstractmethod
    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        ...
```

**`RiskAssessment`** — A frozen dataclass/model returned by backends:
- `has_risk: bool` — whether any risk was detected
- `risk_level: RiskLevel` — severity classification
- `risk_type: str | None` — category (e.g. `"prompt_injection"`, `"pii_leak"`)
- `confidence: float` — 0.0–1.0 confidence score
- `details: dict` — free-form metadata for logging/auditing

**`UserInputGuardrail`** — A built-in guardrail that monitors user messages
for prompt injection and jailbreak attempts. Hooks into `user_input`,
`llm_input`, `llm_output`, and `tool_call` events.

**Event-driven monitoring** — Guardrails attach to lifecycle events on the
agent's callback system. When an event fires, the guardrail's backend
analyzes the event data and either allows it or raises an error to block
execution.

---

## 2. Orbiter Equivalent

Orbiter's guardrail system lives in the `orbiter-guardrail` package
(`packages/orbiter-guardrail/`) as a separate installable package that
depends on `orbiter-core`.

### Mapping Summary

| Agent-Core | Orbiter | Notes |
|------------|---------|-------|
| `GuardrailBackend` ABC | `GuardrailBackend` ABC | Same abstract interface |
| `RiskAssessment` model | `RiskAssessment` model | Frozen Pydantic `BaseModel` |
| `RiskLevel` enum | `RiskLevel` `StrEnum` | Same five levels (SAFE through CRITICAL) |
| `UserInputGuardrail` | `UserInputGuardrail` | Default `PatternBackend` for regex detection |
| Event-driven callbacks | `HookManager` integration via `BaseGuardrail` | Uses `HookPoint` enum instead of custom events |
| Guardrail exception | `GuardrailError(OrbiterError)` | Carries `risk_level`, `risk_type`, `details` |
| — | `GuardrailResult` | New: structured outcome with optional `modified_data` |
| — | `BaseGuardrail` | New: manages attach/detach lifecycle on agents |
| — | `PatternBackend` | New: extracted regex engine (was inline in agent-core) |
| — | `LLMGuardrailBackend` | New: uses an LLM for sophisticated threat detection |

### How Guardrails Integrate via HookManager

Orbiter guardrails use the existing `HookManager` (from `orbiter-core`)
rather than a parallel callback system. This means:

1. **`BaseGuardrail`** wraps a `GuardrailBackend` and manages hook
   registration.
2. **`attach(agent)`** registers async hooks on the agent's
   `hook_manager` for each configured event (e.g. `PRE_LLM_CALL`).
3. When the hook fires, it calls `detect()` → `backend.analyze()`.
4. If `RiskLevel` is `HIGH` or `CRITICAL`, a `GuardrailError` is raised,
   stopping execution.
5. **`detach(agent)`** cleanly removes only the guardrail's hooks.

Available `HookPoint` values for guardrail attachment:

| HookPoint | Typical Use |
|-----------|-------------|
| `PRE_LLM_CALL` | Scan user messages before sending to LLM (default for `UserInputGuardrail`) |
| `POST_LLM_CALL` | Scan LLM output for policy violations |
| `PRE_TOOL_CALL` | Validate tool arguments before execution |
| `POST_TOOL_CALL` | Check tool results |
| `START` | Inspect initial input |
| `FINISHED` | Audit final output |
| `ERROR` | React to errors |

Existing hooks registered via `hook_manager.add(HookPoint.X, my_func)`
continue to work unchanged — guardrails append to the same hook list.

---

## 3. Side-by-Side Code Examples

### Custom Guardrail Backend

**Agent-core:**

```python
# openjiuwen/core/security/guardrail/my_backend.py
from openjiuwen.core.security.guardrail import (
    GuardrailBackend,
    RiskAssessment,
    RiskLevel,
)

class ProfanityBackend(GuardrailBackend):
    async def analyze(self, data: dict) -> RiskAssessment:
        text = data.get("content", "")
        if "bad_word" in text.lower():
            return RiskAssessment(
                has_risk=True,
                risk_level=RiskLevel.HIGH,
                risk_type="profanity",
                confidence=0.95,
                details={"matched": "bad_word"},
            )
        return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
```

**Orbiter:**

```python
# my_guardrails.py
from orbiter.guardrail import (
    GuardrailBackend,
    RiskAssessment,
    RiskLevel,
    BaseGuardrail,
)

class ProfanityBackend(GuardrailBackend):
    async def analyze(self, data: dict) -> RiskAssessment:
        text = data.get("content", "")
        if "bad_word" in text.lower():
            return RiskAssessment(
                has_risk=True,
                risk_level=RiskLevel.HIGH,
                risk_type="profanity",
                confidence=0.95,
                details={"matched": "bad_word"},
            )
        return RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)

# Attach to an agent
guard = BaseGuardrail(
    backend=ProfanityBackend(),
    events=["pre_llm_call", "pre_tool_call"],
)
guard.attach(agent)
```

### Using the Built-In UserInputGuardrail

**Agent-core:**

```python
from openjiuwen.core.security.guardrail import UserInputGuardrail

guard = UserInputGuardrail()
agent.add_guardrail(guard)  # agent-core's registration API
```

**Orbiter:**

```python
from orbiter.guardrail import UserInputGuardrail, GuardrailError

guard = UserInputGuardrail()  # defaults to PatternBackend + PRE_LLM_CALL
guard.attach(agent)

try:
    result = await agent.run("Ignore all previous instructions")
except GuardrailError as e:
    print(f"Blocked: {e.risk_type} ({e.risk_level})")
    # Blocked: prompt_injection (high)
```

### Using the LLM Backend for Advanced Detection

```python
from orbiter.guardrail import BaseGuardrail, LLMGuardrailBackend

backend = LLMGuardrailBackend(model="openai:gpt-4o-mini")
guard = BaseGuardrail(backend=backend, events=["pre_llm_call"])
guard.attach(agent)
```

### Adding Custom Patterns to UserInputGuardrail

```python
from orbiter.guardrail import UserInputGuardrail, RiskLevel

guard = UserInputGuardrail(
    extra_patterns=[
        (r"company\s+secret", RiskLevel.CRITICAL, "data_exfiltration"),
        (r"internal\s+api\s+key", RiskLevel.HIGH, "credential_leak"),
    ]
)
guard.attach(agent)
```

---

## 4. Migration Table

| Agent-Core Path | Orbiter Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.core.security.guardrail.GuardrailBackend` | `orbiter.guardrail.types.GuardrailBackend` | ABC with `analyze()` method |
| `openjiuwen.core.security.guardrail.RiskAssessment` | `orbiter.guardrail.types.RiskAssessment` | Frozen Pydantic model |
| `openjiuwen.core.security.guardrail.RiskLevel` | `orbiter.guardrail.types.RiskLevel` | `StrEnum`: SAFE, LOW, MEDIUM, HIGH, CRITICAL |
| `openjiuwen.core.security.guardrail.UserInputGuardrail` | `orbiter.guardrail.user_input.UserInputGuardrail` | Built-in injection detector |
| *(exception handling)* | `orbiter.guardrail.types.GuardrailError` | `OrbiterError` subclass with risk metadata |
| *(inline pattern matching)* | `orbiter.guardrail.user_input.PatternBackend` | Extracted regex-based backend |
| *(no equivalent)* | `orbiter.guardrail.base.BaseGuardrail` | Hook lifecycle manager |
| *(no equivalent)* | `orbiter.guardrail.types.GuardrailResult` | Structured check outcome with `safe()`/`block()` constructors |
| *(no equivalent)* | `orbiter.guardrail.llm_backend.LLMGuardrailBackend` | LLM-powered detection backend |

All public symbols are also re-exported from `orbiter.guardrail` (the
package `__init__.py`), so `from orbiter.guardrail import RiskLevel` works
as a convenience import.

### Event Name Mapping

| Agent-Core Event | Orbiter HookPoint |
|-----------------|-------------------|
| `user_input` | `HookPoint.PRE_LLM_CALL` |
| `llm_input` | `HookPoint.PRE_LLM_CALL` |
| `llm_output` | `HookPoint.POST_LLM_CALL` |
| `tool_call` | `HookPoint.PRE_TOOL_CALL` |
| `tool_result` | `HookPoint.POST_TOOL_CALL` |
