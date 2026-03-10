# Guardrail Framework Design — Pluggable Security Detection

**Status:** Proposed
**Epic:** 1 — Security Guardrail Framework
**Package:** New `orbiter-guardrail` (depends on `orbiter-core`)
**Date:** 2026-03-10

---

## 1. Motivation

Orbiter agents execute LLM calls and tool invocations without any built-in
security screening. While the web UI has safety evaluation prompts
(`services/safety.py`), there is no framework-level mechanism to:

- **Detect prompt injection** before user input reaches the LLM.
- **Assess risk** of tool calls before execution.
- **Block or modify** dangerous content at runtime.
- **Swap detection backends** (regex patterns, LLM-based analysis, external
  APIs) without changing agent code.

Agent-core's guardrail module (`openjiuwen/core/security/guardrail/`) provides
a pluggable architecture with `RiskAssessment` levels, backend-agnostic
`GuardrailBackend` protocol, event-driven hook integration, and built-in
injection detection. This document designs Orbiter's equivalent, integrating
with the existing `HookManager` and the new `RailManager` from Epic 6.

---

## 2. Agent-Core Reference Architecture

Agent-core's guardrail system consists of five modules:

| Module | Purpose |
|--------|---------|
| `enums.py` | `RiskLevel` enum: SAFE, LOW, MEDIUM, HIGH, CRITICAL |
| `models.py` | `RiskAssessment` data model with risk metadata |
| `backends.py` | `GuardrailBackend` ABC — pluggable detection interface |
| `guardrail.py` | `BaseGuardrail` — registers with callback framework, runs detection |
| `builtin.py` | `UserInputGuardrail` — pattern-based injection detection |

Key design choices in agent-core:

1. **Backend-agnostic**: Detection logic lives in `GuardrailBackend.analyze()`,
   making it trivial to swap regex patterns for LLM-based analysis.
2. **Event-driven**: Guardrails register on the callback framework's events
   (user_input, llm_input, llm_output, tool_call) with priority=100.
3. **Risk model**: `RiskAssessment` carries `has_risk`, `risk_level`,
   `risk_type`, `confidence`, and `details` — enough metadata for logging,
   auditing, and policy decisions.
4. **Built-in detection**: `UserInputGuardrail` ships with regex patterns for
   common prompt injection and jailbreak attempts.

---

## 3. Key Decision: Guardrails as a Separate Package Using HookManager

### Option A — Guardrails inside orbiter-core (rejected)

Adding guardrail types to `orbiter-core` would couple security concerns with
the core agent loop. Not all users need guardrails, and the dependency on
pattern libraries or LLM backends should be optional.

### Option B — Guardrails as Rails in RailManager (rejected)

While guardrails conceptually share the "lifecycle guard" pattern with rails,
making every guardrail a `Rail` subclass would:

- Force guardrail authors to understand the Rail ABC and RailAction semantics.
- Conflate two concerns: rails control execution flow (SKIP, RETRY, ABORT),
  while guardrails assess risk and block based on policy.
- Make it harder to attach/detach guardrails dynamically at runtime.

### Option C — Separate `orbiter-guardrail` package with HookManager integration (chosen)

A new `orbiter-guardrail` package that:

1. Defines its own type hierarchy (`RiskLevel`, `RiskAssessment`,
   `GuardrailBackend`, `GuardrailResult`, `BaseGuardrail`).
2. Integrates with agents via `HookManager` — guardrails register as hooks
   at specific `HookPoint` values.
3. Can coexist with rails — both register as hooks on the same `HookManager`.
4. Is independently installable (`pip install orbiter-guardrail`).

**Why Option C:**

- Clean separation of concerns — security detection is a distinct domain.
- Independent versioning and optional installation.
- Guardrails use the same `HookManager` integration point as rails, so they
  coexist naturally without special coordination.
- Dynamic attach/detach via `BaseGuardrail.attach(agent)` / `.detach(agent)`.

---

## 4. Type Hierarchy

### 4.1 RiskLevel

```python
class RiskLevel(StrEnum):
    """Severity level of a detected risk."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### 4.2 RiskAssessment

```python
class RiskAssessment(BaseModel, frozen=True):
    """Result of a backend's risk analysis.

    Attributes:
        has_risk: Whether any risk was detected.
        risk_level: Severity of the detected risk.
        risk_type: Category of risk (e.g., "prompt_injection", "pii_leak").
        confidence: Backend's confidence in the assessment (0.0–1.0).
        details: Additional metadata for logging and auditing.
    """
    has_risk: bool
    risk_level: RiskLevel
    risk_type: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)
```

Frozen because assessments are immutable facts — once produced by a backend,
they should not be modified downstream.

### 4.3 GuardrailError

```python
class GuardrailError(OrbiterError):
    """Raised when a guardrail blocks an operation.

    Attributes:
        risk_level: The risk level that triggered the block.
        risk_type: Category of the detected risk.
        details: Additional context from the risk assessment.
    """
    def __init__(
        self,
        message: str,
        *,
        risk_level: RiskLevel,
        risk_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None: ...
```

### 4.4 GuardrailBackend ABC

```python
class GuardrailBackend(ABC):
    """Abstract interface for risk detection logic.

    Implementations analyze event data and return a risk assessment.
    Backends are stateless and reusable across multiple guardrails.
    """
    @abstractmethod
    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        """Analyze event data for security risks.

        Args:
            data: Event-specific data (messages, tool_name, arguments, etc.)

        Returns:
            A RiskAssessment indicating the detected risk level.
        """
        ...
```

### 4.5 GuardrailResult

```python
class GuardrailResult(BaseModel, frozen=True):
    """Outcome of a guardrail check — used by BaseGuardrail.detect().

    Attributes:
        is_safe: Whether the content passed the guardrail check.
        risk_level: Severity if unsafe.
        risk_type: Category of risk if unsafe.
        details: Additional context.
        modified_data: Optional modified version of the input data
            (e.g., with PII redacted). None means no modification.
    """
    is_safe: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    risk_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    modified_data: dict[str, Any] | None = None

    @classmethod
    def safe(cls) -> GuardrailResult:
        """Create a safe result (no risk detected)."""
        return cls(is_safe=True)

    @classmethod
    def block(
        cls,
        risk_level: RiskLevel,
        risk_type: str,
        details: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Create a blocking result (risk detected)."""
        return cls(
            is_safe=False,
            risk_level=risk_level,
            risk_type=risk_type,
            details=details or {},
        )
```

### 4.6 BaseGuardrail

```python
class BaseGuardrail:
    """Base class for guardrails that integrate with Agent via HookManager.

    A guardrail wraps a GuardrailBackend and registers itself as hooks on
    an agent's HookManager for the specified events. When those hooks fire,
    the guardrail calls the backend to assess risk and raises GuardrailError
    if the risk level meets or exceeds the blocking threshold.

    Attributes:
        name: Human-readable identifier.
        backend: Optional detection backend. If None, detect() returns safe.
        events: List of HookPoint values to attach to.
        block_threshold: Minimum RiskLevel that triggers a block (default: HIGH).
    """

    def __init__(
        self,
        name: str,
        *,
        backend: GuardrailBackend | None = None,
        events: list[HookPoint] | None = None,
        block_threshold: RiskLevel = RiskLevel.HIGH,
    ) -> None: ...

    def attach(self, agent: Agent) -> None:
        """Register guardrail hooks on the agent's hook_manager."""
        ...

    def detach(self, agent: Agent) -> None:
        """Remove guardrail hooks from the agent's hook_manager."""
        ...

    async def detect(self, event: HookPoint, **data: Any) -> GuardrailResult:
        """Run the backend and return a GuardrailResult.

        If no backend is set, returns GuardrailResult.safe().
        """
        ...
```

---

## 5. HookPoint Attachment

Guardrails attach to `HookPoint` values via `HookManager.add()`, just like
rails and plain hooks. The recommended attachment points are:

| HookPoint | Guardrail Use Case |
|-----------|-------------------|
| `PRE_LLM_CALL` | **Primary.** Inspect messages before they reach the LLM. Detect prompt injection, jailbreak, PII in user input. |
| `PRE_TOOL_CALL` | **Primary.** Inspect tool name and arguments before execution. Block dangerous tools or suspicious arguments. |
| `POST_LLM_CALL` | **Secondary.** Inspect LLM response for harmful content, PII leakage, or policy violations before returning to user. |
| `POST_TOOL_CALL` | **Secondary.** Inspect tool results for sensitive data before they enter the conversation. |
| `START` | **Optional.** Validate the initial user input before any processing. |

The minimum recommended set is **`PRE_LLM_CALL`** and **`PRE_TOOL_CALL`** —
these catch risks before they can cause harm (input to LLM, execution of
tools).

---

## 6. Integration with Existing Hooks and Rails

### 6.1 Execution Order

All hooks, rails, and guardrails register on the same `HookManager` via
`HookManager.add()`. Hooks execute sequentially in **registration order**.
This means the execution order depends on when each component calls `add()`:

```
Agent.__init__()
  │
  ├─ 1. Rails registered (if any) — via RailManager.hook_for()
  │     One hook per HookPoint, runs all rails internally by priority.
  │
  ├─ 2. Traditional hooks registered (if any)
  │     Registered via agent constructor's hooks parameter.
  │
  └─ 3. Guardrails attached (post-construction) — via guardrail.attach(agent)
        One hook per event HookPoint.
```

Within a single `HookManager.run()` call, the order is:

```
hook_manager.run(PRE_LLM_CALL, ...)
  │
  ├─ RailManager hook (runs all rails by priority)
  │    ├─ Security rail (priority 10)
  │    ├─ Default rail (priority 50)
  │    └─ Logging rail (priority 90)
  │
  ├─ Traditional hook #1
  ├─ Traditional hook #2
  │
  └─ Guardrail hook (calls backend.analyze())
       └─ If risk >= block_threshold → raises GuardrailError
```

### 6.2 Non-Interference Guarantees

1. **Guardrails do not modify the hook list.** `attach()` appends hooks;
   `detach()` removes only the guardrail's own hooks. Other hooks and rails
   are untouched.

2. **Guardrails do not interact with RailManager.** They are independent
   hooks on the same `HookManager`. A guardrail does not return `RailAction`
   and does not participate in rail priority ordering.

3. **Exception propagation is consistent.** If a guardrail raises
   `GuardrailError`, it propagates through `HookManager.run()` exactly like
   `RailAbortError` — the agent run stops. Both inherit from `OrbiterError`.

4. **No performance impact when absent.** If no guardrails are attached,
   zero additional hooks are registered.

### 6.3 Guardrails vs. Rails — When to Use Which

| Concern | Use Rails | Use Guardrails |
|---------|-----------|---------------|
| Typed lifecycle interception | Yes | No |
| Priority-ordered execution | Yes (via RailManager) | No (registration order) |
| Cross-guard state sharing | Yes (via `extra` dict) | No |
| Risk assessment with confidence scores | No | Yes |
| Swappable detection backends | No | Yes |
| Dynamic attach/detach at runtime | No (set at construction) | Yes |
| Blocking based on risk policy | Possible (ABORT action) | Primary purpose |

Rails and guardrails are complementary. A security-focused rail (priority 10)
could perform fast checks, while a guardrail with an LLM backend could
perform deeper analysis. Both can coexist on the same agent.

---

## 7. Built-In Guardrails

### 7.1 UserInputGuardrail

```python
class UserInputGuardrail(BaseGuardrail):
    """Detects prompt injection and jailbreak in user messages.

    Uses PatternBackend by default. Attaches to PRE_LLM_CALL.
    """
    def __init__(
        self,
        *,
        patterns: list[str] | None = None,
        backend: GuardrailBackend | None = None,
    ) -> None:
        # If no backend provided, use built-in PatternBackend
        super().__init__(
            name="user_input",
            backend=backend or PatternBackend(patterns=patterns),
            events=[HookPoint.PRE_LLM_CALL],
        )
```

### 7.2 PatternBackend

```python
class PatternBackend(GuardrailBackend):
    """Regex-based detection backend for common injection patterns.

    Configurable pattern list. Checks the latest user message in
    data["messages"] against compiled regex patterns.
    """
    DEFAULT_PATTERNS: ClassVar[list[str]] = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"forget\s+(?:all\s+)?(?:your|previous)\s+",
        r"system\s*prompt",
        r"act\s+as\s+(?:if|though)\s+you",
        r"pretend\s+(?:you\s+are|to\s+be)\s+",
        r"do\s+not\s+follow\s+(?:any|your)\s+",
        r"override\s+(?:your|all)\s+",
    ]

    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        """Check latest user message against injection patterns."""
        ...
```

### 7.3 LLMGuardrailBackend

```python
class LLMGuardrailBackend(GuardrailBackend):
    """LLM-powered detection for sophisticated content analysis.

    Uses an LLM to assess risk based on a configurable prompt template.
    Parses structured JSON response into RiskAssessment.
    """
    def __init__(
        self,
        model: str,  # provider:model format
        *,
        prompt_template: str | None = None,  # Uses default if None
    ) -> None: ...

    async def analyze(self, data: dict[str, Any]) -> RiskAssessment:
        """Format data into prompt, call LLM, parse response."""
        ...
```

---

## 8. Event Flow Diagram

```
Agent.run(input)
  │
  ├─ hook_manager.run(START, ...)
  │    ├─ [RailManager hook → sorted rails]
  │    ├─ [plain hooks]
  │    └─ [guardrail hooks (if attached to START)]
  │
  ├─ Agent._call_llm()
  │    ├─ hook_manager.run(PRE_LLM_CALL, messages=..., tools=...)
  │    │    ├─ [RailManager hook → sorted rails]
  │    │    ├─ [plain hooks]
  │    │    └─ [UserInputGuardrail hook]
  │    │         ├─ PatternBackend.analyze(data)
  │    │         │    ├─ No match → GuardrailResult.safe() → proceed
  │    │         │    └─ Match → RiskAssessment(HIGH) → GuardrailError ✘
  │    │         └─ (or LLMGuardrailBackend.analyze(data))
  │    │
  │    ├─ provider.complete(...)  ← only reached if all hooks pass
  │    │
  │    └─ hook_manager.run(POST_LLM_CALL, response=...)
  │         ├─ [RailManager hook → sorted rails]
  │         ├─ [plain hooks]
  │         └─ [guardrail hooks (if attached to POST_LLM_CALL)]
  │
  ├─ Agent._execute_tools()
  │    ├─ hook_manager.run(PRE_TOOL_CALL, tool_name=..., arguments=...)
  │    │    ├─ [RailManager hook → sorted rails]
  │    │    ├─ [plain hooks]
  │    │    └─ [guardrail hooks (if attached to PRE_TOOL_CALL)]
  │    │         └─ Backend.analyze({tool_name, arguments})
  │    │              ├─ Safe → proceed to tool execution
  │    │              └─ Risk >= threshold → GuardrailError ✘
  │    │
  │    ├─ tool.execute(...)
  │    │
  │    └─ hook_manager.run(POST_TOOL_CALL, result=...)
  │         └─ [guardrail hooks (if attached to POST_TOOL_CALL)]
  │
  └─ hook_manager.run(FINISHED, ...)
       └─ [all hooks including any guardrails]
```

---

## 9. Package Layout

```
packages/orbiter-guardrail/
├── pyproject.toml          # hatchling, depends on orbiter-core
├── src/
│   └── orbiter/
│       ├── __init__.py     # extend_path for namespace package
│       └── guardrail/
│           ├── __init__.py # Public exports
│           ├── types.py    # RiskLevel, RiskAssessment, GuardrailResult, GuardrailError
│           ├── backend.py  # GuardrailBackend ABC
│           ├── base.py     # BaseGuardrail
│           ├── builtin.py  # PatternBackend, UserInputGuardrail
│           └── llm.py      # LLMGuardrailBackend
└── tests/
    ├── test_guardrail_types.py
    ├── test_backend.py
    ├── test_base_guardrail.py
    ├── test_user_input_guardrail.py
    ├── test_llm_backend.py
    └── test_integration.py
```

---

## 10. Interaction Summary

### How guardrails preserve backward compatibility

1. **No changes to HookManager.** Guardrails use the existing public API
   (`add`, `remove`, `run`). No modifications to `hooks.py`.

2. **No changes to RailManager or Rail.** Guardrails are independent of the
   rail system. They happen to register on the same `HookManager` but do not
   import or depend on `rail.py`.

3. **No changes to Agent.** Guardrails attach post-construction via
   `guardrail.attach(agent)`. The `Agent` class does not need to know about
   guardrails — it only sees additional hooks on its `hook_manager`.

4. **All existing tests pass.** Since no existing code is modified,
   all ~2,900 tests remain unaffected.

5. **Optional dependency.** `orbiter-guardrail` is a separate pip-installable
   package. Projects that don't need guardrails don't need to install it.

---

## 11. Open Questions

1. **Block vs. modify.** `GuardrailResult.modified_data` supports content
   modification (e.g., PII redaction). Should `BaseGuardrail` automatically
   apply modifications to the hook data, or just log them?
   **Recommendation:** Log only in v1; apply modifications in a follow-up
   story if users need it.

2. **Multiple backends per guardrail.** Should a single guardrail support
   chaining multiple backends (e.g., pattern check first, then LLM if
   uncertain)? **Recommendation:** Use composition — create a
   `ChainedBackend` that wraps multiple backends. Defer to a future story.

3. **Guardrail ordering.** Since guardrails use registration order (not
   priority), should we add a priority field to `BaseGuardrail`?
   **Recommendation:** Not needed for v1. If priority ordering is needed,
   users can convert their guardrail into a Rail with priority semantics.

4. **Async context for LLMGuardrailBackend.** The LLM backend needs a
   provider instance. Should it create one per call or accept a reusable
   provider? **Recommendation:** Accept `model: str` and resolve the
   provider internally using `get_provider()`, consistent with agent-core's
   approach.
