# Workflow Extension Design — Loops, Branches, and Sub-Workflows for Swarm

**Status:** Proposed
**Epic:** 2 — Advanced Workflow Engine
**Date:** 2026-03-10

---

## 1. Motivation

Orbiter's `Swarm` provides clean multi-agent orchestration with three modes:

- **Workflow** — agents execute in topological order, chaining output→input.
- **Handoff** — agent-driven delegation via declared handoff targets.
- **Team** — lead-worker delegation via auto-generated tools.

The flow DSL (`"a >> b >> c"`, `"(a | b) >> c"`) is concise for linear and
fan-out/fan-in patterns.  However, it cannot express:

1. **Loops** — repeating an agent or group N times, over a collection, or until
   a condition is met.
2. **Conditional branches** — routing to different agents based on runtime state.
3. **Sub-workflows** — embedding a named, reusable workflow as a single step.
4. **Shared workflow state** — a mutable dict accessible by all nodes during
   execution for inter-node data passing.

Agent-core (`openjiuwen/core/workflow/components/`) addresses these with
`LoopComponent`, `BranchComponent`, `SubWorkflowComponent`, and a safe
expression evaluator.  This document designs equivalent extensions for Orbiter's
Swarm that preserve full backward compatibility.

---

## 2. Current Architecture Summary

### 2.1 Swarm (664 lines)

| Element | Description |
|---------|-------------|
| `Swarm.__init__()` | Indexes agents by name, parses flow DSL, resolves topological order |
| `Swarm.run()` | Dispatches to `_run_workflow`, `_run_handoff`, `_run_team` |
| `Swarm.stream()` | Dispatches to `_stream_workflow`, `_stream_handoff`, `_stream_team` |
| `_run_workflow()` | Iterates `flow_order`, chains output→input, delegates to groups/nested swarms |
| `_detect_handoff()` | Matches stripped output against `agent.handoffs` dict keys |
| `_DelegateTool` | Auto-generated tool for team-mode worker delegation |

### 2.2 Graph (165 lines)

| Element | Description |
|---------|-------------|
| `Graph` | Directed adjacency list, `add_node`, `add_edge`, `successors`, `in_degree` |
| `topological_sort()` | Kahn's algorithm with cycle detection |
| `parse_flow_dsl()` | Tokenizes `>>` stages, detects `(a \| b)` parallel groups |

### 2.3 Nested / SwarmNode (107 lines)

| Element | Description |
|---------|-------------|
| `SwarmNode` | Wraps a `Swarm` as a node; `is_swarm = True` marker for duck-typing |
| Context isolation | Inner swarm gets fresh message history; no state leaks |

### 2.4 Agent Groups (237 lines)

| Element | Description |
|---------|-------------|
| `ParallelGroup` | Concurrent execution with `asyncio.TaskGroup`; join outputs |
| `SerialGroup` | Sequential execution with output→input chaining |

### 2.5 RunState (170 lines)

| Element | Description |
|---------|-------------|
| `RunState` | Mutable execution state: messages, nodes, iterations, usage |
| `RunNode` | Per-step tracking with status, timing, tool calls, usage |

---

## 3. Design Overview

### 3.1 Core Principle: Extend, Don't Replace

All extensions integrate into the existing Swarm execution model via the same
duck-typing pattern used by `ParallelGroup`, `SerialGroup`, and `SwarmNode`:

- New node types have a `name` attribute and `run()` / `stream()` methods.
- Boolean markers (`is_loop`, `is_branch`, `is_subworkflow`) allow the Swarm
  execution loop to detect them.
- Existing modes (workflow, handoff, team) remain **completely unchanged**.
- The flow DSL continues to work as-is; new node types are just agents in the
  flow.

### 3.2 New Components

```
orbiter/_internal/
├── expression.py          # Safe expression evaluator (~200 lines)
├── branch_node.py         # BranchNode for conditional routing (~150 lines)
├── loop_node.py           # LoopNode for iteration (~200 lines)
├── workflow_state.py      # WorkflowState shared dict (~80 lines)
└── ... (existing files unchanged)
```

### 3.3 Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                         Swarm                             │
│  flow_order: [a, loop1, branch1, c]                       │
│                                                           │
│  _run_workflow() iterates flow_order:                      │
│    ┌─────┐   ┌──────────┐   ┌────────────┐   ┌─────┐     │
│    │  a  │──▶│ LoopNode │──▶│ BranchNode │──▶│  c  │     │
│    └─────┘   └──────────┘   └────────────┘   └─────┘     │
│                   │                │                       │
│        iterates inner       evaluates condition            │
│        agent N times        routes to true/false agent     │
│                                                           │
│  WorkflowState: shared mutable dict across all nodes       │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Design

### 4.1 WorkflowState — Shared Execution Context

A simple mutable dict that flows through all nodes during workflow execution,
enabling inter-node data passing.

```python
# orbiter/_internal/workflow_state.py

from __future__ import annotations
from typing import Any

class WorkflowState:
    """Shared mutable state for workflow execution.

    All nodes in a workflow can read and write to this state dict.
    The state persists across the full workflow run.

    Args:
        initial: Optional initial state values.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial) if initial else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state."""
        self._data[key] = value

    def update(self, values: dict[str, Any]) -> None:
        """Merge values into state."""
        self._data.update(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the state dict."""
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"WorkflowState({self._data!r})"
```

**Integration with Swarm:** The state is created at the start of `_run_workflow()`
and passed to each node's `run()` via an optional `state` keyword argument.
Existing agents ignore the unknown kwarg (or we check for a marker before passing).

```python
# In Swarm._run_workflow(), changed lines:
state = WorkflowState()
for agent_name in self.flow_order:
    agent = self.agents[agent_name]
    if getattr(agent, "accepts_state", False):
        result = await agent.run(current_input, state=state, ...)
    else:
        # Existing behavior unchanged
        result = await call_runner(agent, current_input, ...)
    current_input = result.output
    # Auto-populate state with agent outputs
    state.set(f"{agent_name}.output", result.output)
```

### 4.2 Safe Expression Evaluator

A sandboxed evaluator for conditions in branches and loops.  Uses Python's
`ast.parse(mode='eval')` — **never** calls `eval()`.

```python
# orbiter/_internal/expression.py

import ast
from typing import Any

MAX_EXPRESSION_LENGTH = 500
MAX_AST_DEPTH = 10
MAX_COLLECTION_SIZE = 1000

SAFE_FUNCTIONS = {"len", "range", "str", "int", "float", "bool", "abs", "min", "max"}

class ExpressionError(Exception):
    """Raised for invalid or unsafe expressions."""

def evaluate(expression: str, variables: dict[str, Any] | None = None) -> Any:
    """Safely evaluate an expression string.

    Supports:
        - Boolean: and, or, not (also &&, ||, !)
        - Comparisons: ==, !=, <, >, <=, >=
        - Arithmetic: +, -, *, /, %, **
        - Subscripting: dict["key"], list[0]
        - Allowlisted functions: len(), str(), int(), etc.
        - Variable resolution from the variables dict

    Security:
        - AST-based parsing, no eval()
        - Blocks dunder access, imports, lambdas
        - Depth and length limits
        - No attribute access on arbitrary objects

    Args:
        expression: String expression to evaluate.
        variables: Variable context for name resolution.

    Returns:
        The evaluated result.

    Raises:
        ExpressionError: If the expression is invalid or unsafe.
    """
```

**Normalization rules** (applied before parsing):

| Input | Normalized |
|-------|-----------|
| `&&` | `and` |
| `\|\|` | `or` |
| `!expr` | `not expr` |
| `true` / `false` | `True` / `False` |

**Blocked constructs:** dunder access (`__class__`, `__bases__`), import
statements, function calls not in allowlist, lambda expressions, starred
expressions, assignment expressions (`:=`).

### 4.3 LoopNode — Iteration

Three loop modes matching agent-core's `LoopComponent`:

```python
# orbiter/_internal/loop_node.py

from __future__ import annotations
from typing import Any
from orbiter.types import RunResult

class LoopNode:
    """Iterates an inner agent with three loop modes.

    Modes:
        count: Execute the inner agent N times.
        items: Iterate over an array (from state or literal).
        condition: Repeat while an expression evaluates to truthy.

    The inner agent receives the current iteration context as input.
    Outputs from each iteration are collected and joined.

    Args:
        name: Node name for the flow DSL.
        agent: The agent to execute on each iteration.
        count: Fixed iteration count (count mode).
        items: List or state key resolving to a list (items mode).
        condition: Expression string evaluated each iteration (condition mode).
        max_iterations: Safety limit (default 100).
        separator: String to join iteration outputs.

    Duck-type markers:
        is_loop = True
        accepts_state = True
    """

    def __init__(
        self,
        *,
        name: str,
        agent: Any,
        count: int | None = None,
        items: list[Any] | str | None = None,
        condition: str | None = None,
        max_iterations: int = 100,
        separator: str = "\n",
    ) -> None: ...

    is_loop = True
    accepts_state = True

    async def run(
        self,
        input: str,
        *,
        state: WorkflowState | None = None,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the loop.

        Count mode: runs agent `count` times.
        Items mode: resolves items list, runs agent once per item.
        Condition mode: evaluates expression before each iteration.

        Break control: if agent output contains "[BREAK]", loop terminates early.
        The "[BREAK]" marker is stripped from the final output.

        State updates per iteration:
            state["loop.index"] = current iteration index (0-based)
            state["loop.value"] = current item (items mode only)
            state["loop.output"] = previous iteration's output
        """
```

**Execution flow (count mode):**
```
outputs = []
for i in range(count):
    state["loop.index"] = i
    state["loop.output"] = outputs[-1] if outputs else ""
    result = await call_runner(agent, current_input, ...)
    if "[BREAK]" in result.output:
        outputs.append(result.output.replace("[BREAK]", "").strip())
        break
    outputs.append(result.output)
    current_input = result.output  # chain iterations
return RunResult(output=separator.join(outputs))
```

**Execution flow (condition mode):**
```
i = 0
while i < max_iterations:
    if not evaluate(condition, state.to_dict()):
        break
    state["loop.index"] = i
    result = await call_runner(agent, current_input, ...)
    if "[BREAK]" in result.output:
        break
    i += 1
```

### 4.4 BranchNode — Conditional Routing

```python
# orbiter/_internal/branch_node.py

from __future__ import annotations
from typing import Any
from orbiter.types import RunResult

class BranchNode:
    """Conditional routing between two agents.

    Evaluates a condition (expression string or callable) and routes
    to either `true_agent` or `false_agent`.

    Args:
        name: Node name for the flow DSL.
        condition: Expression string or callable (receives state dict, returns bool).
        true_agent: Agent to execute when condition is truthy.
        false_agent: Agent to execute when condition is falsy (optional).

    If false_agent is None and condition is falsy, returns RunResult
    with empty output (passthrough).

    Duck-type markers:
        is_branch = True
        accepts_state = True
    """

    def __init__(
        self,
        *,
        name: str,
        condition: str | Callable[[dict[str, Any]], bool],
        true_agent: Any,
        false_agent: Any | None = None,
    ) -> None: ...

    is_branch = True
    accepts_state = True

    async def run(
        self,
        input: str,
        *,
        state: WorkflowState | None = None,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Evaluate condition and route to the appropriate agent.

        For string conditions, uses the safe expression evaluator with
        the current WorkflowState as variable context.

        For callable conditions, passes state.to_dict().
        """
```

**Execution flow:**
```
if isinstance(condition, str):
    result = evaluate(condition, state.to_dict() if state else {})
else:
    result = condition(state.to_dict() if state else {})

if result:
    return await call_runner(true_agent, input, ...)
elif false_agent is not None:
    return await call_runner(false_agent, input, ...)
else:
    return RunResult(output=input)  # passthrough
```

### 4.5 Sub-Workflows (Already Supported)

Orbiter already has `SwarmNode` for nested swarms.  The only enhancement needed:

1. **State forwarding** — optionally pass parent `WorkflowState` to inner swarm
   (currently uses context isolation, which remains the default).
2. **Input/output mapping** — optional dict-based mapping of state keys.

```python
# Enhancement to SwarmNode:

class SwarmNode:
    def __init__(
        self,
        *,
        swarm: Any,
        name: str | None = None,
        input_mapping: dict[str, str] | None = None,   # NEW
        output_mapping: dict[str, str] | None = None,   # NEW
        share_state: bool = False,                       # NEW
    ) -> None: ...
```

When `share_state=True`, the inner swarm receives the parent `WorkflowState`.
When `input_mapping` is provided, mapped state values are passed as inner
swarm's initial state.  When `output_mapping` is provided, inner swarm's state
values are written back to parent state after execution.

Default behavior (context isolation) is **unchanged**.

---

## 5. DSL Extensions

The flow DSL syntax does **not** change.  Loop, branch, and sub-workflow nodes
are just agents in the flow — they're defined in Python and placed in the
agent list:

```python
# No DSL changes needed — nodes are agents
loop = LoopNode(name="retry", agent=validator, count=3)
branch = BranchNode(
    name="router",
    condition='score > 0.8',
    true_agent=fast_path,
    false_agent=slow_path,
)
sub = SwarmNode(swarm=inner_swarm, name="sub_pipeline")

swarm = Swarm(
    agents=[preprocessor, loop, branch, sub, postprocessor],
    flow="preprocessor >> retry >> router >> sub_pipeline >> postprocessor",
)
```

This is intentional — the DSL remains a topology specification while
nodes define their own execution semantics.

---

## 6. Swarm Execution Loop Changes

### 6.1 `_run_workflow()` Changes

The existing workflow loop needs minimal changes to support the new node types:

```python
async def _run_workflow(self, input, *, messages=None, provider=None, max_retries=3):
    state = WorkflowState()          # NEW: shared state
    current_input = input
    last_result = None

    for agent_name in self.flow_order:
        agent = self.agents[agent_name]

        # NEW: detect state-aware nodes
        if getattr(agent, "accepts_state", False):
            last_result = await agent.run(
                current_input,
                state=state,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
        elif getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
            # Existing group/nested swarm behavior — UNCHANGED
            last_result = await agent.run(
                current_input, messages=messages,
                provider=provider, max_retries=max_retries,
            )
        else:
            # Existing agent behavior — UNCHANGED
            last_result = await call_runner(
                agent, current_input, messages=messages,
                provider=provider, max_retries=max_retries,
            )

        current_input = last_result.output
        state.set(f"{agent_name}.output", current_input)   # NEW: auto-populate

    assert last_result is not None
    return last_result
```

### 6.2 `_stream_workflow()` Changes

Analogous changes to `_stream_workflow()` for streaming support.  State-aware
nodes that support streaming yield events directly; otherwise, fall back to
`run()` and emit the output as a single `TextEvent`.

### 6.3 Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| Swarm with only regular agents | **Identical** — `accepts_state` is False, no state created overhead is negligible |
| Swarm with groups | **Identical** — `is_group` check comes before `accepts_state` |
| Swarm with SwarmNode (no share_state) | **Identical** — context isolation preserved |
| Existing flow DSL strings | **Identical** — parser unchanged |
| Handoff mode | **Unchanged** — no state integration (handoff is conversation-driven) |
| Team mode | **Unchanged** — no state integration (team uses tool delegation) |

---

## 7. Expression Evaluator Security Model

The evaluator is the primary attack surface.  Security constraints:

| Constraint | Limit |
|-----------|-------|
| Expression length | 500 characters |
| AST depth | 10 levels |
| Collection size | 1000 elements |
| Allowed functions | `len`, `range`, `str`, `int`, `float`, `bool`, `abs`, `min`, `max` |
| Blocked | `__dunder__`, imports, lambda, walrus `:=`, `exec`, `eval`, attribute access |

**Implementation approach:** Walk the AST tree node by node, rejecting any node
type not in an explicit allowlist.  This is a whitelist approach (safer than
blacklisting dangerous constructs).

Allowed AST node types:
- `ast.Expression`, `ast.BoolOp`, `ast.BinOp`, `ast.UnaryOp`, `ast.Compare`
- `ast.Constant` (str, int, float, bool, None)
- `ast.Name` (variable lookup from `variables` dict)
- `ast.Subscript`, `ast.Index`, `ast.Slice`
- `ast.List`, `ast.Tuple`, `ast.Dict` (with size limits)
- `ast.Call` (only for allowlisted functions)
- `ast.IfExp` (ternary)

---

## 8. Comparison with Agent-Core

| Feature | Agent-Core | Orbiter (Proposed) |
|---------|-----------|-------------------|
| Loop types | count, items, condition, always_true | count, items, condition (always_true = `condition="True"`) |
| Break control | `[BREAK]` in output | Same |
| Max iterations | Default 100 | Default 100 |
| Branch routing | Expression or callable | Same |
| Expression evaluator | AST-based, safe | Same approach, same limits |
| Sub-workflows | SubWorkflowComponent | SwarmNode (existing) + state mapping |
| Shared state | WorkflowState with commit/rollback | WorkflowState (simpler, no transactions needed yet) |
| Checkpointing | GraphStore save/restore | Not in scope for Epic 2 (add if needed later) |
| Mermaid visualization | Built-in | Not in scope (can add as separate story) |
| DSL | Component-based config | Flow DSL + Python node types |

**Key difference:** Agent-core uses a Pregel graph engine where components are
registered as graph vertices with channel-based communication.  Orbiter uses a
simpler sequential execution model where special nodes (loop, branch) handle
their own internal complexity.  This is simpler and sufficient — Orbiter's
ParallelGroup already handles fan-out, and the topological sort handles DAGs.

---

## 9. Verification: Existing Modes Unmodified

The following existing Swarm capabilities are **NOT modified** by this design:

1. **Workflow mode** — unchanged execution loop, new nodes opt-in via markers.
2. **Handoff mode** — no state integration, conversation-driven delegation
   unchanged.
3. **Team mode** — no state integration, tool-based delegation unchanged.
4. **Flow DSL** — parser unchanged, no new syntax.
5. **ParallelGroup / SerialGroup** — unchanged, work as before.
6. **SwarmNode** — default context isolation unchanged, new `share_state` is
   opt-in.
7. **`_detect_handoff()`** — unchanged.
8. **`_DelegateTool`** — unchanged.
9. **`to_dict()` / `from_dict()`** — will need extension for new node types
   but existing serialization unchanged.

---

## 10. Implementation Plan (User Stories)

| # | Story | New File | Estimated Lines |
|---|-------|----------|----------------|
| US-026 | Safe expression evaluator | `_internal/expression.py` | ~200 |
| US-027 | WorkflowState | `_internal/workflow_state.py` | ~80 |
| US-028 | LoopNode | `_internal/loop_node.py` | ~200 |
| US-029 | BranchNode | `_internal/branch_node.py` | ~150 |
| US-030 | Swarm workflow loop integration | modify `swarm.py` | ~30 changed |
| US-031 | SwarmNode state mapping | modify `nested.py` | ~40 changed |

Total new code: ~630 lines + tests.

---

## 11. Open Questions

1. **Streaming from LoopNode** — Should each iteration's stream events be
   yielded to the caller, or only the final aggregated output?
   **Recommendation:** Yield per-iteration with `StatusEvent` markers for
   iteration boundaries.

2. **State serialization** — Should `WorkflowState` support `to_dict()` /
   `from_dict()` for checkpointing?
   **Recommendation:** Yes, include from the start (it's trivial since the
   underlying data is already a dict).

3. **Multi-branch routing** — Should `BranchNode` support more than two paths
   (switch-case)?
   **Recommendation:** Start with true/false.  Multi-branch can be composed
   via nested `BranchNode`s or added as `SwitchNode` later.

4. **State access in handoff/team modes** — Should `WorkflowState` be
   available in non-workflow modes?
   **Recommendation:** No, keep state scoped to workflow mode.  Handoff mode
   uses conversation history; team mode uses tool delegation.
