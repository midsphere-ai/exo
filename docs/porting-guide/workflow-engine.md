# Workflow Engine — agent-core to Orbiter Mapping

**Epic:** 2 — Advanced Workflow Engine
**Date:** 2026-03-10

This document maps agent-core's (openJiuwen) Pregel graph engine to Orbiter's
extended Swarm system, helping contributors familiar with either framework
navigate both.

---

## 1. Agent-Core Overview

Agent-core's workflow engine lives in `openjiuwen/core/workflow/` and is built
on a **Pregel-style graph engine** — a directed graph of `PregelNode` vertices
connected by typed channels.

### Key Components

**`PregelNode`** — A graph vertex wrapping an agent or component.  Nodes
consume messages from input channels and produce messages on output channels.
The engine routes data between nodes by matching channel names.

**Channels** — Typed message buses that carry data between nodes.  Each channel
has a reducer that merges multiple messages (e.g., last-writer-wins, list
append).  Channels decouple producers from consumers and enable fan-out /
fan-in patterns.

**`LoopComponent`** — Iteration with four modes:

| Mode | Behavior |
|------|----------|
| `count` | Execute inner node N times |
| `items` | Iterate over a collection from workflow state |
| `condition` | Repeat while an expression evaluates to truthy |
| `always_true` | Infinite loop (relies on `[BREAK]` sentinel) |

All modes respect a `max_iterations` safety limit (default 100) and support
early exit via `[BREAK]` in output.

**`BranchComponent`** — Conditional routing to one of two target nodes.
Conditions are either expression strings evaluated by a safe AST-based
evaluator, or callable predicates that receive the workflow state dict.

**`SubWorkflowComponent`** — Embeds a named sub-graph as a single node.
Sub-graphs can share parent state or run in isolation.

**`WorkflowState`** — A shared mutable dict with commit/rollback semantics
for transactional state updates across nodes.

**Safe expression evaluator** — AST-based expression evaluation (never uses
`eval()`) with whitelist of allowed operations, depth limits, and blocked
constructs (`__dunder__`, imports, lambdas).

**Checkpoint/Resume** — `GraphStore` persists node completion status and state
snapshots so that a failed workflow can resume from the last successful node.

**Mermaid visualization** — Built-in rendering of the graph topology as a
Mermaid diagram for debugging and documentation.

---

## 2. Orbiter Equivalent

Orbiter replaces the Pregel graph engine with a simpler sequential execution
model built on the existing `Swarm` orchestrator.  Special node types (loop,
branch) handle their own internal complexity while appearing as regular agents
in the flow.

### Mapping Summary

| Agent-Core | Orbiter | Notes |
|------------|---------|-------|
| `PregelNode` | Regular `Agent` in `Swarm` flow | No separate node wrapper needed |
| Channels (typed message buses) | Output→input chaining in `_run_workflow()` | Simpler — previous agent's output becomes next agent's input |
| `LoopComponent` | `LoopNode` (`_internal/loop_node.py`) | Three modes: count, items, condition (`always_true` = `condition="True"`) |
| `BranchComponent` | `BranchNode` (`_internal/branch_node.py`) | Expression string or callable; routes to `if_true` / `if_false` agent |
| `SubWorkflowComponent` | `SwarmNode` (`_internal/nested.py`) | Wraps a `Swarm` as a node; context isolation by default |
| `WorkflowState` (commit/rollback) | `WorkflowState` (`_internal/workflow_state.py`) | Simpler — no transactions, just a shared mutable dict |
| Safe expression evaluator | `expression.py` (`_internal/expression.py`) | Same AST-based approach with JS-style normalization (`&&`→`and`, `\|\|`→`or`) |
| `GraphStore` checkpoint | `WorkflowCheckpoint` / `WorkflowCheckpointStore` (`_internal/workflow_checkpoint.py`) | Immutable snapshots; in-memory store; `Swarm.resume()` to continue |
| Mermaid visualization | `visualization.py` (`_internal/visualization.py`) / `Swarm.to_mermaid()` | Node-shape conventions: diamond for branch, hexagon for loop |
| `ParallelGroup` (fan-out) | `ParallelGroup` (`_internal/agent_group.py`) | Concurrent execution via `asyncio.TaskGroup`; custom or default aggregation |
| `SerialGroup` (pipeline) | `SerialGroup` (`_internal/agent_group.py`) | Sequential output→input chaining within a group |
| Graph topology | Flow DSL (`"a >> b >> c"`, `"(a \| b) >> c"`) | Parsed into `Graph` adjacency list, topologically sorted via Kahn's algorithm |

### How Workflow Execution Works in Orbiter

Orbiter's `Swarm` uses duck-typing markers to detect special node types during
workflow execution:

| Marker | Node Type | Detection |
|--------|-----------|-----------|
| `is_branch = True` | `BranchNode` | `getattr(agent, "is_branch", False)` |
| `is_loop = True` | `LoopNode` | `getattr(agent, "is_loop", False)` |
| `is_group = True` | `ParallelGroup` / `SerialGroup` | `getattr(agent, "is_group", False)` |
| `is_swarm = True` | `SwarmNode` | `getattr(agent, "is_swarm", False)` |

The execution loop in `Swarm._run_workflow()`:

1. Create `WorkflowState` with initial input
2. For each agent in topological order:
   - Save `WorkflowCheckpoint` (if checkpointing enabled)
   - If `BranchNode`: evaluate condition, route to target agent
   - If `LoopNode`: iterate body agent(s) per mode
   - If `Group` / `SwarmNode`: delegate to their `run()`
   - Otherwise: execute via `call_runner()`
   - Update `WorkflowState` with output
   - Chain output→input to next agent
3. Return final `RunResult`

---

## 3. Code Comparison — Branching Workflow

### Agent-core (Pregel)

```python
from openjiuwen.core.workflow import (
    PregelEngine,
    PregelNode,
    BranchComponent,
    Channel,
)

# Define agents as nodes
classifier = PregelNode("classifier", agent=classify_agent)
fast_path = PregelNode("fast_path", agent=fast_agent)
slow_path = PregelNode("slow_path", agent=slow_agent)
merger = PregelNode("merger", agent=merge_agent)

# Branch component reads from channel, routes by expression
branch = BranchComponent(
    name="router",
    condition="confidence > 0.8",
    true_target="fast_path",
    false_target="slow_path",
)

# Wire channels
engine = PregelEngine()
engine.add_node(classifier)
engine.add_node(branch)
engine.add_node(fast_path)
engine.add_node(slow_path)
engine.add_node(merger)

engine.add_channel("classifier", "router", Channel("classify_out"))
engine.add_channel("fast_path", "merger", Channel("fast_out"))
engine.add_channel("slow_path", "merger", Channel("slow_out"))

result = await engine.run("Analyze this document")
```

### Orbiter (Swarm)

```python
from orbiter import Agent, Swarm
from orbiter._internal.branch_node import BranchNode

# Define agents
classifier = Agent(name="classifier", instructions="Classify input confidence")
fast_agent = Agent(name="fast_path", instructions="Fast processing")
slow_agent = Agent(name="slow_path", instructions="Thorough processing")
merger = Agent(name="merger", instructions="Merge results")

# Branch node — just another agent in the flow
router = BranchNode(
    name="router",
    condition="confidence > 0.8",
    if_true="fast_path",
    if_false="slow_path",
)

# Flow DSL defines topology — no manual channel wiring
swarm = Swarm(
    agents=[classifier, router, fast_agent, slow_agent, merger],
    flow="classifier >> router >> merger",
)

result = await swarm.run("Analyze this document")
```

**Key differences:**

1. **No channels** — Orbiter chains output→input directly; no need to define
   and wire typed message buses between nodes.
2. **Flow DSL** — Topology is a string (`"a >> b >> c"`), not manual
   `add_node` / `add_channel` calls.
3. **Nodes are agents** — `BranchNode` is placed in the agent list and
   referenced by name in the flow string.  The Swarm detects it via
   `is_branch = True` and delegates routing.
4. **Condition evaluation** — Both use AST-based safe expression evaluation
   against workflow state.  Orbiter additionally normalizes JS-style operators
   (`&&`, `||`, `===`).

### Loop Example Comparison

**Agent-core:**

```python
loop = LoopComponent(
    name="refiner",
    mode="count",
    count=3,
    inner_node=PregelNode("refine", agent=refine_agent),
    max_iterations=10,
)
engine.add_node(loop)
```

**Orbiter:**

```python
from orbiter._internal.loop_node import LoopNode

loop = LoopNode(
    name="refiner",
    body="refine",
    count=3,
    max_iterations=10,
)

swarm = Swarm(
    agents=[preprocessor, loop, refine_agent, postprocessor],
    flow="preprocessor >> refiner >> postprocessor",
)
```

---

## 4. Key Differences — Pregel Channels vs. Swarm Transfer Functions

| Aspect | Agent-Core (Pregel) | Orbiter (Swarm) |
|--------|-------------------|-----------------|
| **Data routing** | Typed channels with reducers (last-write, append) | Direct output→input chaining; `WorkflowState` for shared data |
| **Fan-out** | Multiple output channels from one node | `ParallelGroup` with concurrent `asyncio.TaskGroup` |
| **Fan-in** | Channel reducers merge multiple inputs | `ParallelGroup` aggregation (custom function or join) |
| **Topology** | Imperative: `add_node()`, `add_channel()` | Declarative: flow DSL string parsed into `Graph` |
| **Node detection** | Explicit node type registration | Duck-typing markers (`is_branch`, `is_loop`, etc.) |
| **State management** | `WorkflowState` with commit/rollback transactions | `WorkflowState` as simple shared dict (no transactions) |
| **Checkpointing** | `GraphStore` — persistent checkpoint storage | `WorkflowCheckpointStore` — in-memory; `Swarm.resume()` restores from snapshot |
| **Expression safety** | AST-based, no `eval()` | Same approach + JS-style normalization + size/depth limits |
| **Sub-workflows** | `SubWorkflowComponent` with state sharing | `SwarmNode` with context isolation (default) or optional state forwarding |
| **Execution model** | Graph traversal with channel-driven scheduling | Topological sort (Kahn's) → sequential iteration with special-node dispatch |
| **Visualization** | Built-in Mermaid | `Swarm.to_mermaid()` with shape conventions (diamond=branch, hexagon=loop) |
| **Parallel execution** | Implicit via channel fan-out | Explicit via `ParallelGroup` or DSL `(a \| b)` syntax |

**Why simpler is sufficient:** Orbiter's sequential model with special-node
dispatch covers the same use cases as Pregel's channel-based scheduling.
`ParallelGroup` handles fan-out, topological sort handles DAGs, and
`WorkflowState` enables inter-node data passing without the overhead of typed
channels and reducers.

---

## 5. Migration Table

| Agent-Core Path | Orbiter Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.core.workflow.PregelEngine` | `orbiter.swarm.Swarm` | Multi-agent orchestrator with workflow/handoff/team modes |
| `openjiuwen.core.workflow.PregelNode` | `orbiter.agent.Agent` | Regular agent — no wrapper node class needed |
| `openjiuwen.core.workflow.Channel` | *(no equivalent)* | Replaced by output→input chaining + `WorkflowState` |
| `openjiuwen.core.workflow.LoopComponent` | `orbiter._internal.loop_node.LoopNode` | Three modes: count, items, condition |
| `openjiuwen.core.workflow.BranchComponent` | `orbiter._internal.branch_node.BranchNode` | Expression or callable condition, true/false routing |
| `openjiuwen.core.workflow.SubWorkflowComponent` | `orbiter._internal.nested.SwarmNode` | Nested swarm with context isolation |
| `openjiuwen.core.workflow.WorkflowState` | `orbiter._internal.workflow_state.WorkflowState` | Shared mutable dict (no transactions) |
| `openjiuwen.core.workflow.expression.evaluate` | `orbiter._internal.expression.evaluate` | Safe AST-based evaluator |
| `openjiuwen.core.workflow.expression.ExpressionError` | `orbiter._internal.expression.ExpressionError` | Evaluation failure exception |
| `openjiuwen.core.workflow.GraphStore` | `orbiter._internal.workflow_checkpoint.WorkflowCheckpointStore` | In-memory checkpoint store |
| *(checkpoint snapshot)* | `orbiter._internal.workflow_checkpoint.WorkflowCheckpoint` | Immutable dataclass: node_name, state, completed_nodes, timestamp |
| *(graph topology)* | `orbiter._internal.graph.Graph` | Directed adjacency list |
| *(topological sort)* | `orbiter._internal.graph.topological_sort` | Kahn's algorithm with cycle detection |
| *(flow DSL)* | `orbiter._internal.graph.parse_flow_dsl` | `"a >> b >> c"` / `"(a \| b) >> c"` syntax |
| *(parallel execution)* | `orbiter._internal.agent_group.ParallelGroup` | Concurrent via `asyncio.TaskGroup` |
| *(serial pipeline)* | `orbiter._internal.agent_group.SerialGroup` | Sequential output→input chaining |
| *(mermaid rendering)* | `orbiter._internal.visualization.to_mermaid` | Diamond=branch, hexagon=loop, subroutine=swarm |
| *(resume)* | `orbiter.swarm.Swarm.resume()` | Resume workflow from checkpoint |
