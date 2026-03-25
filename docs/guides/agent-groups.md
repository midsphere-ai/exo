# Agent Groups

Agent groups enable parallel and serial execution patterns within a [Swarm](multi-agent.md). Groups behave like agents from the swarm's perspective -- they have a `name` attribute and a `run()` method, so they can be placed in the agent list and flow DSL.

## ParallelGroup

A `ParallelGroup` runs multiple agents concurrently using `asyncio.TaskGroup`. All agents receive the same input and their outputs are aggregated.

```python
from exo.agent import Agent
from exo._internal.agent_group import ParallelGroup
from exo.swarm import Swarm
from exo.runner import run

analyst_a = Agent(name="financial", instructions="Analyze financial aspects.")
analyst_b = Agent(name="technical", instructions="Analyze technical aspects.")
analyst_c = Agent(name="market", instructions="Analyze market trends.")

parallel = ParallelGroup(
    name="analysis_team",
    agents=[analyst_a, analyst_b, analyst_c],
)

# Use in a swarm
synthesizer = Agent(name="synthesizer", instructions="Combine analyses into a report.")
swarm = Swarm(
    agents=[parallel, synthesizer],
    flow="analysis_team >> synthesizer",
)

result = await run(swarm, "Analyze the potential of solar energy")
```

### Constructor

```python
class ParallelGroup:
    def __init__(
        self,
        *,
        name: str,
        agents: list[Any],
        separator: str = "\n\n",
        aggregate_fn: Any = None,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Group name (used as a node in flow DSL) |
| `agents` | `list[Agent]` | *required* | Agents to run concurrently (at least one) |
| `separator` | `str` | `"\n\n"` | String used to join agent outputs |
| `aggregate_fn` | `Callable[[list[RunResult]], str] \| None` | `None` | Custom aggregation function; overrides `separator` |

### Custom Aggregation

By default, outputs are joined with the separator string. For more control, provide an aggregation function:

```python
from exo.types import RunResult

def weighted_aggregate(results: list[RunResult]) -> str:
    """Custom aggregation that labels each output."""
    parts = []
    for i, r in enumerate(results):
        parts.append(f"--- Analysis {i+1} ---\n{r.output}")
    return "\n\n".join(parts)

parallel = ParallelGroup(
    name="analysis_team",
    agents=[analyst_a, analyst_b],
    aggregate_fn=weighted_aggregate,
)
```

### Result Merging

The `ParallelGroup.run()` method returns a single `RunResult` with:

- **output:** Aggregated text from all agents
- **messages:** All messages from all agents combined
- **usage:** Sum of all `input_tokens` and `output_tokens` across agents
- **steps:** Sum of steps from all agents

## SerialGroup

A `SerialGroup` runs agents sequentially, chaining each agent's output as the next agent's input. This is similar to workflow mode in a swarm but packaged as a single node.

```python
from exo._internal.agent_group import SerialGroup

drafter = Agent(name="drafter", instructions="Write a first draft.")
reviewer = Agent(name="reviewer", instructions="Review and improve the draft.")

serial = SerialGroup(
    name="draft_pipeline",
    agents=[drafter, reviewer],
)
```

### Constructor

```python
class SerialGroup:
    def __init__(
        self,
        *,
        name: str,
        agents: list[Any],
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Group name (used as a node in flow DSL) |
| `agents` | `list[Agent]` | *required* | Agents to run sequentially (at least one) |

### Chaining Behavior

```python
serial = SerialGroup(
    name="pipeline",
    agents=[agent_a, agent_b, agent_c],
)

# When run("Hello!"):
# 1. agent_a receives "Hello!" -> produces output_a
# 2. agent_b receives output_a -> produces output_b
# 3. agent_c receives output_b -> produces output_c
# result.output = output_c
```

The final `RunResult` contains:

- **output:** Output from the last agent
- **messages:** All messages from all agents combined
- **usage:** Sum of all tokens across agents
- **steps:** Sum of steps from all agents

## Combining Groups in Swarms

Groups are designed to be composed within swarms. A common pattern is parallel analysis followed by sequential synthesis:

```python
# Three analysts run in parallel
parallel_analysis = ParallelGroup(
    name="analysts",
    agents=[financial_analyst, tech_analyst, market_analyst],
)

# Two-stage synthesis pipeline
synthesis = SerialGroup(
    name="synthesis",
    agents=[summarizer, editor],
)

# Full swarm: parallel analysis >> synthesis >> final review
swarm = Swarm(
    agents=[parallel_analysis, synthesis, final_reviewer],
    flow="analysts >> synthesis >> final_reviewer",
)
```

## Nested Swarms with SwarmNode

A `SwarmNode` wraps a `Swarm` so it can be used as a node inside another swarm. This enables hierarchical multi-agent architectures.

```python
from exo._internal.nested import SwarmNode

# Inner swarm: a workflow pipeline
inner = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
)

# Wrap it as a node
inner_node = SwarmNode(swarm=inner, name="research_pipeline")

# Outer swarm uses the inner as a single node
outer = Swarm(
    agents=[coordinator, inner_node, reviewer],
    flow="coordinator >> research_pipeline >> reviewer",
)

result = await run(outer, "Create a report on AI trends")
```

### SwarmNode Constructor

```python
class SwarmNode:
    def __init__(
        self,
        *,
        swarm: Any,
        name: str | None = None,   # defaults to inner swarm's name
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swarm` | `Swarm` | *required* | The inner swarm to wrap |
| `name` | `str \| None` | `None` | Node name for the outer flow DSL (defaults to `swarm.name`) |

### Context Isolation

`SwarmNode` provides context isolation between the inner and outer swarms:

- The inner swarm starts with a **fresh message history** on each invocation
- The outer swarm's messages are **not forwarded** to the inner swarm
- No mutable state leaks between inner and outer executions

This ensures that nested swarms behave predictably and do not accumulate cross-boundary state.

## Detection by Swarm

Groups and nested swarms are detected by the `Swarm._run_workflow()` method through duck-typing markers:

- **Groups** set `self.is_group = True`
- **SwarmNode** sets `self.is_swarm = True`

Both types provide the same interface: a `name` attribute and an async `run()` method.

## Error Handling

```python
from exo._internal.agent_group import GroupError, ParallelGroup
from exo._internal.nested import NestedSwarmError, SwarmNode

# Empty group
try:
    ParallelGroup(name="empty", agents=[])
except GroupError as e:
    print(e)  # "ParallelGroup requires at least one agent"

# Invalid swarm node
try:
    SwarmNode(swarm="not a swarm")
except NestedSwarmError as e:
    print(e)  # "SwarmNode requires a Swarm instance (object with flow_order)"
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `ParallelGroup` | `exo._internal.agent_group` | Concurrent agent execution group |
| `SerialGroup` | `exo._internal.agent_group` | Sequential agent execution group |
| `GroupError` | `exo._internal.agent_group` | Group-level error |
| `SwarmNode` | `exo._internal.nested` | Wraps a swarm as a node in another swarm |
| `NestedSwarmError` | `exo._internal.nested` | Nested swarm error |
