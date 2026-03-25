# Multi-Agent Swarms

A `Swarm` groups multiple agents and defines their execution topology. Exo supports three orchestration modes: **workflow** (sequential pipeline), **handoff** (agent-driven delegation), and **team** (lead-worker delegation).

## Basic Usage

```python
from exo.agent import Agent
from exo.swarm import Swarm
from exo.runner import run

researcher = Agent(name="researcher", instructions="You research topics deeply.")
writer = Agent(name="writer", instructions="You write clear, concise articles.")
editor = Agent(name="editor", instructions="You edit text for grammar and clarity.")

swarm = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
)

result = await run(swarm, "Write an article about quantum computing")
print(result.output)
```

## Constructor Parameters

All parameters are keyword-only.

```python
class Swarm:
    def __init__(
        self,
        *,
        agents: list[Any],
        flow: str | None = None,
        mode: str = "workflow",
        max_handoffs: int = 10,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[Agent]` | *required* | List of agents in the swarm (at least one) |
| `flow` | `str \| None` | `None` | Flow DSL string defining execution order. If `None`, agents run in list order |
| `mode` | `str` | `"workflow"` | Execution mode: `"workflow"`, `"handoff"`, or `"team"` |
| `max_handoffs` | `int` | `10` | Maximum handoff transitions (handoff mode only) |

## Flow DSL

The flow DSL uses `>>` to define sequential execution order:

```python
# Simple pipeline
flow = "a >> b >> c"

# Two-stage pipeline
flow = "researcher >> writer"
```

The flow string is parsed into a directed acyclic graph (DAG), topologically sorted, and validated against the registered agent names. Cycles in the flow raise `SwarmError`.

When `flow` is `None`, agents execute in the order they appear in the `agents` list.

## Mode: Workflow

In workflow mode (the default), agents execute sequentially. Each agent's output becomes the next agent's input:

```python
swarm = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
    mode="workflow",  # default
)

result = await run(swarm, "Write about AI safety")
# 1. researcher receives "Write about AI safety"
# 2. writer receives researcher's output
# 3. editor receives writer's output
# result.output = editor's final output
```

Workflow mode also supports [agent groups](agent-groups.md) and [nested swarms](agent-groups.md#nested-swarms-with-swarmnode) as nodes in the flow.

## Mode: Handoff

In handoff mode, agents delegate dynamically. The first agent in the flow runs first. If its output matches a handoff target name, control transfers to that target with the full conversation history:

```python
triage = Agent(
    name="triage",
    instructions="Route to 'billing' for payment issues, 'technical' for tech issues.",
    handoffs=[billing_agent, technical_agent],
)

swarm = Swarm(
    agents=[triage, billing_agent, technical_agent],
    mode="handoff",
    max_handoffs=5,
)

result = await run(swarm, "I can't log into my account")
```

The handoff chain continues until an agent produces output that does not match any handoff target name, or `max_handoffs` is exceeded.

**How handoff detection works:** The agent's output text (stripped of whitespace) is compared against the names of its declared handoff targets. If it matches exactly, control transfers to that agent.

## Mode: Team

In team mode, the first agent is the **lead** and the remaining agents are **workers**. The lead receives auto-generated `delegate_to_{name}` tools that invoke worker agents:

```python
lead = Agent(
    name="project_manager",
    instructions="Coordinate work between the researcher and coder.",
)
researcher = Agent(name="researcher", instructions="Research topics.")
coder = Agent(name="coder", instructions="Write Python code.")

swarm = Swarm(
    agents=[lead, researcher, coder],
    mode="team",
)

result = await run(swarm, "Build a CLI tool that fetches weather data")
# The project_manager can call:
#   delegate_to_researcher(task="...")
#   delegate_to_coder(task="...")
```

Each delegate tool has a `task` parameter (string) that describes what the worker should do. The worker runs with that task as input, and its output is returned as the tool result to the lead.

Team mode requires at least two agents (lead + at least one worker).

## Running a Swarm

Swarms work seamlessly with `run()`:

```python
from exo.runner import run

# Async
result = await run(swarm, "Hello!")

# Sync
result = run.sync(swarm, "Hello!")
```

The `run()` function detects swarms by checking for the `flow_order` attribute and delegates to the swarm's own `run()` method.

## Swarm.describe()

Inspect a swarm's configuration:

```python
print(swarm.describe())
# {
#     "mode": "workflow",
#     "flow": "researcher >> writer >> editor",
#     "flow_order": ["researcher", "writer", "editor"],
#     "agents": {
#         "researcher": {"name": "researcher", "model": "openai:gpt-4o", ...},
#         "writer": {"name": "writer", "model": "openai:gpt-4o", ...},
#         "editor": {"name": "editor", "model": "openai:gpt-4o", ...},
#     },
# }
```

## Error Handling

Swarm errors raise `SwarmError`:

```python
from exo.swarm import Swarm, SwarmError

# No agents
try:
    Swarm(agents=[])
except SwarmError as e:
    print(e)  # "Swarm requires at least one agent"

# Duplicate agent names
try:
    Swarm(agents=[agent_a, agent_a_copy])
except SwarmError as e:
    print(e)  # "Duplicate agent name 'a' in swarm"

# Flow references unknown agent
try:
    Swarm(agents=[agent_a], flow="a >> unknown")
except SwarmError as e:
    print(e)  # "Flow references unknown agent 'unknown'"

# Cycle in flow
try:
    Swarm(agents=[agent_a, agent_b], flow="a >> b >> a")
except SwarmError as e:
    print(e)  # "Cycle in flow DSL: ..."
```

## Advanced: Combining Modes

You can nest swarms within swarms using [SwarmNode](agent-groups.md#nested-swarms-with-swarmnode) to combine different orchestration modes:

```python
from exo._internal.nested import SwarmNode

# Inner pipeline (workflow mode)
pipeline = Swarm(agents=[writer, editor], flow="writer >> editor")
pipeline_node = SwarmNode(swarm=pipeline, name="write_pipeline")

# Outer swarm (team mode with nested workflow)
outer = Swarm(
    agents=[lead, researcher, pipeline_node],
    mode="team",
)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Swarm` | `exo.swarm` | Multi-agent orchestration container |
| `SwarmError` | `exo.swarm` | Swarm-level error |
| `SwarmNode` | `exo._internal.nested` | Wrap a swarm as a node in another swarm |
| `ParallelGroup` | `exo._internal.agent_group` | Concurrent agent execution |
| `SerialGroup` | `exo._internal.agent_group` | Sequential agent execution |
