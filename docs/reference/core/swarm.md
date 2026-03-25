# exo.swarm

Multi-agent orchestration with flow DSL, supporting workflow, handoff, and team execution modes.

**Module:** `exo.swarm`

```python
from exo.swarm import Swarm, SwarmError
# or
from exo import Swarm
```

---

## SwarmError

```python
class SwarmError(ExoError)
```

Raised for swarm-level errors (invalid flow, missing agents, duplicate agents, unsupported mode, max handoffs exceeded, etc.). Inherits from `ExoError`.

---

## Swarm

```python
class Swarm
```

Multi-agent orchestration container. Groups agents and defines their execution topology via a flow DSL.

### Execution Modes

- **workflow** -- Agents run sequentially with output-to-input chaining. Each agent's output becomes the next agent's input.
- **handoff** -- Agents delegate dynamically via handoff targets. The first agent runs, and if its output matches a handoff target name, control transfers to that target.
- **team** -- The first agent is the lead and others are workers. The lead can delegate to workers via auto-generated `delegate_to_{name}` tools.

### Constructor

```python
def __init__(
    self,
    *,
    agents: list[Any],
    flow: str | None = None,
    mode: str = "workflow",
    max_handoffs: int = 10,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agents` | `list[Any]` | *(required)* | List of `Agent` instances to include in the swarm. |
| `flow` | `str \| None` | `None` | Flow DSL string defining execution order (e.g. `"a >> b >> c"`). If not provided, agents run in the order they are given. |
| `mode` | `str` | `"workflow"` | Execution mode: `"workflow"`, `"handoff"`, or `"team"`. |
| `max_handoffs` | `int` | `10` | Maximum number of handoff transitions before raising an error (handoff mode only). |

**Raises:**
- `SwarmError` -- if `agents` is empty.
- `SwarmError` -- if duplicate agent names are found.
- `SwarmError` -- if the flow DSL references unknown agents.
- `SwarmError` -- if the flow DSL contains a cycle.

### Instance Attributes

| Name | Type | Description |
|------|------|-------------|
| `mode` | `str` | Execution mode. |
| `max_handoffs` | `int` | Max handoff transitions. |
| `agents` | `dict[str, Any]` | Agents indexed by name. |
| `flow_order` | `list[str]` | Topologically sorted execution order. |
| `flow` | `str \| None` | Original flow DSL string. |
| `name` | `str` | Auto-generated name, e.g. `"swarm(first_agent...)"`. |

### Methods

#### run()

```python
async def run(
    self,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
) -> RunResult
```

Execute the swarm according to its mode.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `str` | *(required)* | User query string. |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history. |
| `provider` | `Any` | `None` | LLM provider for all agents. |
| `max_retries` | `int` | `3` | Retry attempts for transient errors. |

**Returns:** `RunResult` from the final agent in the chain.

**Raises:** `SwarmError` if mode is unsupported or execution fails.

#### describe()

```python
def describe(self) -> dict[str, Any]
```

Return a summary of the swarm's configuration.

**Returns:** Dict with keys:
- `mode` (str)
- `flow` (str | None)
- `flow_order` (list[str])
- `agents` (dict[str, dict]) -- each agent's `describe()` output

#### \_\_repr\_\_()

```python
def __repr__(self) -> str
```

E.g. `Swarm(mode='workflow', agents=['a', 'b', 'c'], flow='a >> b >> c')`.

### Flow DSL Syntax

The flow DSL uses `>>` to denote sequential dependency and `(x | y)` for parallel groups:

```
"a >> b >> c"           # Linear chain: a then b then c
"(a | b) >> c"          # Parallel a and b, then c
"a >> (b | c) >> d"     # a, then parallel b and c, then d
```

### Example: Workflow Mode

```python
from exo import Agent, Swarm, run

researcher = Agent(name="researcher", instructions="Research the topic.")
writer = Agent(name="writer", instructions="Write an article based on the research.")
editor = Agent(name="editor", instructions="Edit the article for clarity.")

swarm = Swarm(
    agents=[researcher, writer, editor],
    flow="researcher >> writer >> editor",
    mode="workflow",
)

result = run.sync(swarm, "Write an article about quantum computing")
print(result.output)
```

### Example: Handoff Mode

```python
from exo import Agent, Swarm, run

triage = Agent(name="triage", instructions="Route to the right specialist.")
billing = Agent(name="billing", instructions="Handle billing questions.")
support = Agent(name="support", instructions="Handle technical support.")

# Wire handoffs
triage_agent = Agent(
    name="triage",
    instructions="Route to billing or support based on the question.",
    handoffs=[billing, support],
)

swarm = Swarm(
    agents=[triage_agent, billing, support],
    mode="handoff",
    max_handoffs=5,
)

result = run.sync(swarm, "I was charged twice for my subscription")
```

### Example: Team Mode

```python
from exo import Agent, Swarm, run

lead = Agent(name="lead", instructions="You are the team lead. Delegate tasks.")
coder = Agent(name="coder", instructions="Write Python code.")
tester = Agent(name="tester", instructions="Write unit tests.")

swarm = Swarm(
    agents=[lead, coder, tester],
    flow="lead >> coder >> tester",
    mode="team",
)

# In team mode, lead gets auto-generated delegate_to_coder and delegate_to_tester tools
result = run.sync(swarm, "Build a calculator module with tests")
```
