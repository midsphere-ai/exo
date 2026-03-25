# exo.context.token_tracker

Per-agent, per-step token tracking for cost analysis and budget enforcement.

## Module Path

```python
from exo.context.token_tracker import TokenTracker, TokenStep, TokenUsageSummary
```

---

## TokenStep

A single token usage observation for one LLM call.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
TokenStep(
    agent_id: str,
    step: int,
    prompt_tokens: int,
    output_tokens: int,
)
```

| Parameter | Type | Description |
|---|---|---|
| `agent_id` | `str` | The agent that made the LLM call |
| `step` | `int` | Zero-based step index within the agent's trajectory |
| `prompt_tokens` | `int` | Number of prompt (input) tokens |
| `output_tokens` | `int` | Number of output (completion) tokens |

### Properties

| Property | Type | Description |
|---|---|---|
| `total_tokens` | `int` | `prompt_tokens + output_tokens` |

---

## TokenUsageSummary

Aggregated token usage across agents and steps.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
TokenUsageSummary(
    prompt_tokens: int,
    output_tokens: int,
    total_tokens: int,
    step_count: int,
)
```

| Field | Type | Description |
|---|---|---|
| `prompt_tokens` | `int` | Total prompt tokens |
| `output_tokens` | `int` | Total output tokens |
| `total_tokens` | `int` | Total tokens (prompt + output) |
| `step_count` | `int` | Number of steps |

---

## TokenTracker

Tracks per-agent, per-step token usage.

### Constructor

```python
TokenTracker()
```

No parameters.

### Methods

#### add_step()

```python
def add_step(
    self,
    agent_id: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
) -> TokenStep
```

Record a token usage step for an agent. The step index is automatically assigned per-agent (0-based count of existing steps for that agent).

| Parameter | Type | Description |
|---|---|---|
| `agent_id` | `str` | Identifier of the agent |
| `prompt_tokens` | `int` | Number of prompt tokens |
| `output_tokens` | `int` | Number of output tokens |

**Returns:** The created `TokenStep`.

#### get_trajectory()

```python
def get_trajectory(self, agent_id: str) -> list[TokenStep]
```

Get the ordered list of token steps for a specific agent.

#### total_usage()

```python
def total_usage(self) -> TokenUsageSummary
```

Aggregate token usage across all agents and steps.

#### agent_usage()

```python
def agent_usage(self, agent_id: str) -> TokenUsageSummary
```

Aggregate token usage for a specific agent.

### Properties

| Property | Type | Description |
|---|---|---|
| `agent_ids` | `list[str]` | Unique agent IDs in first-seen order |
| `steps` | `list[TokenStep]` | All recorded steps in order (copy) |

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Total number of recorded steps |
| `__repr__` | `TokenTracker(agents=2, steps=5)` |

### Example

```python
from exo.context.token_tracker import TokenTracker

tracker = TokenTracker()

# Record steps for agent-a
tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)

# Record steps for agent-b
tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)

# Per-agent trajectory
trajectory = tracker.get_trajectory("agent-a")
assert len(trajectory) == 2
assert trajectory[0].step == 0
assert trajectory[1].step == 1

# Per-agent aggregation
usage_a = tracker.agent_usage("agent-a")
assert usage_a.prompt_tokens == 220
assert usage_a.output_tokens == 110
assert usage_a.step_count == 2

# Global aggregation
total = tracker.total_usage()
assert total.total_tokens == 610   # 100+50 + 120+60 + 200+80
assert total.step_count == 3

# Agent list
assert tracker.agent_ids == ["agent-a", "agent-b"]
```
