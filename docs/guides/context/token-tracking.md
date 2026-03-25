# Token Tracking

The `TokenTracker` records prompt and output token usage per agent, per step. It provides trajectory-level tracking, per-agent summaries, and total usage across an entire context. This data feeds into the Ralph Loop for cost analysis, budget enforcement, and optimization decisions.

## Basic Usage

```python
from exo.context import TokenTracker

tracker = TokenTracker()

# Record token usage for each step
tracker.add_step("agent-1", step=1, prompt_tokens=500, output_tokens=150)
tracker.add_step("agent-1", step=2, prompt_tokens=800, output_tokens=200)
tracker.add_step("agent-2", step=1, prompt_tokens=300, output_tokens=100)

# Total usage across all agents
total = tracker.total_usage()
print(f"Total: {total.prompt_tokens} prompt, {total.output_tokens} output")
# Total: 1600 prompt, 450 output
```

## Token Steps

Each call to `add_step()` creates an immutable `TokenStep` record:

```python
from exo.context.token_tracker import TokenStep

# TokenStep is a frozen dataclass
step = TokenStep(
    agent_id="agent-1",
    step=1,
    prompt_tokens=500,
    output_tokens=150,
)
print(step.agent_id)       # "agent-1"
print(step.step)           # 1
print(step.prompt_tokens)  # 500
print(step.output_tokens)  # 150
```

## Trajectory

Get the full sequence of token steps in chronological order:

```python
trajectory = tracker.get_trajectory()
for step in trajectory:
    print(f"Agent {step.agent_id}, Step {step.step}: "
          f"{step.prompt_tokens}+{step.output_tokens} tokens")
```

## Per-Agent Usage

Get a `TokenUsageSummary` for a specific agent:

```python
usage = tracker.agent_usage("agent-1")
print(f"Prompt: {usage.prompt_tokens}")
print(f"Output: {usage.output_tokens}")
print(f"Total: {usage.total_tokens}")
print(f"Steps: {usage.steps}")
```

`TokenUsageSummary` is a frozen dataclass:

```python
@dataclass(frozen=True)
class TokenUsageSummary:
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    steps: int
```

## Total Usage

Aggregate usage across all agents:

```python
total = tracker.total_usage()
print(f"All agents: {total.prompt_tokens} prompt + {total.output_tokens} output "
      f"= {total.total_tokens} total across {total.steps} steps")
```

## Agent Discovery

List all agent IDs that have recorded steps:

```python
agent_ids = tracker.agent_ids
print(agent_ids)  # {"agent-1", "agent-2"}
```

## Integration with Context

The `Context` object holds a `TokenTracker` in its `token_usage` attribute:

```python
from exo.context import Context, ContextConfig

ctx = Context(task_id="task-1", config=ContextConfig())

# Record usage during agent execution
ctx.token_usage.add_step("main-agent", step=1, prompt_tokens=1000, output_tokens=300)

# Check budget
total = ctx.token_usage.total_usage()
if total.total_tokens > 100_000:
    print("Token budget exceeded!")
```

When forking and merging contexts, the child's token delta is consolidated into the parent:

```python
child = ctx.fork("subtask")
child.token_usage.add_step("sub-agent", step=1, prompt_tokens=200, output_tokens=50)

# Merge consolidates token usage
ctx.merge(child)
total = ctx.token_usage.total_usage()
# Now includes sub-agent's tokens
```

## Advanced Patterns

### Budget Enforcement

Use token tracking in a processor to enforce budget limits:

```python
from exo.context import ContextProcessor

class BudgetGuard(ContextProcessor):
    event = "pre_llm_call"
    name = "budget_guard"

    def __init__(self, max_tokens: int = 100_000):
        self._max_tokens = max_tokens

    async def process(self, ctx, payload: dict) -> dict:
        total = ctx.token_usage.total_usage()
        if total.total_tokens >= self._max_tokens:
            raise RuntimeError(
                f"Token budget exceeded: {total.total_tokens}/{self._max_tokens}"
            )
        payload["remaining_budget"] = self._max_tokens - total.total_tokens
        return payload
```

### Cost Estimation

Map token counts to cost using provider pricing:

```python
PRICING = {
    "gpt-4o": {"prompt": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "claude-3-opus": {"prompt": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
}

def estimate_cost(tracker: TokenTracker, model: str) -> float:
    total = tracker.total_usage()
    rates = PRICING.get(model, {"prompt": 0, "output": 0})
    return (total.prompt_tokens * rates["prompt"] +
            total.output_tokens * rates["output"])
```

### Multi-Agent Comparison

Compare token efficiency across agents in a swarm:

```python
for agent_id in tracker.agent_ids:
    usage = tracker.agent_usage(agent_id)
    ratio = usage.output_tokens / max(usage.prompt_tokens, 1)
    print(f"{agent_id}: {usage.total_tokens} tokens "
          f"({usage.steps} steps, {ratio:.2f} output/prompt ratio)")
```

### Checkpoint Token State

Token usage is included in checkpoint snapshots:

```python
cp = ctx.snapshot(metadata={"reason": "progress save"})
print(cp.token_usage)  # Token usage at checkpoint time

# After restore, token usage is reset to checkpoint state
ctx.restore(cp)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `TokenTracker` | `exo.context` | Records and queries per-agent per-step token usage |
| `TokenTracker.add_step(agent_id, step, prompt_tokens, output_tokens)` | | Record one step's usage |
| `TokenTracker.get_trajectory()` | | Get all steps in chronological order |
| `TokenTracker.total_usage()` | | Aggregate `TokenUsageSummary` across all agents |
| `TokenTracker.agent_usage(agent_id)` | | `TokenUsageSummary` for one agent |
| `TokenTracker.agent_ids` | | Set of all agent IDs with recorded steps |
| `TokenStep` | `exo.context.token_tracker` | Frozen dataclass: `agent_id`, `step`, `prompt_tokens`, `output_tokens` |
| `TokenUsageSummary` | `exo.context.token_tracker` | Frozen dataclass: `prompt_tokens`, `output_tokens`, `total_tokens`, `steps` |
