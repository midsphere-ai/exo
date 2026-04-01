---
name: exo:ralph
description: "Use when implementing iterative refinement loops with Ralph — RalphRunner, 5-phase loop (Run/Analyze/Learn/Plan/Halt), scorers, reflectors, stop conditions, streaming with RalphIterationEvent/RalphStopEvent, from_agent() factory, RalphNode for Swarm integration. Triggers on: ralph, RalphRunner, iterative refinement, ralph loop, scorer, reflector, stop condition, RalphIterationEvent, RalphStopEvent, RalphNode, from_agent, ralph streaming."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Ralph — Iterative Refinement Loops

## When To Use This Skill

Use this skill when the developer needs to:
- Run an agent in an iterative loop with scoring, reflection, and re-prompting
- Configure stop conditions (max iterations, score threshold, timeout, cost limit)
- Score agent output with custom scorers
- Add reflection to learn from failures and improve across iterations
- Stream Ralph loop events (iteration progress, inner agent events, stop events)
- Plug a Ralph loop into a Swarm as a node via `RalphNode`

## Decision Guide

1. **Simple iterative loop with an agent?** → `RalphRunner.from_agent(agent, scorers=[...])` — wires both `run()` and `stream()` automatically
2. **Custom execution logic (not an Agent)?** → `RalphRunner(execute_fn=my_fn, scorers=[...])` with manual async callable
3. **Want streaming events during the loop?** → Use `.stream()` — requires `stream_execute_fn` (provided automatically by `from_agent()`)
4. **Need to plug a Ralph loop into a Swarm?** → Wrap in `RalphNode(runner=ralph_runner, name="loop_name")`
5. **Need to stop on quality threshold?** → Set `StopConditionConfig(score_threshold=0.9)`
6. **Need reflection on failures?** → Pass a `Reflector` — reflects when scores are below threshold or execution fails

## Reference

### RalphRunner Constructor

```python
from exo.eval.ralph import RalphRunner, RalphConfig

runner = RalphRunner(
    execute_fn=my_execute_fn,           # Required: async (str) -> str
    scorers=[scorer1, scorer2],          # Required: list of Scorer instances
    stream_execute_fn=my_stream_fn,      # Optional: async (str) -> AsyncIterator[StreamEvent]
    config=RalphConfig(...),             # Optional: loop configuration
    reflector=my_reflector,              # Optional: Reflector for Learn phase
    replan_fn=None,                      # Reserved for future use
)
```

### from_agent() Factory (Recommended)

Creates a `RalphRunner` wired to an `Agent`'s `run()` and `run.stream()`:

```python
from exo import Agent
from exo.eval.ralph import RalphRunner

agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="Research the topic thoroughly.",
    tools=[search, fetch_paper],
)

runner = RalphRunner.from_agent(
    agent,
    scorers=[quality_scorer, completeness_scorer],
    config=RalphConfig(
        stop_condition=StopConditionConfig(
            max_iterations=5,
            score_threshold=0.85,
        ),
    ),
)
```

**What `from_agent()` does:**
- Creates `execute_fn` that calls `await run(agent, input)` and returns `result.output`
- Creates `stream_execute_fn` that yields events from `run.stream(agent, input)`
- Both are set on the runner — enabling `.run()` and `.stream()`

### The 5-Phase Loop

Each iteration runs these phases in order:

```
1. Run     → Execute the agent/task with current input
2. Analyze → Score output with all configured scorers
3. Learn   → Reflect on failures (if reflector configured)
4. Plan    → Re-prompt by appending reflection suggestions
5. Halt    → Check stop conditions; break or continue
```

### RalphRunner.run()

```python
result = await runner.run("Analyze market trends")
# Returns RalphResult
```

**RalphResult fields:**
```python
@dataclass(frozen=True)
class RalphResult:
    output: str                         # Final iteration's output
    stop_type: StopType                 # Why the loop stopped
    reason: str                         # Human-readable stop reason
    iterations: int                     # Total iterations executed
    scores: dict[str, float]            # Final iteration's scores
    state: dict[str, Any]               # Serialized LoopState
    reflections: list[dict[str, Any]]   # Reflection history
```

### RalphRunner.stream()

Streams the full loop with lifecycle events interleaved with inner agent events:

```python
from exo.types import TextEvent
from exo.eval.ralph import RalphIterationEvent, RalphStopEvent

async for event in runner.stream("Analyze market trends", name="research"):
    match event:
        case RalphIterationEvent(iteration=n, status="started"):
            print(f"\n--- Iteration {n} ---")
        case TextEvent(text=t):
            print(t, end="", flush=True)
        case RalphIterationEvent(status="completed", scores=s):
            print(f"\n[Scores: {s}]")
        case RalphStopEvent(stop_type=st, reason=r):
            print(f"\n=== Stopped ({st}): {r} ===")
```

**Requires `stream_execute_fn`** — provided automatically by `from_agent()`. Raises `ValueError` if not set.

**Event sequence per iteration:**
```
RalphIterationEvent(status="started", iteration=1)
  TextEvent(...)         ← inner agent events
  ToolCallEvent(...)     ← inner agent events
  TextEvent(...)         ← inner agent events
RalphIterationEvent(status="completed", iteration=1, scores={...})
... (next iteration)
RalphStopEvent(stop_type="score_threshold", iterations=3)
```

### RalphIterationEvent

```python
class RalphIterationEvent:
    type: Literal["ralph_iteration"] = "ralph_iteration"
    iteration: int                      # 1-based iteration number
    status: Literal["started", "completed", "failed"]
    scores: dict[str, float] = {}       # Scorer results (on completed)
    agent_name: str = ""                # Set to the `name` param of stream()
```

### RalphStopEvent

```python
class RalphStopEvent:
    type: Literal["ralph_stop"] = "ralph_stop"
    stop_type: str                      # StopType enum value as string
    reason: str                         # Human-readable stop reason
    iterations: int                     # Total iterations executed
    final_scores: dict[str, float] = {} # Last iteration's scores
    agent_name: str = ""
```

### Configuration

#### RalphConfig

```python
from exo.eval.ralph import RalphConfig, StopConditionConfig, ValidationConfig, ReflectionConfig

config = RalphConfig(
    validation=ValidationConfig(
        enabled=True,                   # Enable scoring (default True)
        min_score_threshold=0.5,        # Threshold for triggering reflection
    ),
    reflection=ReflectionConfig(
        enabled=True,                   # Enable reflection (default True)
        level="medium",                 # Reflection depth
        max_history=50,                 # Max reflection history entries
    ),
    stop_condition=StopConditionConfig(
        max_iterations=10,              # Hard cap on iterations (default 10)
        timeout=0.0,                    # Seconds (0 = no timeout)
        max_cost=0.0,                   # Cost limit (0 = no limit)
        max_consecutive_failures=3,     # Failures before stopping (default 3)
        score_threshold=0.0,            # Stop when mean score >= this (0 = disabled)
    ),
)
```

#### StopType Enum

```python
from exo.eval.ralph import StopType

StopType.COMPLETION                     # Explicit completion signal
StopType.MAX_ITERATIONS                 # Hit max_iterations
StopType.TIMEOUT                        # Hit timeout
StopType.MAX_COST                       # Hit cost limit
StopType.MAX_CONSECUTIVE_FAILURES       # Too many failures in a row
StopType.SCORE_THRESHOLD                # Mean score met threshold
StopType.USER_INTERRUPTED               # User-initiated stop
StopType.SYSTEM_ERROR                   # Unrecoverable error
```

### Scorers

Implement the `Scorer` ABC to evaluate agent output:

```python
from exo.eval.base import Scorer, ScorerResult

class QualityScorer(Scorer):
    async def score(self, case_id: str, input: str, output: str) -> ScorerResult:
        # Score the output (0.0 to 1.0)
        quality = len(output) / 1000  # Simple length-based example
        return ScorerResult(scorer_name="quality", score=min(quality, 1.0))
```

Multiple scorers run in sequence. Failed scorers are silently skipped (logged as warnings).

### Reflectors

Implement the `Reflector` ABC for the Learn phase:

```python
from exo.eval.reflection import Reflector, ReflectionResult

class SimpleReflector(Reflector):
    def __init__(self):
        super().__init__(name="simple")

    async def analyze(self, context: dict) -> dict:
        return {
            "summary": f"Iteration {context['iteration']} needs improvement",
            "suggestions": ["Be more specific", "Add more examples"],
        }

    async def suggest(self, insights: dict) -> dict:
        return {"suggestions": insights.get("suggestions", [])}
```

The reflector fires when:
- Execution fails, OR
- Mean score is below `validation.min_score_threshold`

Suggestions are appended to the original input in the Plan phase:
```
Original task

[Previous feedback]
- Be more specific
- Add more examples
```

### RalphNode (Swarm Integration)

Wraps a `RalphRunner` so it can be a node in a Swarm workflow:

```python
from exo import RalphNode, Swarm, Agent
from exo.eval.ralph import RalphRunner

researcher = Agent(name="researcher", instructions="Research deeply.")
ralph = RalphNode(
    runner=RalphRunner.from_agent(researcher, scorers=[quality_scorer]),
    name="research_loop",
)
summarizer = Agent(name="summarizer", instructions="Summarize findings.")

swarm = Swarm(
    agents=[ralph, summarizer],
    flow="research_loop >> summarizer",
)

# Run (non-streaming)
result = await swarm.run("Analyze AI trends", provider=provider)

# Stream (all events flow through)
async for event in swarm.stream("Analyze AI trends", provider=provider):
    print(event)
```

**How it works:**
- `is_group = True` triggers Swarm's duck-typing path (`getattr(agent, "is_group", False)`)
- Swarm calls `node.stream()` which delegates to `runner.stream()`
- `RalphIterationEvent`, inner agent events, and `RalphStopEvent` all flow to the outer stream
- The Ralph loop's final text output chains to the next agent in the workflow

### LoopState

Tracks iteration progress:

```python
from exo.eval.ralph import LoopState

state = LoopState()
state.iteration          # Current iteration number
state.elapsed()          # Seconds since loop start
state.success_rate()     # Fraction of successful steps
state.latest_score()     # Most recent score snapshot
state.best_score("quality")  # Highest value for a metric
state.to_dict()          # Serialize to plain dict
```

## Patterns

### Quality-Gated Research

```python
runner = RalphRunner.from_agent(
    research_agent,
    scorers=[quality_scorer, completeness_scorer],
    config=RalphConfig(
        stop_condition=StopConditionConfig(
            max_iterations=5,
            score_threshold=0.9,
        ),
        validation=ValidationConfig(min_score_threshold=0.7),
    ),
    reflector=SimpleReflector(),
)

result = await runner.run("Comprehensive analysis of quantum computing in 2026")
print(f"Stopped: {result.stop_type.value} after {result.iterations} iterations")
print(f"Final scores: {result.scores}")
```

### Streaming Progress to CLI

```python
async for event in runner.stream("Research task", name="research"):
    match event:
        case RalphIterationEvent(iteration=n, status="started"):
            print(f"\n{'='*40}")
            print(f"Iteration {n}")
            print(f"{'='*40}")
        case TextEvent(text=t):
            print(t, end="", flush=True)
        case RalphIterationEvent(status="completed", scores=s):
            for name, score in s.items():
                bar = "#" * int(score * 20)
                print(f"\n  {name}: [{bar:<20}] {score:.2f}")
        case RalphIterationEvent(status="failed"):
            print("\n  [FAILED]")
        case RalphStopEvent(stop_type=st, reason=r, iterations=n):
            print(f"\n\nDone: {st} ({r}) in {n} iterations")
```

### Ralph in a Multi-Agent Pipeline

```python
# Research loop feeds into writer and editor
research_ralph = RalphNode(
    runner=RalphRunner.from_agent(researcher, scorers=[quality]),
    name="research_loop",
)
writer = Agent(name="writer", instructions="Write a report from research.")
editor = Agent(name="editor", instructions="Edit for clarity and accuracy.")

pipeline = Swarm(
    agents=[research_ralph, writer, editor],
    flow="research_loop >> writer >> editor",
)

result = await pipeline.run("AI safety trends", provider=provider)
```

### Custom Execute Function (No Agent)

```python
import httpx

async def fetch_and_analyze(input: str) -> str:
    """Custom execution that doesn't use an Agent."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/search?q={input}")
        return resp.text

runner = RalphRunner(
    execute_fn=fetch_and_analyze,
    scorers=[relevance_scorer],
    config=RalphConfig(stop_condition=StopConditionConfig(max_iterations=3)),
)

result = await runner.run("latest AI research")
```

## Gotchas

- **`from_agent()` is the recommended way to create a runner** — it wires both `execute_fn` and `stream_execute_fn` automatically
- **`stream()` requires `stream_execute_fn`** — raises `ValueError` if not provided. Use `from_agent()` or pass it manually.
- **Scorers run in sequence, not parallel** — each `scorer.score()` is awaited one at a time
- **Failed scorers are silently skipped** — they log a warning but don't stop the loop
- **Reflection fires on low scores OR failure** — controlled by `validation.min_score_threshold`
- **`score_threshold=0.0` disables score-based stopping** — set to a positive value to enable
- **`max_iterations` defaults to 10** — always set this to prevent runaway loops
- **`max_consecutive_failures` defaults to 3** — 3 failures in a row triggers `MAX_CONSECUTIVE_FAILURES` stop
- **`RalphStopEvent.stop_type` is a string** — it's the `.value` of the `StopType` enum, not the enum itself
- **`RalphNode` ignores provider/messages** — Ralph manages its own execution context. The Swarm passes them but they're not used.
- **Text assembly for scoring** — during `stream()`, `TextEvent.text` chunks are concatenated to form the output passed to scorers. Non-text events don't contribute to the scored output.
- **`from_agent()` uses a deferred import** — `from exo.runner import run` is imported inside the method to avoid circular dependencies
