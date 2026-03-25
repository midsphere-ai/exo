# Ralph Loop

The Ralph Loop is an iterative refinement framework that implements a five-phase cycle: **Run -> Analyze -> Learn -> Plan -> Halt**. It drives agents through repeated execution, evaluation, reflection, and planning until a stop condition is met (score threshold reached, budget exhausted, or iteration limit hit).

## Basic Usage

```python
from exo.eval.ralph import RalphRunner, RalphConfig, RalphResult

async def execute_fn(input: str, iteration: int) -> dict:
    """Run the agent and return results."""
    result = await my_agent.run(input)
    return {"output": result.output, "score": evaluate(result)}

async def analyze_fn(result: dict, iteration: int) -> dict:
    """Analyze the execution results."""
    return {"quality": result["score"], "issues": find_issues(result)}

async def learn_fn(analysis: dict, iteration: int) -> dict:
    """Extract insights from analysis."""
    return {"insights": extract_insights(analysis)}

async def plan_fn(learnings: dict, iteration: int) -> dict:
    """Plan the next iteration."""
    return {"adjustments": plan_next(learnings)}

config = RalphConfig(max_iterations=10)
runner = RalphRunner(config=config)

result: RalphResult = await runner.run(
    input="Write a Python web scraper",
    execute_fn=execute_fn,
    analyze_fn=analyze_fn,
    learn_fn=learn_fn,
    plan_fn=plan_fn,
)

print(f"Iterations: {result.iterations}")
print(f"Final score: {result.final_score}")
print(f"Stop reason: {result.stop_reason}")
```

## The Five Phases

Each iteration of the Ralph Loop executes five phases in order:

### 1. Run (Execute)

Execute the agent with the current input and any adjustments from previous planning:

```python
async def execute_fn(input: str, iteration: int) -> dict:
    result = await agent.run(input)
    return {
        "output": result.output,
        "score": result.score,
        "tokens": result.token_count,
        "elapsed_ms": result.elapsed * 1000,
    }
```

### 2. Analyze

Evaluate the execution results. Identify strengths, weaknesses, and issues:

```python
async def analyze_fn(result: dict, iteration: int) -> dict:
    score = result["score"]
    return {
        "quality": score,
        "issues": ["incomplete answer"] if score < 0.8 else [],
        "strengths": ["correct format"],
    }
```

### 3. Learn

Extract insights and patterns from the analysis:

```python
async def learn_fn(analysis: dict, iteration: int) -> dict:
    insights = []
    if analysis["quality"] < 0.5:
        insights.append("Agent struggles with this type of task")
    return {"insights": insights, "patterns": []}
```

### 4. Plan

Generate a plan for the next iteration based on learnings:

```python
async def plan_fn(learnings: dict, iteration: int) -> dict:
    adjustments = []
    for insight in learnings.get("insights", []):
        adjustments.append(f"Address: {insight}")
    return {"adjustments": adjustments}
```

### 5. Halt (Check Stop Conditions)

The runner automatically checks stop conditions after each iteration. If any condition triggers, the loop ends.

## Configuration

### RalphConfig

```python
from exo.eval.ralph import RalphConfig, ValidationConfig, ReflectionConfig, StopConditionConfig

config = RalphConfig(
    max_iterations=20,
    validation=ValidationConfig(
        min_score=0.9,            # target score
        required_validators=[],    # optional validator names
    ),
    reflection=ReflectionConfig(
        enabled=True,
        depth="deep",             # reflection depth
    ),
    stop_conditions=StopConditionConfig(
        max_iterations=20,
        timeout_seconds=300,       # 5 minutes
        cost_limit=10.0,          # dollar limit
        consecutive_failures=3,    # stop after 3 failures in a row
        score_threshold=0.95,      # stop when score exceeds this
    ),
)
```

### LoopState

The `LoopState` tracks iteration progress:

```python
from exo.eval.ralph import LoopState

state = LoopState(
    iteration=0,
    scores=[],
    total_cost=0.0,
    start_time=time.time(),
    history=[],
)
```

## Stop Detectors

Stop detectors decide when the loop should terminate. Each detector checks one condition:

### MaxIterationDetector

```python
from exo.eval.ralph import MaxIterationDetector

detector = MaxIterationDetector(max_iterations=20)
decision = detector.check(state)
if decision.should_stop:
    print(f"Stopping: {decision.reason}")
```

### TimeoutDetector

```python
from exo.eval.ralph import TimeoutDetector

detector = TimeoutDetector(timeout_seconds=300)  # 5 minutes
```

### CostLimitDetector

```python
from exo.eval.ralph import CostLimitDetector

detector = CostLimitDetector(cost_limit=10.0)  # $10 budget
```

### ConsecutiveFailureDetector

```python
from exo.eval.ralph import ConsecutiveFailureDetector

detector = ConsecutiveFailureDetector(max_failures=3)
```

### ScoreThresholdDetector

```python
from exo.eval.ralph import ScoreThresholdDetector

detector = ScoreThresholdDetector(threshold=0.95)
```

### CompositeDetector

Combine multiple detectors -- stops when **any** detector triggers:

```python
from exo.eval.ralph import CompositeDetector

detector = CompositeDetector(detectors=[
    MaxIterationDetector(max_iterations=20),
    TimeoutDetector(timeout_seconds=300),
    CostLimitDetector(cost_limit=10.0),
    ScoreThresholdDetector(threshold=0.95),
])

decision = detector.check(state)
```

### Custom Detectors

Subclass `StopDetector` for custom stop conditions:

```python
from exo.eval.ralph import StopDetector, StopDecision

class ConvergenceDetector(StopDetector):
    """Stop when scores stop improving."""

    def __init__(self, window: int = 5, min_improvement: float = 0.01):
        self._window = window
        self._min_improvement = min_improvement

    def check(self, state) -> StopDecision:
        scores = state.scores[-self._window:]
        if len(scores) < self._window:
            return StopDecision(should_stop=False)

        improvement = max(scores) - min(scores)
        if improvement < self._min_improvement:
            return StopDecision(
                should_stop=True,
                reason=f"Score converged (improvement {improvement:.4f} < {self._min_improvement})",
            )
        return StopDecision(should_stop=False)
```

## Advanced Patterns

### Integration with Evaluation

Use evaluation scorers in the analyze phase:

```python
from exo.eval import OutputQualityScorer

quality_scorer = OutputQualityScorer(judge=llm_judge)

async def analyze_fn(result: dict, iteration: int) -> dict:
    score_result = await quality_scorer.score(
        f"iter-{iteration}",
        result.get("input", ""),
        result.get("output", ""),
    )
    return {
        "quality": score_result.score,
        "details": score_result.details,
    }
```

### Integration with Reflection

Use reflectors in the learn phase:

```python
from exo.eval import GeneralReflector

reflector = GeneralReflector(judge=llm_judge)

async def learn_fn(analysis: dict, iteration: int) -> dict:
    reflection = await reflector.reflect({
        "output": analysis.get("output", ""),
        "score": analysis.get("quality", 0),
        "iteration": iteration,
    })
    return {
        "insights": reflection.insights,
        "suggestions": reflection.suggestions,
    }
```

### Progressive Refinement

Pass learnings from each iteration to improve the next:

```python
accumulated_insights = []

async def learn_fn(analysis: dict, iteration: int) -> dict:
    new_insights = extract_insights(analysis)
    accumulated_insights.extend(new_insights)
    return {"insights": accumulated_insights}

async def plan_fn(learnings: dict, iteration: int) -> dict:
    # Use all accumulated insights to plan
    prompt_additions = format_insights(learnings["insights"])
    return {"prompt_suffix": prompt_additions}
```

### Cost-Aware Iteration

Track costs across iterations and stop when budget is exhausted:

```python
async def execute_fn(input: str, iteration: int) -> dict:
    result = await agent.run(input)
    cost = estimate_cost(result.token_count, model="gpt-4o")
    return {"output": result.output, "cost": cost}

config = RalphConfig(
    stop_conditions=StopConditionConfig(
        cost_limit=5.0,  # stop at $5
        score_threshold=0.95,
    ),
)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `RalphRunner` | `exo.eval.ralph` | Main loop runner with 5-phase cycle |
| `RalphResult` | `exo.eval.ralph` | Result: iterations, final_score, stop_reason |
| `RalphConfig` | `exo.eval.ralph` | Configuration: iterations, validation, reflection, stop conditions |
| `LoopState` | `exo.eval.ralph` | Tracks iteration progress, scores, cost |
| `ValidationConfig` | `exo.eval.ralph` | Target score and required validators |
| `ReflectionConfig` | `exo.eval.ralph` | Reflection enablement and depth |
| `StopConditionConfig` | `exo.eval.ralph` | Stop condition parameters |
| `StopDetector` | `exo.eval.ralph` | ABC for stop condition detectors |
| `StopDecision` | `exo.eval.ralph` | Result of a stop check: `should_stop`, `reason` |
| `MaxIterationDetector` | `exo.eval.ralph` | Stop after N iterations |
| `TimeoutDetector` | `exo.eval.ralph` | Stop after timeout |
| `CostLimitDetector` | `exo.eval.ralph` | Stop when cost exceeds budget |
| `ConsecutiveFailureDetector` | `exo.eval.ralph` | Stop after N consecutive failures |
| `ScoreThresholdDetector` | `exo.eval.ralph` | Stop when score exceeds threshold |
| `CompositeDetector` | `exo.eval.ralph` | Combine multiple detectors (any triggers stop) |
