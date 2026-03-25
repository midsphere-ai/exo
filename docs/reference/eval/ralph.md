# exo.eval.ralph

Ralph iterative refinement loop: Run, Analyze, Learn, Plan, Halt. A 5-phase execution engine that combines scoring, reflection, and re-prompting to iteratively improve agent outputs.

## Module Paths

```python
from exo.eval.ralph.runner import RalphRunner, RalphResult, ExecuteFn, RePlanFn
from exo.eval.ralph.config import (
    StopType,
    ValidationConfig,
    ReflectionConfig,
    StopConditionConfig,
    RalphConfig,
    LoopState,
)
from exo.eval.ralph.detectors import (
    StopDecision,
    StopDetector,
    MaxIterationDetector,
    TimeoutDetector,
    CostLimitDetector,
    ConsecutiveFailureDetector,
    ScoreThresholdDetector,
    CompositeDetector,
)
```

---

## StopType

Categorised exit reason for loop termination.

```python
class StopType(StrEnum):
    NONE = "none"
    COMPLETION = "completion"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    MAX_COST = "max_cost"
    MAX_CONSECUTIVE_FAILURES = "max_consecutive_failures"
    SCORE_THRESHOLD = "score_threshold"
    USER_INTERRUPTED = "user_interrupted"
    SYSTEM_ERROR = "system_error"
```

| Value | Description |
|---|---|
| `NONE` | No stop condition triggered |
| `COMPLETION` | Task completed successfully |
| `MAX_ITERATIONS` | Reached maximum iteration count |
| `TIMEOUT` | Wall-clock time limit exceeded |
| `MAX_COST` | Cumulative cost limit reached |
| `MAX_CONSECUTIVE_FAILURES` | Too many failures in a row |
| `SCORE_THRESHOLD` | Score meets or exceeds threshold |
| `USER_INTERRUPTED` | User requested interruption |
| `SYSTEM_ERROR` | Unrecoverable system error |

### Methods

#### is_success()

```python
def is_success(self) -> bool
```

Returns `True` for `COMPLETION` and `SCORE_THRESHOLD`.

#### is_failure()

```python
def is_failure(self) -> bool
```

Returns `True` for `MAX_CONSECUTIVE_FAILURES` and `SYSTEM_ERROR`.

---

## ValidationConfig

Configuration for the Analyze (scoring) phase.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
ValidationConfig(
    enabled: bool = True,
    scorer_names: tuple[str, ...] = (),
    min_score_threshold: float = 0.5,
    parallel: int = 4,
    timeout: float = 0.0,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Whether scoring is enabled |
| `scorer_names` | `tuple[str, ...]` | `()` | Names of scorers to use (empty = use all) |
| `min_score_threshold` | `float` | `0.5` | Below this mean score, reflection is triggered |
| `parallel` | `int` | `4` | Max parallel scorer executions |
| `timeout` | `float` | `0.0` | Scorer timeout (0 = no timeout) |

**Raises:** `ValueError` if `min_score_threshold` is not in `[0.0, 1.0]`.

---

## ReflectionConfig

Configuration for the Learn (reflection) phase.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
ReflectionConfig(
    enabled: bool = True,
    level: str = "medium",
    max_history: int = 50,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Whether reflection is enabled |
| `level` | `str` | `"medium"` | Reflection depth level |
| `max_history` | `int` | `50` | Maximum reflection history entries |

---

## StopConditionConfig

Configuration for the Halt (stop detection) phase.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
StopConditionConfig(
    max_iterations: int = 10,
    timeout: float = 0.0,
    max_cost: float = 0.0,
    max_consecutive_failures: int = 3,
    score_threshold: float = 0.0,
    enable_user_interrupt: bool = False,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `int` | `10` | Stop after this many iterations |
| `timeout` | `float` | `0.0` | Wall-clock timeout in seconds (0 = disabled) |
| `max_cost` | `float` | `0.0` | Cumulative cost limit (0 = disabled) |
| `max_consecutive_failures` | `int` | `3` | Stop after N consecutive failures |
| `score_threshold` | `float` | `0.0` | Stop when mean score reaches this (0 = disabled) |
| `enable_user_interrupt` | `bool` | `False` | Allow user-initiated interruption |

**Raises:** `ValueError` if `max_iterations < 1`.

---

## RalphConfig

Unified configuration for the Ralph iterative refinement loop. Aggregates validation, reflection, and stop-condition settings.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
RalphConfig(
    validation: ValidationConfig = ValidationConfig(),
    reflection: ReflectionConfig = ReflectionConfig(),
    stop_condition: StopConditionConfig = StopConditionConfig(),
    metadata: dict[str, Any] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `validation` | `ValidationConfig` | `ValidationConfig()` | Scoring phase configuration |
| `reflection` | `ReflectionConfig` | `ReflectionConfig()` | Reflection phase configuration |
| `stop_condition` | `StopConditionConfig` | `StopConditionConfig()` | Halt phase configuration |
| `metadata` | `dict[str, Any]` | `{}` | Custom metadata |

---

## LoopState

Mutable runtime state for a Ralph loop execution. Tracks iteration count, timing, cost, and aggregated score/reflection history.

**Decorator:** `@dataclass(slots=True)`

### Constructor

```python
LoopState()
```

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `iteration` | `int` | `0` | Current iteration number |
| `start_time` | `float` | `time.monotonic()` | Loop start timestamp |
| `cumulative_cost` | `float` | `0.0` | Total cost across iterations |
| `consecutive_failures` | `int` | `0` | Current consecutive failure streak |
| `successful_steps` | `int` | `0` | Total successful steps |
| `failed_steps` | `int` | `0` | Total failed steps |
| `total_tokens` | `int` | `0` | Total tokens consumed |
| `score_history` | `list[dict[str, float]]` | `[]` | Score snapshots per iteration |
| `reflection_history` | `list[dict[str, Any]]` | `[]` | Reflection summaries per iteration |
| `metadata` | `dict[str, Any]` | `{}` | Custom metadata |

### Query Methods

#### elapsed()

```python
def elapsed(self) -> float
```

Seconds since the loop started.

#### success_rate()

```python
def success_rate(self) -> float
```

Fraction of successful steps. Returns `0.0` when no steps have been executed.

#### latest_score()

```python
def latest_score(self) -> dict[str, float]
```

Return the most recent score snapshot, or empty dict.

#### best_score()

```python
def best_score(self, metric: str) -> float
```

Return the highest value seen for a metric across all iterations.

| Parameter | Type | Description |
|---|---|---|
| `metric` | `str` | Scorer name to look up |

### Mutation Methods

#### record_score()

```python
def record_score(self, scores: dict[str, float]) -> None
```

Append a score snapshot for the current iteration.

#### record_reflection()

```python
def record_reflection(self, reflection: dict[str, Any]) -> None
```

Append a reflection summary for the current iteration.

#### record_success()

```python
def record_success(self, *, tokens: int = 0, cost: float = 0.0) -> None
```

Mark the current step as successful. Resets `consecutive_failures` to 0.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tokens` | `int` | `0` | Tokens consumed |
| `cost` | `float` | `0.0` | Cost incurred |

#### record_failure()

```python
def record_failure(self, *, cost: float = 0.0) -> None
```

Mark the current step as failed. Increments `consecutive_failures`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cost` | `float` | `0.0` | Cost incurred |

### Serialization

#### to_dict()

```python
def to_dict(self) -> dict[str, Any]
```

Serialise to a plain dict for checkpointing.

### Dunder Methods

| Method | Description |
|---|---|
| `__repr__` | `LoopState(iteration=5, success_rate=80.0%, elapsed=12.3s)` |

---

## StopDecision

Outcome of a single detector evaluation.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
StopDecision(
    should_stop: bool,
    stop_type: StopType = StopType.NONE,
    reason: str = "",
    metadata: dict[str, Any] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `should_stop` | `bool` | *(required)* | Whether the loop should stop |
| `stop_type` | `StopType` | `NONE` | The category of stop condition |
| `reason` | `str` | `""` | Human-readable explanation |
| `metadata` | `dict[str, Any]` | `{}` | Additional decision context |

### Dunder Methods

| Method | Description |
|---|---|
| `__bool__` | Returns `should_stop` |

---

## StopDetector (ABC)

Base class for pluggable stop-condition detectors. Each detector examines the current `LoopState` and the static `StopConditionConfig`.

### Abstract Methods

#### check()

```python
async def check(
    self,
    state: LoopState,
    config: StopConditionConfig,
) -> StopDecision
```

Return a stop decision for the current loop state.

---

## Built-in Detectors

### MaxIterationDetector

Stops when the loop has reached `config.max_iterations`.

```python
MaxIterationDetector()
```

Triggers `StopType.MAX_ITERATIONS`.

### TimeoutDetector

Stops when elapsed wall-clock time exceeds `config.timeout`. A timeout of `0.0` (the default) disables this detector.

```python
TimeoutDetector()
```

Triggers `StopType.TIMEOUT`.

### CostLimitDetector

Stops when cumulative cost meets or exceeds `config.max_cost`. A `max_cost` of `0.0` (the default) disables this detector.

```python
CostLimitDetector()
```

Triggers `StopType.MAX_COST`.

### ConsecutiveFailureDetector

Stops when consecutive failures reach `config.max_consecutive_failures`. A value of `0` disables this detector.

```python
ConsecutiveFailureDetector()
```

Triggers `StopType.MAX_CONSECUTIVE_FAILURES`.

### ScoreThresholdDetector

Stops when the mean of the latest score snapshot meets or exceeds `config.score_threshold`. A `score_threshold` of `0.0` (the default) disables this detector.

```python
ScoreThresholdDetector()
```

Triggers `StopType.SCORE_THRESHOLD`.

---

## CompositeDetector

Aggregates multiple detectors and returns the first triggered decision.

### Constructor

```python
CompositeDetector(detectors: list[StopDetector] | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detectors` | `list[StopDetector] \| None` | `None` | List of detectors to run (empty if `None`) |

### Methods

#### add()

```python
def add(self, detector: StopDetector) -> CompositeDetector
```

Append a detector. Returns `self` for chaining.

#### check()

```python
async def check(
    self,
    state: LoopState,
    config: StopConditionConfig,
) -> StopDecision
```

Iterates through all detectors in order. Returns the first `StopDecision` where `should_stop=True`, or a "continue" decision if none trigger.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of detectors |
| `__repr__` | `CompositeDetector(detectors=5)` |

---

## Type Aliases

```python
ExecuteFn = Callable[..., Any]  # Async callable: (input: str) -> str
RePlanFn = Callable[..., Any]   # Async callable: (prompt: str) -> str
```

---

## RalphResult

Final outcome of a Ralph loop execution.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
RalphResult(
    output: str,
    stop_type: StopType,
    reason: str,
    iterations: int,
    scores: dict[str, float],
    state: dict[str, Any],
    reflections: list[dict[str, Any]] = [],
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `output` | `str` | *(required)* | Final output from the last iteration |
| `stop_type` | `StopType` | *(required)* | Why the loop stopped |
| `reason` | `str` | *(required)* | Human-readable stop reason |
| `iterations` | `int` | *(required)* | Total iterations executed |
| `scores` | `dict[str, float]` | *(required)* | Final scores from the last iteration |
| `state` | `dict[str, Any]` | *(required)* | Serialized `LoopState` |
| `reflections` | `list[dict[str, Any]]` | `[]` | Reflection summaries per iteration |

---

## RalphRunner

Implements the 5-phase Ralph iterative refinement loop.

### Phases Per Iteration

1. **Run** -- Execute the agent/task via `execute_fn`
2. **Analyze** -- Score the output using configured scorers
3. **Learn** -- Reflect on failures to extract actionable insights
4. **Plan** -- Re-prompt by appending reflection suggestions to input
5. **Halt** -- Check stop conditions; break or continue

### Constructor

```python
RalphRunner(
    execute_fn: ExecuteFn,
    scorers: list[Scorer],
    *,
    config: RalphConfig | None = None,
    reflector: Reflector | None = None,
    replan_fn: RePlanFn | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `execute_fn` | `ExecuteFn` | *(required)* | Async callable to run the task |
| `scorers` | `list[Scorer]` | *(required)* | Scorers for the Analyze phase |
| `config` | `RalphConfig \| None` | `None` | Loop configuration (defaults to `RalphConfig()`) |
| `reflector` | `Reflector \| None` | `None` | Reflector for the Learn phase |
| `replan_fn` | `RePlanFn \| None` | `None` | Custom re-plan function (unused in current implementation) |

### Methods

#### run()

```python
async def run(self, input: str) -> RalphResult
```

Execute the full Ralph loop on the given input and return the result.

| Parameter | Type | Description |
|---|---|---|
| `input` | `str` | Initial input to the task |

**Returns:** `RalphResult` with the final output, stop type, scores, and reflections.

**Behavior per iteration:**
1. **Run:** Calls `execute_fn(current_input)`. On success, records success. On exception, records failure.
2. **Analyze:** If validation is enabled and execution succeeded, runs all scorers and records scores.
3. **Learn:** If reflection is enabled and either execution failed or mean score < `min_score_threshold`, calls `reflector.reflect()`.
4. **Plan:** If reflection produced suggestions, appends them to the original input as `[Previous feedback]`.
5. **Halt:** Runs all stop detectors via `CompositeDetector`. Stops on the first triggered condition.

### Built-in Detectors

The runner automatically creates a `CompositeDetector` with these detectors (in order):

1. `MaxIterationDetector`
2. `TimeoutDetector`
3. `CostLimitDetector`
4. `ConsecutiveFailureDetector`
5. `ScoreThresholdDetector`

### Dunder Methods

| Method | Description |
|---|---|
| `__repr__` | `RalphRunner(scorers=3, config=RalphConfig(...))` |

### Example

```python
import asyncio
from exo.eval import (
    GeneralReflector,
    OutputCorrectnessScorer,
    OutputLengthScorer,
)
from exo.eval.ralph.runner import RalphRunner
from exo.eval.ralph.config import (
    RalphConfig,
    ValidationConfig,
    ReflectionConfig,
    StopConditionConfig,
)

iteration_count = 0

async def my_agent(input: str) -> str:
    global iteration_count
    iteration_count += 1
    if iteration_count >= 3:
        return "The capital of France is Paris."
    return "I'm not sure about that."

async def my_judge(prompt: str) -> str:
    return '''{
        "summary": "The answer was incomplete.",
        "key_findings": ["Missing specific answer"],
        "root_causes": ["Insufficient confidence"],
        "insights": ["Need to be more decisive"],
        "suggestions": ["Provide a direct, specific answer"]
    }'''

async def main():
    config = RalphConfig(
        validation=ValidationConfig(min_score_threshold=0.8),
        reflection=ReflectionConfig(enabled=True),
        stop_condition=StopConditionConfig(
            max_iterations=5,
            score_threshold=0.9,
        ),
    )

    runner = RalphRunner(
        execute_fn=my_agent,
        scorers=[
            OutputCorrectnessScorer(keywords=["Paris", "capital"]),
            OutputLengthScorer(min_length=10),
        ],
        config=config,
        reflector=GeneralReflector(judge=my_judge),
    )

    result = await runner.run("What is the capital of France?")

    print(f"Output: {result.output}")
    print(f"Stop type: {result.stop_type}")
    print(f"Iterations: {result.iterations}")
    print(f"Scores: {result.scores}")
    print(f"Success: {result.stop_type.is_success()}")

    # Check reflections
    for ref in result.reflections:
        print(f"  Iteration {ref['iteration']}: {ref['summary']}")

asyncio.run(main())
```
