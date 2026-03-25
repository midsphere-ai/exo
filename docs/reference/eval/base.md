# exo.eval.base

Core evaluation framework: targets, scorers, criteria, result types, and the parallel evaluator.

## Module Path

```python
from exo.eval.base import (
    EvalError,
    EvalStatus,
    ScorerResult,
    EvalCaseResult,
    EvalResult,
    EvalCriteria,
    EvalTarget,
    Scorer,
    Evaluator,
)
```

---

## EvalError

Raised when an evaluation fails. Inherits from `ExoError`.

```python
class EvalError(ExoError): ...
```

---

## EvalStatus

Outcome status for a single metric evaluation.

```python
class EvalStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
```

| Value | Description |
|---|---|
| `PASSED` | Score meets or exceeds the threshold |
| `FAILED` | Score is below the threshold |
| `NOT_EVALUATED` | No criteria applied (default) |

---

## ScorerResult

Output from a single scorer applied to one case.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
ScorerResult(
    scorer_name: str,
    score: float,
    status: EvalStatus = EvalStatus.NOT_EVALUATED,
    details: dict[str, Any] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `scorer_name` | `str` | *(required)* | Name identifying the scorer |
| `score` | `float` | *(required)* | Numeric score (typically 0.0-1.0) |
| `status` | `EvalStatus` | `NOT_EVALUATED` | Pass/fail status |
| `details` | `dict[str, Any]` | `{}` | Additional scorer-specific information |

---

## EvalCaseResult

Result for one evaluation case (one input/output pair).

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
EvalCaseResult(
    case_id: str,
    input: Any,
    output: Any,
    scores: dict[str, ScorerResult] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `case_id` | `str` | *(required)* | Unique case identifier |
| `input` | `Any` | *(required)* | The input fed to the target |
| `output` | `Any` | *(required)* | The output from the target |
| `scores` | `dict[str, ScorerResult]` | `{}` | Scorer results keyed by scorer name |

---

## EvalResult

Aggregated result across all cases.

**Decorator:** `@dataclass(slots=True)`

### Constructor

```python
EvalResult(
    case_results: list[EvalCaseResult] = [],
    summary: dict[str, Any] = {},
    pass_at_k: dict[int, float] = {},
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `case_results` | `list[EvalCaseResult]` | `[]` | Per-case results |
| `summary` | `dict[str, Any]` | `{}` | Mean score per scorer across all cases |
| `pass_at_k` | `dict[int, float]` | `{}` | Pass@k metrics (populated when `repeat_times > 1` and criteria are set) |

---

## EvalCriteria

Threshold-based pass/fail criteria for a metric.

**Decorator:** `@dataclass(frozen=True, slots=True)`

### Constructor

```python
EvalCriteria(
    metric_name: str,
    threshold: float = 0.5,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metric_name` | `str` | *(required)* | Name matching a scorer's `scorer_name` |
| `threshold` | `float` | `0.5` | Minimum score to pass |

### Methods

#### judge()

```python
def judge(self, value: float) -> EvalStatus
```

Return `PASSED` if `value >= threshold`, else `FAILED`.

---

## EvalTarget (ABC)

Abstract callable evaluation subject -- wraps the system under test.

### Abstract Methods

#### predict()

```python
async def predict(self, case_id: str, input: Any) -> Any
```

Run the system under test and return its output.

| Parameter | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique identifier for this evaluation case |
| `input` | `Any` | Input data from the dataset |

**Returns:** The system's output for scoring.

---

## Scorer (ABC)

Abstract scorer that evaluates one (input, output) pair.

### Abstract Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Score a single case and return a `ScorerResult`.

| Parameter | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique case identifier |
| `input` | `Any` | The input from the dataset |
| `output` | `Any` | The output from the evaluation target |

**Returns:** `ScorerResult`.

---

## Evaluator

Runs an `EvalTarget` over a dataset and scores results. Supports parallel execution via semaphore and `repeat_times` for pass@k metric computation.

### Constructor

```python
Evaluator(
    scorers: list[Scorer],
    *,
    criteria: list[EvalCriteria] | None = None,
    parallel: int = 4,
    repeat_times: int = 1,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scorers` | `list[Scorer]` | *(required)* | List of scorers to apply |
| `criteria` | `list[EvalCriteria] \| None` | `None` | Optional pass/fail criteria per scorer |
| `parallel` | `int` | `4` | Max concurrent evaluations |
| `repeat_times` | `int` | `1` | Repeat each case N times (for pass@k) |

**Raises:** `EvalError` if `parallel < 1` or `repeat_times < 1`.

### Methods

#### evaluate()

```python
async def evaluate(
    self,
    target: EvalTarget,
    dataset: list[dict[str, Any]],
) -> EvalResult
```

Run the target over every case in the dataset, score each case, and return aggregated results.

| Parameter | Type | Description |
|---|---|---|
| `target` | `EvalTarget` | The system under test |
| `dataset` | `list[dict[str, Any]]` | List of cases, each with `"id"` and `"input"` keys |

**Returns:** `EvalResult` with per-case results, summary (mean scores), and pass@k metrics.

**Behavior:**
1. Creates an `asyncio.Semaphore(parallel)` for concurrency control
2. For each case, repeats `repeat_times` invocations
3. Calls `target.predict()` to get output, then each scorer's `score()`
4. If criteria exist for a scorer, applies `judge()` to set status
5. Computes mean score per scorer (summary) and pass@k

### Dunder Methods

| Method | Description |
|---|---|
| `__repr__` | `Evaluator(scorers=3, parallel=4, repeat_times=1)` |

### Example

```python
import asyncio
from exo.eval import (
    Evaluator,
    EvalTarget,
    EvalCriteria,
    FormatValidationScorer,
    OutputCorrectnessScorer,
    OutputLengthScorer,
)

class MyAgent(EvalTarget):
    async def predict(self, case_id, input):
        return f"The answer is {input.get('expected', 'unknown')}"

async def main():
    evaluator = Evaluator(
        scorers=[
            OutputCorrectnessScorer(keywords=["answer"]),
            OutputLengthScorer(min_length=10, max_length=500),
        ],
        criteria=[
            EvalCriteria("correctness", threshold=0.8),
            EvalCriteria("length", threshold=1.0),
        ],
        parallel=8,
        repeat_times=3,  # For pass@k computation
    )

    dataset = [
        {"id": "q1", "input": {"expected": "42"}},
        {"id": "q2", "input": {"expected": "Paris"}},
    ]

    result = await evaluator.evaluate(MyAgent(), dataset)

    # Summary: mean scores per scorer
    print(result.summary)

    # Pass@k metrics
    print(result.pass_at_k)

    # Per-case results
    for cr in result.case_results:
        print(f"Case {cr.case_id}: {cr.scores}")

asyncio.run(main())
```
