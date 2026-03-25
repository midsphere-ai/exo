# exo.eval.trajectory_scorers

Trajectory validation, time cost, accuracy, and label distribution scorers. Includes a scorer registry with `@scorer_register()` decorator for automatic discovery and factory-based creation.

## Module Path

```python
from exo.eval.trajectory_scorers import (
    scorer_register,
    get_scorer,
    list_scorers,
    TrajectoryValidator,
    TimeCostScorer,
    AnswerAccuracyLLMScorer,
    LabelDistributionScorer,
)
```

---

## Scorer Registry

A module-level registry for `Scorer` subclasses, enabling factory-based lookup by name.

### scorer_register()

Decorator that registers a `Scorer` subclass under a given name.

```python
def scorer_register(name: str) -> Callable
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Registry key for the scorer class |

**Returns:** A decorator that registers the class and returns it unchanged.

**Usage:**

```python
from exo.eval import Scorer, ScorerResult, scorer_register

@scorer_register("my_metric")
class MyScorer(Scorer):
    async def score(self, case_id, input, output):
        return ScorerResult(scorer_name="my_metric", score=1.0)
```

### get_scorer()

Lookup a registered scorer class by name.

```python
def get_scorer(name: str) -> type[Scorer]
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Registry key |

**Returns:** The scorer class.

**Raises:** `KeyError` if the name is not registered.

### list_scorers()

Return all registered scorer names (sorted).

```python
def list_scorers() -> list[str]
```

**Returns:** Sorted list of registered scorer names.

### Built-in Registrations

| Name | Class |
|---|---|
| `"trajectory"` | `TrajectoryValidator` |
| `"time_cost"` | `TimeCostScorer` |
| `"answer_accuracy"` | `AnswerAccuracyLLMScorer` |
| `"label_distribution"` | `LabelDistributionScorer` |

---

## TrajectoryValidator

Validates a trajectory (list of step dicts) for structural integrity. Checks each step for required keys and returns the fraction of valid steps.

**Registry name:** `"trajectory"`

### Constructor

```python
TrajectoryValidator(
    *,
    required_keys: Sequence[str] = ("action",),
    name: str = "trajectory",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `required_keys` | `Sequence[str]` | `("action",)` | Keys required in each step dict |
| `name` | `str` | `"trajectory"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

**Output format:** Expects `output` to be:
- A `list[dict]` of step dicts, OR
- A `dict` with a `"trajectory"` key containing a list of step dicts

**Per-step validation:**
- Must have a `"step"` or `"id"` key
- Must have all keys in `required_keys`

**Score:** `valid_steps / total_steps`. Returns `0.0` for empty or invalid trajectories.

**Details:** `{"valid": int, "total": int, "errors": [str]}`.

### Example

```python
import asyncio
from exo.eval import TrajectoryValidator

async def main():
    scorer = TrajectoryValidator(required_keys=("action", "observation"))

    trajectory = [
        {"step": 1, "action": "search", "observation": "found 3 results"},
        {"step": 2, "action": "click"},  # Missing observation
        {"id": "s3", "action": "submit", "observation": "success"},
    ]

    result = await scorer.score("c1", None, trajectory)
    print(f"Score: {result.score:.2f}")  # 0.67 (2 of 3 valid)
    print(result.details["errors"])

asyncio.run(main())
```

---

## TimeCostScorer

Scores based on execution time relative to a maximum budget. Reads `_time_cost_ms` from the output dict.

**Registry name:** `"time_cost"`

### Constructor

```python
TimeCostScorer(*, max_ms: float = 30_000.0, name: str = "time_cost")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_ms` | `float` | `30_000.0` | Maximum time budget in milliseconds |
| `name` | `str` | `"time_cost"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

**Score formula:** `clamp(1.0 - elapsed / max_ms, 0.0, 1.0)`

Reads `output["_time_cost_ms"]` if output is a dict. Falls back to `0.0` elapsed time.

**Details:** `{"elapsed_ms": float, "max_ms": float}`.

### Example

```python
import asyncio
from exo.eval import TimeCostScorer

async def main():
    scorer = TimeCostScorer(max_ms=10_000.0)

    # Fast execution
    r1 = await scorer.score("c1", None, {"_time_cost_ms": 2000.0, "result": "ok"})
    print(f"Score: {r1.score:.2f}")  # 0.80

    # Slow execution
    r2 = await scorer.score("c2", None, {"_time_cost_ms": 15000.0, "result": "ok"})
    print(f"Score: {r2.score:.2f}")  # 0.00

asyncio.run(main())
```

---

## AnswerAccuracyLLMScorer

LLM-as-Judge scorer comparing agent output to a reference answer. Extends `LLMAsJudgeScorer`.

**Registry name:** `"answer_accuracy"`

### Constructor

```python
AnswerAccuracyLLMScorer(
    judge: Any = None,
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    name: str = "answer_accuracy",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `question_key` | `str` | `"question"` | Key in input dict for the question |
| `answer_key` | `str` | `"answer"` | Key in input dict for the reference answer |
| `name` | `str` | `"answer_accuracy"` | Scorer name |

### Overridden Methods

#### build_prompt()

Formats the prompt with three sections: `[Question]`, `[Correct Answer]`, and `[Agent Response]`.

### Expected LLM Response

```json
{"score": 0.85, "explanation": "The answer is mostly correct but missed..."}
```

### Example

```python
import asyncio
from exo.eval import AnswerAccuracyLLMScorer

async def judge(prompt: str) -> str:
    return '{"score": 0.9, "explanation": "Correct with minor omissions."}'

async def main():
    scorer = AnswerAccuracyLLMScorer(judge=judge)

    result = await scorer.score(
        "c1",
        {"question": "What is 2+2?", "answer": "4"},
        "The answer is 4.",
    )
    print(f"Score: {result.score}")  # 0.9

asyncio.run(main())
```

---

## LabelDistributionScorer

Evaluates label balance / distribution skew across a dataset. Per-case score is `0.0` (placeholder). The real value is in the `details` dict and the `summarize()` method.

**Registry name:** `"label_distribution"`

### Constructor

```python
LabelDistributionScorer(*, label_key: str = "label", name: str = "label_distribution")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `label_key` | `str` | `"label"` | Key in the input dict to extract label from |
| `name` | `str` | `"label_distribution"` | Scorer name |

### Methods

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Returns score `0.0` with `details={"label": <value>}`. The label is extracted from `input[label_key]`.

#### summarize()

```python
def summarize(self, results: list[ScorerResult]) -> dict[str, Any]
```

Compute label distribution across all scored cases.

| Parameter | Type | Description |
|---|---|---|
| `results` | `list[ScorerResult]` | Scored results to aggregate |

**Returns:** Dict with:

| Key | Type | Description |
|---|---|---|
| `labels` | `list` | Sorted unique labels |
| `fractions` | `list[float]` | Fraction for each label |
| `counts` | `dict` | Raw counts per label |
| `skew` | `float` | `max_fraction - min_fraction` (0 = perfectly balanced) |

### Example

```python
import asyncio
from exo.eval import LabelDistributionScorer

async def main():
    scorer = LabelDistributionScorer(label_key="category")

    cases = [
        {"category": "positive"},
        {"category": "positive"},
        {"category": "negative"},
        {"category": "neutral"},
    ]

    results = []
    for i, case in enumerate(cases):
        r = await scorer.score(f"c{i}", case, "output")
        results.append(r)

    summary = scorer.summarize(results)
    print(summary["counts"])    # {'negative': 1, 'neutral': 1, 'positive': 2}
    print(f"Skew: {summary['skew']:.2f}")  # 0.25

asyncio.run(main())
```
