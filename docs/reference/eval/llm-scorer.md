# exo.eval.llm_scorer

LLM-as-Judge scorers and multi-dimensional quality assessment. Delegates evaluation to an LLM via a judge callable.

## Module Path

```python
from exo.eval.llm_scorer import (
    extract_json,
    LLMAsJudgeScorer,
    OutputQualityScorer,
    LogicConsistencyScorer,
    ReasoningValidityScorer,
    ConstraintSatisfactionScorer,
)
```

---

## extract_json()

Extract the first JSON object from text (supports nested braces).

```python
def extract_json(text: str) -> dict[str, Any]
```

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Text containing a JSON object |

**Returns:** The first valid JSON object found, or `{}` if none found.

**Behavior:** Scans for `{`, tracks brace depth, attempts `json.loads()` on each candidate substring. Falls back to the next `{` on parse failure.

---

## LLMAsJudgeScorer

Scorer that delegates evaluation to an LLM judge. Subclass and override `build_prompt()` and `parse_response()` for domain-specific judges, or use directly with a custom `system_prompt` and a `judge` callable.

The `judge` is an async callable `(prompt: str) -> str` -- any function that takes a prompt and returns the LLM response text. This keeps the scorer decoupled from a specific model provider.

### Constructor

```python
LLMAsJudgeScorer(
    judge: Any = None,
    *,
    system_prompt: str | None = None,
    name: str = "llm_judge",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `system_prompt` | `str \| None` | `None` | Custom system prompt (uses default if `None`) |
| `name` | `str` | `"llm_judge"` | Scorer name |

### Methods

#### build_prompt()

```python
def build_prompt(self, case_id: str, input: Any, output: Any) -> str
```

Build the user-facing prompt sent to the judge LLM. Override for custom prompt formats.

Default format:
```
{system_prompt}

[Input]
{input}
[Output]
{output}

Return a JSON object with at minimum {"score": <float 0.0-1.0>}.
```

#### parse_response()

```python
def parse_response(self, response: str) -> tuple[float, dict[str, Any]]
```

Extract score and details from the judge LLM response. Uses `extract_json()` to find the JSON object, then clamps the `"score"` field to `[0.0, 1.0]`.

**Returns:** `(score, details_dict)`.

#### score()

```python
async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult
```

Score a single case. If no judge callable is provided, returns score `0.0` with error details.

### Default System Prompt

```
You are an expert evaluator. Score the output on a scale of 0.0 to 1.0.
Respond with a JSON object: {"score": <float>, "explanation": "<reasoning>"}.
```

### Example

```python
import asyncio
from exo.eval import LLMAsJudgeScorer

async def my_llm_judge(prompt: str) -> str:
    # In practice, call your LLM provider here
    return '{"score": 0.85, "explanation": "Clear and accurate response."}'

async def main():
    scorer = LLMAsJudgeScorer(
        judge=my_llm_judge,
        system_prompt="Evaluate the output for technical accuracy.",
        name="accuracy",
    )

    result = await scorer.score(
        "c1",
        "Explain recursion",
        "Recursion is when a function calls itself to solve smaller subproblems.",
    )
    print(f"Score: {result.score}")  # 0.85
    print(result.details)

asyncio.run(main())
```

---

## OutputQualityScorer

Weighted 5-dimensional quality scorer. Extends `LLMAsJudgeScorer` with structured multi-dimensional scoring.

### Default Dimensions and Weights

| Dimension | Weight |
|---|---|
| `correctness` | 0.40 |
| `relevance` | 0.20 |
| `completeness` | 0.20 |
| `clarity` | 0.10 |
| `professionalism` | 0.10 |

### Quality Labels

| Threshold | Label |
|---|---|
| >= 0.90 | Excellent |
| >= 0.80 | Good |
| >= 0.60 | Medium |
| >= 0.40 | Pass |
| < 0.40 | Fail |

### Constructor

```python
OutputQualityScorer(
    judge: Any = None,
    *,
    dimensions: dict[str, float] | None = None,
    name: str = "output_quality",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `dimensions` | `dict[str, float] \| None` | `None` | Custom dimension weights (uses defaults if `None`) |
| `name` | `str` | `"output_quality"` | Scorer name |

### Overridden Methods

#### build_prompt()

Requests the LLM to score each dimension individually and return:
```json
{
    "dimension_scores": {"correctness": 0.9, "relevance": 0.8, ...},
    "score": 0.85,
    "quality_label": "Good",
    "reason": "..."
}
```

#### parse_response()

Computes weighted score from dimension scores. Assigns a quality label based on the thresholds above.

### Example

```python
import asyncio
from exo.eval import OutputQualityScorer

async def judge(prompt: str) -> str:
    return '''{
        "dimension_scores": {
            "correctness": 0.9,
            "relevance": 0.8,
            "completeness": 0.7,
            "clarity": 0.9,
            "professionalism": 0.8
        }
    }'''

async def main():
    scorer = OutputQualityScorer(judge=judge)
    result = await scorer.score("c1", "Explain Python", "Python is a language...")
    print(f"Score: {result.score:.2f}")  # Weighted average
    print(result.details["quality_label"])

asyncio.run(main())
```

---

## LogicConsistencyScorer

Detects internal contradictions, causal fallacies, and data inconsistencies. Extends `LLMAsJudgeScorer`.

### Sub-scores and Weights

| Sub-score | Weight |
|---|---|
| `contradiction_score` | 0.5 |
| `causal_score` | 0.3 |
| `data_score` | 0.2 |

### Constructor

```python
LogicConsistencyScorer(judge: Any = None, *, name: str = "logic_consistency")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `name` | `str` | `"logic_consistency"` | Scorer name |

### Expected LLM Response

```json
{
    "contradiction_score": 0.9,
    "causal_score": 0.8,
    "data_score": 0.7,
    "score": 0.85,
    "issues": ["Minor temporal inconsistency in paragraph 3"]
}
```

### Overridden Methods

#### parse_response()

Computes weighted total from the three sub-scores and clamps to `[0.0, 1.0]`.

---

## ReasoningValidityScorer

Validates argumentation logic and detects formal/informal fallacies. Extends `LLMAsJudgeScorer`.

### Constructor

```python
ReasoningValidityScorer(judge: Any = None, *, name: str = "reasoning_validity")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `name` | `str` | `"reasoning_validity"` | Scorer name |

### Expected LLM Response

```json
{
    "score": 0.75,
    "is_valid": true,
    "fallacies": ["hasty generalization"],
    "reasoning_type": "inductive",
    "explanation": "The argument uses inductive reasoning but contains..."
}
```

Uses the default `parse_response()` from `LLMAsJudgeScorer` (extracts `"score"` from JSON).

---

## ConstraintSatisfactionScorer

Binary constraint checking -- PASS/FAIL per constraint, no partial credit. Extends `LLMAsJudgeScorer`.

### Constructor

```python
ConstraintSatisfactionScorer(
    constraints: list[str],
    judge: Any = None,
    *,
    name: str = "constraint_satisfaction",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `constraints` | `list[str]` | *(required)* | List of constraint descriptions |
| `judge` | `Any` | `None` | Async callable `(prompt: str) -> str` |
| `name` | `str` | `"constraint_satisfaction"` | Scorer name |

### Overridden Methods

#### build_prompt()

Formats constraints as a numbered list and asks the LLM to evaluate each:
```
Constraints:
  1. Must be under 500 words
  2. Must include a code example
  3. Must cite at least one source
```

#### parse_response()

Computes score from individual constraint results. Score = `passed_count / total_constraints`. Falls back to the `"score"` field from the JSON if no `constraint_results` array is present.

### Expected LLM Response

```json
{
    "constraint_results": [
        {"id": 1, "status": "PASS"},
        {"id": 2, "status": "FAIL"},
        {"id": 3, "status": "PASS"}
    ],
    "score": 0.67
}
```

### Example

```python
import asyncio
from exo.eval import ConstraintSatisfactionScorer

async def judge(prompt: str) -> str:
    return '''{
        "constraint_results": [
            {"id": 1, "status": "PASS"},
            {"id": 2, "status": "PASS"},
            {"id": 3, "status": "FAIL"}
        ],
        "score": 0.67
    }'''

async def main():
    scorer = ConstraintSatisfactionScorer(
        constraints=[
            "Response must be in English",
            "Response must include an example",
            "Response must not exceed 200 words",
        ],
        judge=judge,
    )

    result = await scorer.score("c1", "Explain OOP", "Object-oriented programming is...")
    print(f"Score: {result.score:.2f}")  # 0.67
    print(result.details["constraint_results"])

asyncio.run(main())
```
