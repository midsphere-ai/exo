# exo.eval

Evaluation and scoring framework with rule-based scorers, LLM-as-Judge assessment, trajectory validation, reflection, and iterative refinement.

## Installation

```bash
uv add exo-eval
```

## Module Path

```python
import exo.eval
```

## Public Exports (31)

| Export | Source Module | Description |
|---|---|---|
| `Evaluator` | `base` | Parallel evaluation runner with pass@k |
| `Scorer` | `base` | Abstract base class for all scorers |
| `ScorerResult` | `base` | Output from a single scorer |
| `EvalCaseResult` | `base` | Result for one input/output pair |
| `EvalResult` | `base` | Aggregated result across all cases |
| `EvalCriteria` | `base` | Threshold-based pass/fail criteria |
| `EvalTarget` | `base` | Abstract evaluation subject |
| `EvalStatus` | `base` | Outcome status enum |
| `EvalError` | `base` | Evaluation error type |
| `FormatValidationScorer` | `scorers` | Format validation (json/xml/yaml/markdown/csv) |
| `SchemaValidationScorer` | `scorers` | JSON Schema validation |
| `OutputCorrectnessScorer` | `scorers` | Ground truth / keyword matching |
| `OutputLengthScorer` | `scorers` | Length constraint checking |
| `OutputRelevanceScorer` | `scorers` | Keyword-overlap relevance |
| `OutputCompletenessScorer` | `scorers` | Required sections checking |
| `LLMAsJudgeScorer` | `llm_scorer` | Base LLM-as-Judge scorer |
| `OutputQualityScorer` | `llm_scorer` | 5-dimensional quality assessment |
| `LogicConsistencyScorer` | `llm_scorer` | Internal contradiction detection |
| `ReasoningValidityScorer` | `llm_scorer` | Argumentation logic validation |
| `ConstraintSatisfactionScorer` | `llm_scorer` | Binary constraint checking |
| `TrajectoryValidator` | `trajectory_scorers` | Trajectory structural integrity |
| `TimeCostScorer` | `trajectory_scorers` | Execution time scoring |
| `AnswerAccuracyLLMScorer` | `trajectory_scorers` | Reference answer comparison |
| `LabelDistributionScorer` | `trajectory_scorers` | Label balance / skew analysis |
| `scorer_register` | `trajectory_scorers` | Decorator to register scorers |
| `get_scorer` | `trajectory_scorers` | Lookup registered scorer by name |
| `list_scorers` | `trajectory_scorers` | List all registered scorer names |
| `Reflector` | `reflection` | Abstract 3-step reflector |
| `GeneralReflector` | `reflection` | LLM-powered reflector |
| `ReflectionHistory` | `reflection` | Tracks reflections over time |
| `ReflectionResult` | `reflection` | Single reflection output |
| `ReflectionType` | `reflection` | Reflection category enum |
| `ReflectionLevel` | `reflection` | Reflection depth enum |

## Import Patterns

```python
# Core evaluation
from exo.eval import Evaluator, Scorer, ScorerResult, EvalCriteria

# Rule-based scorers
from exo.eval import (
    FormatValidationScorer,
    SchemaValidationScorer,
    OutputCorrectnessScorer,
    OutputLengthScorer,
    OutputRelevanceScorer,
    OutputCompletenessScorer,
)

# LLM-as-Judge scorers
from exo.eval import (
    LLMAsJudgeScorer,
    OutputQualityScorer,
    LogicConsistencyScorer,
    ReasoningValidityScorer,
    ConstraintSatisfactionScorer,
)

# Trajectory scorers + registry
from exo.eval import (
    TrajectoryValidator,
    TimeCostScorer,
    AnswerAccuracyLLMScorer,
    LabelDistributionScorer,
    scorer_register,
    get_scorer,
    list_scorers,
)

# Reflection
from exo.eval import (
    Reflector,
    GeneralReflector,
    ReflectionHistory,
    ReflectionResult,
    ReflectionType,
    ReflectionLevel,
)

# Ralph iterative refinement (sub-package)
from exo.eval.ralph.runner import RalphRunner, RalphResult
from exo.eval.ralph.config import RalphConfig, LoopState, StopType
from exo.eval.ralph.detectors import StopDetector, CompositeDetector
```

## Architecture

```
exo.eval
  base.py              Evaluator, Scorer ABC, result types, criteria
  scorers.py           6 rule-based scorers
  llm_scorer.py        LLMAsJudgeScorer + 4 specialized subclasses
  trajectory_scorers.py  Trajectory/time/accuracy scorers + registry
  reflection.py        Reflector ABC, GeneralReflector, history
  ralph/
    config.py          RalphConfig, LoopState, StopType, sub-configs
    runner.py          RalphRunner (5-phase loop)
    detectors.py       StopDetector ABC + 5 built-in + composite
```

## Quick Example

```python
import asyncio
from exo.eval import (
    Evaluator,
    EvalTarget,
    EvalCriteria,
    FormatValidationScorer,
    OutputCorrectnessScorer,
)

class MySystem(EvalTarget):
    async def predict(self, case_id, input):
        return '{"answer": "Paris"}'

async def main():
    evaluator = Evaluator(
        scorers=[
            FormatValidationScorer("json"),
            OutputCorrectnessScorer(keywords=["Paris"]),
        ],
        criteria=[EvalCriteria("format_json", threshold=1.0)],
        parallel=4,
    )

    dataset = [
        {"id": "q1", "input": "What is the capital of France?"},
    ]

    result = await evaluator.evaluate(MySystem(), dataset)
    print(result.summary)
    # {'format_json': 1.0, 'correctness': 1.0}

asyncio.run(main())
```

## Submodule Reference

| Page | Description |
|---|---|
| [base](base.md) | Evaluator, Scorer, result types, criteria, EvalTarget |
| [scorers](scorers.md) | Rule-based format, schema, correctness, length, relevance, completeness scorers |
| [llm-scorer](llm-scorer.md) | LLM-as-Judge scorers for quality, logic, reasoning, constraints |
| [trajectory-scorers](trajectory-scorers.md) | Trajectory validation, time cost, accuracy, label distribution, scorer registry |
| [reflection](reflection.md) | Reflector framework with 3-step pipeline and history tracking |
| [ralph](ralph.md) | Ralph iterative refinement loop (Run-Analyze-Learn-Plan-Halt) |
