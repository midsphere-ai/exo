# Evaluation

The `exo-eval` package provides a framework for evaluating agent outputs with pluggable scorers, parallel execution, pass@k computation, and both rule-based and LLM-as-Judge scoring. It also includes a reflection system for structured analysis of agent behavior.

## Basic Usage

```python
from exo.eval import Evaluator, EvalTarget, Scorer, ScorerResult

# Define an evaluation target
class MyTarget(EvalTarget):
    async def run(self, input: str) -> str:
        # Run your agent and return output
        return await my_agent.execute(input)

# Create scorers
from exo.eval import OutputCorrectnessScorer, FormatValidationScorer

scorers = [
    OutputCorrectnessScorer(expected_outputs={"q1": "42", "q2": "yes"}),
    FormatValidationScorer(format_type="json"),
]

# Run evaluation
evaluator = Evaluator(target=MyTarget(), scorers=scorers)
result = await evaluator.evaluate(
    cases=[
        {"id": "q1", "input": "What is 6 * 7?"},
        {"id": "q2", "input": "Is Python interpreted?"},
    ],
    concurrency=4,
)

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Avg score: {result.avg_score:.3f}")
```

## Evaluation Targets

Subclass `EvalTarget` to define what is being evaluated:

```python
from exo.eval import EvalTarget

class AgentTarget(EvalTarget):
    def __init__(self, agent):
        self._agent = agent

    async def run(self, input: str) -> str:
        result = await self._agent.execute(input)
        return result.output
```

## Rule-Based Scorers

### FormatValidationScorer

Validates output format (JSON, XML, YAML, Markdown, CSV):

```python
from exo.eval import FormatValidationScorer

scorer = FormatValidationScorer(format_type="json")
result = await scorer.score("case-1", "input", '{"key": "value"}')
print(result.score)  # 1.0 (valid JSON)
```

Supported formats: `"json"`, `"xml"`, `"yaml"`, `"markdown"`, `"csv"`.

### SchemaValidationScorer

Validates output against a JSON schema:

```python
from exo.eval import SchemaValidationScorer

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
scorer = SchemaValidationScorer(schema=schema)
```

### OutputCorrectnessScorer

Compares output against expected values:

```python
from exo.eval import OutputCorrectnessScorer

scorer = OutputCorrectnessScorer(
    expected_outputs={"case-1": "42", "case-2": "yes"},
)
result = await scorer.score("case-1", "What is 6*7?", "42")
print(result.score)  # 1.0
```

### OutputLengthScorer

Scores based on output length within a target range:

```python
from exo.eval import OutputLengthScorer

scorer = OutputLengthScorer(min_length=50, max_length=500)
```

### OutputRelevanceScorer

Checks if the output addresses the input topic:

```python
from exo.eval import OutputRelevanceScorer

scorer = OutputRelevanceScorer()
```

### OutputCompletenessScorer

Evaluates whether the output covers all required aspects:

```python
from exo.eval import OutputCompletenessScorer

scorer = OutputCompletenessScorer()
```

## LLM-as-Judge Scorers

### OutputQualityScorer

Evaluates output quality across five dimensions: accuracy, clarity, completeness, relevance, and coherence:

```python
from exo.eval import OutputQualityScorer

scorer = OutputQualityScorer(
    judge=my_llm_callable,  # async (prompt: str) -> str
)
result = await scorer.score("case-1", "Explain Python", "Python is...")
print(result.score)           # 0.0-1.0
print(result.details)         # per-dimension scores
```

### LogicConsistencyScorer

Checks for internal logical consistency:

```python
from exo.eval import LogicConsistencyScorer

scorer = LogicConsistencyScorer(judge=my_llm_callable)
```

### ReasoningValidityScorer

Evaluates the validity of reasoning chains:

```python
from exo.eval import ReasoningValidityScorer

scorer = ReasoningValidityScorer(judge=my_llm_callable)
```

### ConstraintSatisfactionScorer

Checks if output satisfies specified constraints:

```python
from exo.eval import ConstraintSatisfactionScorer

scorer = ConstraintSatisfactionScorer(judge=my_llm_callable)
```

### Custom LLM-as-Judge

Subclass `LLMAsJudgeScorer` for custom evaluation criteria:

```python
from exo.eval import LLMAsJudgeScorer

class ToneScorer(LLMAsJudgeScorer):
    def build_prompt(self, case_id: str, input: str, output: str) -> str:
        return (
            f"Evaluate whether this response has a professional tone.\n\n"
            f"[Input]\n{input}\n\n"
            f"[Output]\n{output}\n\n"
            f'Return JSON: {{"score": <0.0-1.0>, "explanation": "<reason>"}}'
        )

scorer = ToneScorer(judge=my_llm_callable, name="tone")
```

## Trajectory Scorers

### TrajectoryValidator

Validates structural integrity of agent execution trajectories:

```python
from exo.eval import TrajectoryValidator

scorer = TrajectoryValidator(required_keys=["action"])
result = await scorer.score("case-1", "input", [
    {"step": 1, "action": "search"},
    {"step": 2, "action": "summarize"},
])
print(result.score)  # 1.0 (all steps valid)
```

### TimeCostScorer

Scores based on execution time relative to a budget:

```python
from exo.eval import TimeCostScorer

scorer = TimeCostScorer(max_ms=30_000)
result = await scorer.score("case-1", "input", {"_time_cost_ms": 15000})
print(result.score)  # 0.5 (used half the budget)
```

### AnswerAccuracyLLMScorer

LLM-judged comparison of agent output to a reference answer:

```python
from exo.eval import AnswerAccuracyLLMScorer

scorer = AnswerAccuracyLLMScorer(judge=my_llm_callable)
result = await scorer.score(
    "case-1",
    {"question": "What is 2+2?", "answer": "4"},
    "The answer is 4.",
)
```

### Scorer Registry

Scorers decorated with `@scorer_register` can be looked up by name:

```python
from exo.eval import get_scorer, list_scorers

# List all registered scorers
print(list_scorers())  # ["answer_accuracy", "label_distribution", "time_cost", "trajectory"]

# Look up by name
ScorerClass = get_scorer("trajectory")
scorer = ScorerClass()
```

## Evaluation Criteria

Set pass/fail thresholds:

```python
from exo.eval import EvalCriteria

criteria = EvalCriteria(
    min_score=0.8,      # minimum average score to pass
    min_pass_rate=0.9,  # minimum fraction of cases that must pass
)

result = await evaluator.evaluate(cases, criteria=criteria)
print(result.passed)  # True/False based on criteria
```

## Pass@k Computation

Evaluate with multiple attempts per case:

```python
result = await evaluator.evaluate(
    cases=test_cases,
    k=3,  # 3 attempts per case
)
print(result.pass_at_k)  # pass@3 metric
```

## Reflection System

The reflection framework provides structured analysis of agent execution:

```python
from exo.eval import GeneralReflector, ReflectionType, ReflectionLevel

reflector = GeneralReflector(
    judge=my_llm_callable,
    level=ReflectionLevel.DEEP,
)

result = await reflector.reflect({
    "input": "Write a Python function",
    "output": "def hello(): ...",
    "iteration": 1,
})

print(result.summary)
print(result.key_findings)
print(result.root_causes)
print(result.insights)
print(result.suggestions)
```

### Reflection History

Track reflections over time:

```python
from exo.eval import ReflectionHistory

history = ReflectionHistory()
history.add(reflection_result)

recent = history.get_recent(n=5)
stats = history.summarize()
print(stats)  # {"total": 10, "success": 7, "failure": 3, "types": {...}}
```

### Custom Reflectors

Subclass `Reflector` for domain-specific analysis:

```python
from exo.eval import Reflector

class CodeReflector(Reflector):
    async def analyze(self, context: dict) -> dict:
        code = context.get("output", "")
        findings = []
        if "try:" not in code:
            findings.append("No error handling found")
        if "def " not in code:
            findings.append("No function definitions")
        return {
            "summary": f"Code analysis: {len(findings)} issues",
            "key_findings": findings,
        }
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Evaluator` | `exo.eval` | Parallel evaluation runner with pass@k |
| `EvalTarget` | `exo.eval` | ABC for evaluation targets |
| `Scorer` | `exo.eval` | ABC for scoring functions |
| `ScorerResult` | `exo.eval` | Score result with details |
| `EvalCriteria` | `exo.eval` | Pass/fail thresholds |
| `EvalResult` | `exo.eval` | Aggregate evaluation result |
| `FormatValidationScorer` | `exo.eval` | Validate output format |
| `SchemaValidationScorer` | `exo.eval` | Validate against JSON schema |
| `OutputCorrectnessScorer` | `exo.eval` | Compare to expected outputs |
| `OutputQualityScorer` | `exo.eval` | LLM-judged 5-dimension quality |
| `LogicConsistencyScorer` | `exo.eval` | LLM-judged logical consistency |
| `TrajectoryValidator` | `exo.eval` | Validate trajectory structure |
| `TimeCostScorer` | `exo.eval` | Score by execution time |
| `AnswerAccuracyLLMScorer` | `exo.eval` | LLM-judged answer accuracy |
| `GeneralReflector` | `exo.eval` | LLM-powered execution reflection |
| `ReflectionHistory` | `exo.eval` | Track reflection results over time |
| `scorer_register` | `exo.eval` | Decorator for scorer registry |
| `get_scorer` | `exo.eval` | Look up scorer by name |
| `list_scorers` | `exo.eval` | List all registered scorer names |
