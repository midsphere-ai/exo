# exo-eval

Evaluation and scoring framework for the [Exo](../../README.md) multi-agent framework. Test agent quality with rule-based scorers, LLM-as-judge, and reflection.

## Installation

```bash
pip install exo-eval
```

Requires Python 3.11+ and `exo-core`.

## What's Included

- **Evaluator** -- run agents against test cases with parallel execution and pass@k support.
- **Rule-based scorers** -- exact match, contains, regex, JSON schema validation, format validators.
- **LLM-as-judge scorer** -- use an LLM to evaluate agent output quality.
- **Trajectory scorers** -- evaluate the agent's step-by-step execution path.
- **Reflection** -- iterative self-improvement via agent reflection on past performance.
- **Ralph Loop** -- Run-Analyze-Learn-Plan-Halt iterative refinement cycle.

## Quick Example

```python
from exo.eval import Evaluator, ExactMatchScorer

evaluator = Evaluator(
    agent=my_agent,
    scorers=[ExactMatchScorer()],
    test_cases=[
        {"input": "What is 2+2?", "expected": "4"},
    ],
)

results = await evaluator.run()
print(f"Pass rate: {results.pass_rate:.0%}")
```

## Documentation

- [Evaluation Guide](../../docs/guides/evaluation.md)
- [Ralph Loop Guide](../../docs/guides/ralph-loop.md)
- [API Reference](../../docs/reference/eval/)
