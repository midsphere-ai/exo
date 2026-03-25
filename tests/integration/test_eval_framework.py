"""Integration tests for the evaluation framework.

Tests that:
- TrajectoryDataset captures an agent run with input, output, and tool_calls fields.
- LLMAsJudgeScorer with an LLM-as-judge callable returns a valid score in [0.0, 1.0].
- A rule-based ExactMatchScorer returns 1.0 for matching output and 0.0 otherwise.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_trajectory_collected_after_run(vertex_model: str) -> None:
    """TrajectoryDataset captures an agent run with input, output, and tool_calls fields."""
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.train.trajectory import TrajectoryDataset  # pyright: ignore[reportMissingImports]

    provider = get_provider(vertex_model)
    agent = Agent(
        name="eval-trajectory-agent",
        model=vertex_model,
        instructions="You are a helpful assistant.",
    )
    prompt = "What is the capital of France? Respond with just the city name."
    result = await agent.run(prompt, provider=provider)

    dataset = TrajectoryDataset()
    dataset.from_messages(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result.text},
        ],
        task_id="eval-test-trajectory",
        agent_id=agent.name,
    )

    assert len(dataset.items) >= 1
    step = dataset.items[0]
    assert step.input  # non-empty input field
    assert step.output  # non-empty output field
    assert hasattr(step, "tool_calls")  # tool_calls field exists (may be empty tuple)


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_llm_judge_scorer_returns_valid_score(vertex_model: str) -> None:
    """LLMAsJudgeScorer with rubric returns a float score in [0.0, 1.0]."""
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.eval.llm_scorer import LLMAsJudgeScorer  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    rubric = (
        "Did the agent answer the question correctly? "
        'Return JSON: {"score": <float 0.0-1.0>, "explanation": "<text>"}.'
    )
    provider = get_provider(vertex_model)

    # Run agent to produce output to evaluate
    agent = Agent(
        name="eval-judge-agent",
        model=vertex_model,
        instructions="You are a helpful assistant.",
    )
    result = await agent.run(
        "What is the capital of France? Respond with just the city name.",
        provider=provider,
    )
    agent_output = result.text

    # Create judge callable using the LLM
    judge_agent = Agent(
        name="eval-judge",
        model=vertex_model,
        instructions="You are an evaluator. Always respond with valid JSON only.",
    )

    async def judge_fn(prompt: str) -> str:
        judge_result = await judge_agent.run(prompt, provider=provider)
        return judge_result.text

    scorer = LLMAsJudgeScorer(judge=judge_fn, system_prompt=rubric, name="llm_judge")
    sr = await scorer.score("case-001", "What is the capital of France?", agent_output)

    assert isinstance(sr.score, float)
    assert 0.0 <= sr.score <= 1.0


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_rule_based_scorer_exact_match() -> None:
    """ExactMatchScorer returns 1.0 when output contains expected string, 0.0 otherwise."""
    from exo.eval.base import Scorer, ScorerResult  # pyright: ignore[reportMissingImports]

    class ExactMatchScorer(Scorer):
        """Scores 1.0 if the expected string appears in the output, 0.0 otherwise."""

        def __init__(self, expected: str, *, name: str = "exact_match") -> None:
            self._expected = expected
            self._name = name

        async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
            text = str(output) if output is not None else ""
            match = self._expected.lower() in text.lower()
            return ScorerResult(scorer_name=self._name, score=1.0 if match else 0.0)

    scorer = ExactMatchScorer(expected="Paris")

    # Output containing "Paris" → score == 1.0
    sr_match = await scorer.score("case-paris", None, "The capital of France is Paris.")
    assert sr_match.score == 1.0

    # Output containing "London" (no "Paris") → score == 0.0
    sr_no_match = await scorer.score("case-london", None, "The capital of England is London.")
    assert sr_no_match.score == 0.0
