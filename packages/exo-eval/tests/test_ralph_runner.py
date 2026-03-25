"""Tests for RalphRunner — iterative refinement loop."""

from __future__ import annotations

from typing import Any

from exo.eval.base import (  # pyright: ignore[reportMissingImports]
    Scorer,
    ScorerResult,
)
from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    RalphConfig,
    StopConditionConfig,
    StopType,
    ValidationConfig,
)
from exo.eval.ralph.runner import (  # pyright: ignore[reportMissingImports]
    RalphResult,
    RalphRunner,
)
from exo.eval.reflection import (  # pyright: ignore[reportMissingImports]
    Reflector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedScorer(Scorer):
    """Scorer that returns a fixed score."""

    def __init__(self, name: str = "fixed", score: float = 0.8) -> None:
        self._name = name
        self._score = score

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        return ScorerResult(scorer_name=self._name, score=self._score)


class _CountingScorer(Scorer):
    """Scorer that returns increasing scores on each call."""

    def __init__(self, name: str = "counting", scores: list[float] | None = None) -> None:
        self._name = name
        self._scores = list(scores or [0.3, 0.6, 0.9])
        self._call = 0

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        idx = min(self._call, len(self._scores) - 1)
        self._call += 1
        return ScorerResult(scorer_name=self._name, score=self._scores[idx])


class _FailingScorer(Scorer):
    """Scorer that raises an exception."""

    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        msg = "scorer failed"
        raise RuntimeError(msg)


class _SimpleReflector(Reflector):
    """Reflector that returns a fixed reflection result."""

    def __init__(self, suggestions: list[str] | None = None) -> None:
        super().__init__(name="simple_reflector")
        self._suggestions = suggestions or ["try harder"]

    async def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "needs improvement",
            "suggestions": self._suggestions,
        }

    async def suggest(self, insights: dict[str, Any]) -> dict[str, Any]:
        return {"suggestions": self._suggestions}


def _make_execute(responses: list[str]) -> Any:
    """Create an async execute_fn that returns responses in order."""
    call = {"idx": 0}

    async def execute(input: str) -> str:
        idx = min(call["idx"], len(responses) - 1)
        call["idx"] += 1
        return responses[idx]

    return execute


def _make_failing_execute(fail_count: int = 1, then: str = "ok") -> Any:
    """Create an execute_fn that fails *fail_count* times then succeeds."""
    call = {"idx": 0}

    async def execute(input: str) -> str:
        call["idx"] += 1
        if call["idx"] <= fail_count:
            msg = f"fail-{call['idx']}"
            raise RuntimeError(msg)
        return then

    return execute


# ---------------------------------------------------------------------------
# RalphResult
# ---------------------------------------------------------------------------


class TestRalphResult:
    def test_creation(self) -> None:
        r = RalphResult(
            output="done",
            stop_type=StopType.MAX_ITERATIONS,
            reason="hit max",
            iterations=5,
            scores={"acc": 0.9},
            state={"iteration": 5},
        )
        assert r.output == "done"
        assert r.stop_type == StopType.MAX_ITERATIONS
        assert r.iterations == 5
        assert r.scores["acc"] == 0.9
        assert r.reflections == []

    def test_frozen(self) -> None:
        r = RalphResult(
            output="x", stop_type=StopType.NONE, reason="", iterations=0, scores={}, state={}
        )
        assert r.output == "x"

    def test_with_reflections(self) -> None:
        r = RalphResult(
            output="done",
            stop_type=StopType.COMPLETION,
            reason="done",
            iterations=2,
            scores={},
            state={},
            reflections=[{"iteration": 1, "summary": "needs work"}],
        )
        assert len(r.reflections) == 1


# ---------------------------------------------------------------------------
# RalphRunner init
# ---------------------------------------------------------------------------


class TestRalphRunnerInit:
    def test_defaults(self) -> None:
        runner = RalphRunner(_make_execute(["hi"]), [])
        assert repr(runner).startswith("RalphRunner(")

    def test_custom_config(self) -> None:
        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=3))
        runner = RalphRunner(_make_execute(["hi"]), [_FixedScorer()], config=cfg)
        assert repr(runner).startswith("RalphRunner(scorers=1")


# ---------------------------------------------------------------------------
# Full loop — single iteration (max_iterations=1)
# ---------------------------------------------------------------------------


class TestRalphRunnerSingleIteration:
    async def test_single_iter_success(self) -> None:
        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
        runner = RalphRunner(
            _make_execute(["hello"]),
            [_FixedScorer(score=0.9)],
            config=cfg,
        )
        result = await runner.run("input")
        assert result.output == "hello"
        assert result.iterations == 1
        assert result.stop_type == StopType.MAX_ITERATIONS
        assert result.scores["fixed"] == 0.9

    async def test_single_iter_no_scorers(self) -> None:
        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
        runner = RalphRunner(_make_execute(["hi"]), [], config=cfg)
        result = await runner.run("input")
        assert result.output == "hi"
        assert result.scores == {}

    async def test_single_iter_execution_failure(self) -> None:
        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=1))
        runner = RalphRunner(_make_failing_execute(fail_count=10), [], config=cfg)
        result = await runner.run("input")
        assert "fail" in result.output
        assert result.iterations == 1


# ---------------------------------------------------------------------------
# Full loop — multi-iteration with score improvement
# ---------------------------------------------------------------------------


class TestRalphRunnerScoreImprovement:
    async def test_improves_over_iterations(self) -> None:
        """Scores increase until threshold triggers stop."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(
                max_iterations=10,
                score_threshold=0.85,
            ),
        )
        runner = RalphRunner(
            _make_execute(["v1", "v2", "v3"]),
            [_CountingScorer(scores=[0.3, 0.6, 0.9])],
            config=cfg,
        )
        result = await runner.run("task")
        assert result.stop_type == StopType.SCORE_THRESHOLD
        assert result.iterations == 3
        assert result.scores["counting"] == 0.9

    async def test_max_iterations_stops(self) -> None:
        """Loop hits max iterations when scores stay low."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(max_iterations=3),
        )
        runner = RalphRunner(
            _make_execute(["low"] * 5),
            [_FixedScorer(score=0.1)],
            config=cfg,
        )
        result = await runner.run("task")
        assert result.stop_type == StopType.MAX_ITERATIONS
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# Early stopping — consecutive failures
# ---------------------------------------------------------------------------


class TestRalphRunnerEarlyStopping:
    async def test_consecutive_failures_stop(self) -> None:
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(
                max_iterations=10,
                max_consecutive_failures=2,
            ),
        )
        runner = RalphRunner(
            _make_failing_execute(fail_count=10),
            [],
            config=cfg,
        )
        result = await runner.run("task")
        assert result.stop_type == StopType.MAX_CONSECUTIVE_FAILURES
        assert result.iterations == 2

    async def test_failure_then_recovery(self) -> None:
        """Failure counter resets after success."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(
                max_iterations=3,
                max_consecutive_failures=5,
            ),
        )
        runner = RalphRunner(
            _make_failing_execute(fail_count=1, then="recovered"),
            [_FixedScorer(score=0.5)],
            config=cfg,
        )
        result = await runner.run("task")
        assert result.stop_type == StopType.MAX_ITERATIONS
        assert result.iterations == 3
        # Second+ iterations succeed
        assert result.output == "recovered"


# ---------------------------------------------------------------------------
# Reflection (Learn phase)
# ---------------------------------------------------------------------------


class TestRalphRunnerReflection:
    async def test_reflection_on_low_scores(self) -> None:
        """Reflector fires when score is below min_score_threshold."""
        cfg = RalphConfig(
            validation=ValidationConfig(min_score_threshold=0.8),
            stop_condition=StopConditionConfig(max_iterations=2),
        )
        runner = RalphRunner(
            _make_execute(["v1", "v2"]),
            [_FixedScorer(score=0.3)],
            config=cfg,
            reflector=_SimpleReflector(suggestions=["be better"]),
        )
        result = await runner.run("input")
        assert len(result.reflections) > 0
        assert result.reflections[0]["summary"] == "needs improvement"

    async def test_no_reflection_on_high_scores(self) -> None:
        """Reflector does NOT fire when score meets threshold."""
        cfg = RalphConfig(
            validation=ValidationConfig(min_score_threshold=0.5),
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_execute(["good"]),
            [_FixedScorer(score=0.9)],
            config=cfg,
            reflector=_SimpleReflector(),
        )
        result = await runner.run("input")
        assert result.reflections == []

    async def test_reflection_on_execution_failure(self) -> None:
        """Reflector fires on execution failure."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_failing_execute(fail_count=10),
            [],
            config=cfg,
            reflector=_SimpleReflector(suggestions=["fix error"]),
        )
        result = await runner.run("input")
        assert len(result.reflections) == 1

    async def test_no_reflector_configured(self) -> None:
        """No reflection when reflector is None."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_execute(["x"]),
            [_FixedScorer(score=0.1)],
            config=cfg,
        )
        result = await runner.run("input")
        assert result.reflections == []


# ---------------------------------------------------------------------------
# Plan phase (re-prompt)
# ---------------------------------------------------------------------------


class TestRalphRunnerPlan:
    async def test_suggestions_appended_to_input(self) -> None:
        """Re-prompt appends reflection suggestions to the original input."""
        calls: list[str] = []

        async def capture_execute(input: str) -> str:
            calls.append(input)
            return "output"

        cfg = RalphConfig(
            validation=ValidationConfig(min_score_threshold=0.9),
            stop_condition=StopConditionConfig(max_iterations=2),
        )
        runner = RalphRunner(
            capture_execute,
            [_FixedScorer(score=0.3)],
            config=cfg,
            reflector=_SimpleReflector(suggestions=["improve X"]),
        )
        await runner.run("original task")
        # First call: original input; second call: original + suggestions
        assert len(calls) == 2
        assert calls[0] == "original task"
        assert "[Previous feedback]" in calls[1]
        assert "improve X" in calls[1]
        assert "original task" in calls[1]


# ---------------------------------------------------------------------------
# Scorer edge cases
# ---------------------------------------------------------------------------


class TestRalphRunnerScorerEdgeCases:
    async def test_failing_scorer_ignored(self) -> None:
        """A scorer that raises is silently skipped."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_execute(["ok"]),
            [_FailingScorer(), _FixedScorer(score=0.7)],
            config=cfg,
        )
        result = await runner.run("input")
        # Only the successful scorer's result appears
        assert "fixed" in result.scores
        assert len(result.scores) == 1

    async def test_validation_disabled(self) -> None:
        """Scores empty when validation.enabled=False."""
        cfg = RalphConfig(
            validation=ValidationConfig(enabled=False),
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_execute(["ok"]),
            [_FixedScorer(score=0.9)],
            config=cfg,
        )
        result = await runner.run("input")
        assert result.scores == {}

    async def test_reflection_disabled(self) -> None:
        """No reflection when reflection.enabled=False even with reflector."""
        from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
            ReflectionConfig,
        )

        cfg = RalphConfig(
            reflection=ReflectionConfig(enabled=False),
            stop_condition=StopConditionConfig(max_iterations=1),
        )
        runner = RalphRunner(
            _make_execute(["x"]),
            [_FixedScorer(score=0.1)],
            config=cfg,
            reflector=_SimpleReflector(),
        )
        result = await runner.run("input")
        assert result.reflections == []


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


class TestRalphRunnerState:
    async def test_state_in_result(self) -> None:
        """Result includes serialised loop state."""
        cfg = RalphConfig(stop_condition=StopConditionConfig(max_iterations=2))
        runner = RalphRunner(
            _make_execute(["a", "b"]),
            [_FixedScorer(score=0.5)],
            config=cfg,
        )
        result = await runner.run("input")
        assert result.state["iteration"] == 2
        assert result.state["successful_steps"] == 2
        assert len(result.state["score_history"]) == 2

    async def test_mixed_success_failure_state(self) -> None:
        """State tracks both successes and failures."""
        cfg = RalphConfig(
            stop_condition=StopConditionConfig(max_iterations=3, max_consecutive_failures=5),
        )
        runner = RalphRunner(
            _make_failing_execute(fail_count=1, then="ok"),
            [_FixedScorer(score=0.5)],
            config=cfg,
        )
        result = await runner.run("input")
        assert result.state["failed_steps"] == 1
        assert result.state["successful_steps"] == 2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRalphRunnerRepr:
    def test_repr(self) -> None:
        runner = RalphRunner(_make_execute(["hi"]), [_FixedScorer()])
        r = repr(runner)
        assert "RalphRunner" in r
        assert "scorers=1" in r
