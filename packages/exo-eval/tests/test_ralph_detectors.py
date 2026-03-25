"""Tests for Ralph loop stop detectors."""

from __future__ import annotations

import time

from exo.eval.ralph.config import (  # pyright: ignore[reportMissingImports]
    LoopState,
    StopConditionConfig,
    StopType,
)
from exo.eval.ralph.detectors import (  # pyright: ignore[reportMissingImports]
    CompositeDetector,
    ConsecutiveFailureDetector,
    CostLimitDetector,
    MaxIterationDetector,
    ScoreThresholdDetector,
    StopDecision,
    StopDetector,
    TimeoutDetector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides: object) -> LoopState:
    """Create a LoopState with sensible defaults, accepting overrides."""
    defaults: dict[str, object] = {
        "iteration": 0,
        "cumulative_cost": 0.0,
        "consecutive_failures": 0,
    }
    defaults.update(overrides)
    return LoopState(**defaults)  # type: ignore[arg-type]


def _config(**overrides: object) -> StopConditionConfig:
    """Create a StopConditionConfig with sensible defaults."""
    return StopConditionConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StopDecision
# ---------------------------------------------------------------------------


class TestStopDecision:
    def test_continue_is_falsy(self) -> None:
        d = StopDecision(should_stop=False)
        assert not d
        assert d.stop_type == StopType.NONE
        assert d.reason == ""

    def test_stop_is_truthy(self) -> None:
        d = StopDecision(should_stop=True, stop_type=StopType.TIMEOUT, reason="too slow")
        assert d
        assert d.stop_type == StopType.TIMEOUT

    def test_metadata_default(self) -> None:
        d = StopDecision(should_stop=False)
        assert d.metadata == {}

    def test_frozen(self) -> None:
        d = StopDecision(should_stop=False)
        try:
            d.should_stop = True  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


# ---------------------------------------------------------------------------
# StopDetector ABC
# ---------------------------------------------------------------------------


class TestStopDetectorABC:
    def test_cannot_instantiate(self) -> None:
        try:
            StopDetector()  # type: ignore[abstract]
            raised = False
        except TypeError:
            raised = True
        assert raised

    async def test_concrete_subclass(self) -> None:
        class AlwaysStop(StopDetector):
            async def check(self, state: LoopState, config: StopConditionConfig) -> StopDecision:
                return StopDecision(should_stop=True, stop_type=StopType.COMPLETION)

        d = AlwaysStop()
        result = await d.check(_state(), _config())
        assert result.should_stop


# ---------------------------------------------------------------------------
# MaxIterationDetector
# ---------------------------------------------------------------------------


class TestMaxIterationDetector:
    async def test_below_limit(self) -> None:
        d = MaxIterationDetector()
        result = await d.check(_state(iteration=3), _config(max_iterations=10))
        assert not result.should_stop

    async def test_at_limit(self) -> None:
        d = MaxIterationDetector()
        result = await d.check(_state(iteration=10), _config(max_iterations=10))
        assert result.should_stop
        assert result.stop_type == StopType.MAX_ITERATIONS

    async def test_above_limit(self) -> None:
        d = MaxIterationDetector()
        result = await d.check(_state(iteration=15), _config(max_iterations=10))
        assert result.should_stop

    async def test_metadata(self) -> None:
        d = MaxIterationDetector()
        result = await d.check(_state(iteration=5), _config(max_iterations=5))
        assert result.metadata["current"] == 5
        assert result.metadata["max"] == 5


# ---------------------------------------------------------------------------
# TimeoutDetector
# ---------------------------------------------------------------------------


class TestTimeoutDetector:
    async def test_disabled_when_zero(self) -> None:
        d = TimeoutDetector()
        result = await d.check(_state(), _config(timeout=0.0))
        assert not result.should_stop

    async def test_within_timeout(self) -> None:
        d = TimeoutDetector()
        # start_time is recent, so elapsed is small
        result = await d.check(_state(), _config(timeout=100.0))
        assert not result.should_stop

    async def test_timeout_exceeded(self) -> None:
        d = TimeoutDetector()
        # Manually set start_time in the past
        state = _state()
        state.start_time = time.monotonic() - 200.0
        result = await d.check(state, _config(timeout=100.0))
        assert result.should_stop
        assert result.stop_type == StopType.TIMEOUT

    async def test_metadata_has_timing(self) -> None:
        d = TimeoutDetector()
        state = _state()
        state.start_time = time.monotonic() - 50.0
        result = await d.check(state, _config(timeout=10.0))
        assert result.should_stop
        assert "elapsed" in result.metadata
        assert "timeout" in result.metadata


# ---------------------------------------------------------------------------
# CostLimitDetector
# ---------------------------------------------------------------------------


class TestCostLimitDetector:
    async def test_disabled_when_zero(self) -> None:
        d = CostLimitDetector()
        result = await d.check(_state(cumulative_cost=999.0), _config(max_cost=0.0))
        assert not result.should_stop

    async def test_below_limit(self) -> None:
        d = CostLimitDetector()
        result = await d.check(_state(cumulative_cost=5.0), _config(max_cost=10.0))
        assert not result.should_stop

    async def test_at_limit(self) -> None:
        d = CostLimitDetector()
        result = await d.check(_state(cumulative_cost=10.0), _config(max_cost=10.0))
        assert result.should_stop
        assert result.stop_type == StopType.MAX_COST

    async def test_above_limit(self) -> None:
        d = CostLimitDetector()
        result = await d.check(_state(cumulative_cost=15.0), _config(max_cost=10.0))
        assert result.should_stop

    async def test_metadata(self) -> None:
        d = CostLimitDetector()
        result = await d.check(_state(cumulative_cost=5.5), _config(max_cost=5.0))
        assert result.metadata["current"] == 5.5
        assert result.metadata["max"] == 5.0


# ---------------------------------------------------------------------------
# ConsecutiveFailureDetector
# ---------------------------------------------------------------------------


class TestConsecutiveFailureDetector:
    async def test_disabled_when_zero(self) -> None:
        d = ConsecutiveFailureDetector()
        result = await d.check(
            _state(consecutive_failures=100),
            _config(max_consecutive_failures=0),
        )
        assert not result.should_stop

    async def test_below_threshold(self) -> None:
        d = ConsecutiveFailureDetector()
        result = await d.check(
            _state(consecutive_failures=1),
            _config(max_consecutive_failures=3),
        )
        assert not result.should_stop

    async def test_at_threshold(self) -> None:
        d = ConsecutiveFailureDetector()
        result = await d.check(
            _state(consecutive_failures=3),
            _config(max_consecutive_failures=3),
        )
        assert result.should_stop
        assert result.stop_type == StopType.MAX_CONSECUTIVE_FAILURES

    async def test_above_threshold(self) -> None:
        d = ConsecutiveFailureDetector()
        result = await d.check(
            _state(consecutive_failures=5),
            _config(max_consecutive_failures=3),
        )
        assert result.should_stop

    async def test_metadata(self) -> None:
        d = ConsecutiveFailureDetector()
        result = await d.check(
            _state(consecutive_failures=3),
            _config(max_consecutive_failures=3),
        )
        assert result.metadata["current"] == 3
        assert result.metadata["max"] == 3


# ---------------------------------------------------------------------------
# ScoreThresholdDetector
# ---------------------------------------------------------------------------


class TestScoreThresholdDetector:
    async def test_disabled_when_zero(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 1.0})
        result = await d.check(state, _config(score_threshold=0.0))
        assert not result.should_stop

    async def test_no_scores(self) -> None:
        d = ScoreThresholdDetector()
        result = await d.check(_state(), _config(score_threshold=0.8))
        assert not result.should_stop

    async def test_below_threshold(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.5})
        result = await d.check(state, _config(score_threshold=0.8))
        assert not result.should_stop

    async def test_at_threshold(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.8})
        result = await d.check(state, _config(score_threshold=0.8))
        assert result.should_stop
        assert result.stop_type == StopType.SCORE_THRESHOLD

    async def test_above_threshold(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.95})
        result = await d.check(state, _config(score_threshold=0.8))
        assert result.should_stop

    async def test_mean_of_multiple_scorers(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.9, "relevance": 0.7})
        # mean = 0.8, threshold = 0.8 → should stop
        result = await d.check(state, _config(score_threshold=0.8))
        assert result.should_stop

    async def test_mean_below_threshold_multi_scorer(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.9, "relevance": 0.5})
        # mean = 0.7, threshold = 0.8 → should not stop
        result = await d.check(state, _config(score_threshold=0.8))
        assert not result.should_stop

    async def test_uses_latest_score_only(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.9})  # old
        state.record_score({"accuracy": 0.3})  # latest
        result = await d.check(state, _config(score_threshold=0.5))
        assert not result.should_stop

    async def test_metadata(self) -> None:
        d = ScoreThresholdDetector()
        state = _state()
        state.record_score({"accuracy": 0.9})
        result = await d.check(state, _config(score_threshold=0.8))
        assert "mean_score" in result.metadata
        assert result.metadata["threshold"] == 0.8


# ---------------------------------------------------------------------------
# CompositeDetector
# ---------------------------------------------------------------------------


class TestCompositeDetector:
    async def test_empty_continues(self) -> None:
        cd = CompositeDetector()
        result = await cd.check(_state(), _config())
        assert not result.should_stop

    async def test_first_match_wins(self) -> None:
        cd = CompositeDetector(
            [
                MaxIterationDetector(),
                TimeoutDetector(),
            ]
        )
        state = _state(iteration=10)
        state.start_time = time.monotonic() - 999.0
        result = await cd.check(state, _config(max_iterations=10, timeout=100.0))
        # MaxIterationDetector is first, should trigger
        assert result.should_stop
        assert result.stop_type == StopType.MAX_ITERATIONS

    async def test_second_detector_triggers(self) -> None:
        cd = CompositeDetector(
            [
                MaxIterationDetector(),
                CostLimitDetector(),
            ]
        )
        # iteration below limit, but cost exceeded
        result = await cd.check(
            _state(iteration=1, cumulative_cost=50.0),
            _config(max_iterations=10, max_cost=10.0),
        )
        assert result.should_stop
        assert result.stop_type == StopType.MAX_COST

    async def test_none_triggers(self) -> None:
        cd = CompositeDetector(
            [
                MaxIterationDetector(),
                CostLimitDetector(),
            ]
        )
        result = await cd.check(
            _state(iteration=1, cumulative_cost=1.0),
            _config(max_iterations=10, max_cost=10.0),
        )
        assert not result.should_stop

    async def test_add_chaining(self) -> None:
        cd = CompositeDetector()
        returned = cd.add(MaxIterationDetector()).add(TimeoutDetector())
        assert returned is cd
        assert len(cd) == 2

    def test_len(self) -> None:
        cd = CompositeDetector([MaxIterationDetector(), TimeoutDetector()])
        assert len(cd) == 2

    def test_repr(self) -> None:
        cd = CompositeDetector([MaxIterationDetector()])
        assert "CompositeDetector" in repr(cd)
        assert "1" in repr(cd)

    async def test_all_five_detectors(self) -> None:
        """Integration: composite with all five built-in detectors."""
        cd = CompositeDetector(
            [
                MaxIterationDetector(),
                TimeoutDetector(),
                CostLimitDetector(),
                ConsecutiveFailureDetector(),
                ScoreThresholdDetector(),
            ]
        )
        # No conditions met → continue
        result = await cd.check(_state(iteration=1), _config(max_iterations=10))
        assert not result.should_stop

        # Score threshold met → stops
        state = _state(iteration=1)
        state.record_score({"accuracy": 0.95})
        result = await cd.check(state, _config(max_iterations=10, score_threshold=0.9))
        assert result.should_stop
        assert result.stop_type == StopType.SCORE_THRESHOLD
