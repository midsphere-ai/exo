"""Tests for exo.context.token_tracker — TokenStep, TokenUsageSummary, TokenTracker."""

from exo.context.token_tracker import (  # pyright: ignore[reportMissingImports]
    TokenStep,
    TokenTracker,
    TokenUsageSummary,
)

# ── TokenStep ─────────────────────────────────────────────────────────


class TestTokenStep:
    def test_creation(self) -> None:
        step = TokenStep(agent_id="a", step=0, prompt_tokens=100, output_tokens=50)
        assert step.agent_id == "a"
        assert step.step == 0
        assert step.prompt_tokens == 100
        assert step.output_tokens == 50

    def test_total_tokens(self) -> None:
        step = TokenStep(agent_id="a", step=0, prompt_tokens=100, output_tokens=50)
        assert step.total_tokens == 150

    def test_frozen(self) -> None:
        step = TokenStep(agent_id="a", step=0, prompt_tokens=100, output_tokens=50)
        import pytest

        with pytest.raises(AttributeError):
            step.prompt_tokens = 200  # type: ignore[misc]

    def test_zero_tokens(self) -> None:
        step = TokenStep(agent_id="a", step=0, prompt_tokens=0, output_tokens=0)
        assert step.total_tokens == 0


# ── TokenUsageSummary ─────────────────────────────────────────────────


class TestTokenUsageSummary:
    def test_creation(self) -> None:
        summary = TokenUsageSummary(
            prompt_tokens=200, output_tokens=100, total_tokens=300, step_count=2
        )
        assert summary.prompt_tokens == 200
        assert summary.output_tokens == 100
        assert summary.total_tokens == 300
        assert summary.step_count == 2

    def test_frozen(self) -> None:
        import pytest

        summary = TokenUsageSummary(
            prompt_tokens=200, output_tokens=100, total_tokens=300, step_count=2
        )
        with pytest.raises(AttributeError):
            summary.prompt_tokens = 999  # type: ignore[misc]


# ── TokenTracker — basic ──────────────────────────────────────────────


class TestTokenTrackerBasic:
    def test_empty_tracker(self) -> None:
        tracker = TokenTracker()
        assert len(tracker) == 0
        assert tracker.steps == []
        assert tracker.agent_ids == []

    def test_add_step_returns_token_step(self) -> None:
        tracker = TokenTracker()
        step = tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        assert isinstance(step, TokenStep)
        assert step.agent_id == "agent-a"
        assert step.step == 0
        assert step.prompt_tokens == 100
        assert step.output_tokens == 50

    def test_add_multiple_steps_same_agent(self) -> None:
        tracker = TokenTracker()
        s0 = tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        s1 = tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        assert s0.step == 0
        assert s1.step == 1
        assert len(tracker) == 2

    def test_add_steps_different_agents(self) -> None:
        tracker = TokenTracker()
        sa = tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        sb = tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)
        # Each agent gets independent step indexing
        assert sa.step == 0
        assert sb.step == 0
        assert len(tracker) == 2

    def test_steps_property_returns_copy(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("a", prompt_tokens=10, output_tokens=5)
        steps = tracker.steps
        steps.clear()
        # Original not affected
        assert len(tracker) == 1

    def test_repr(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("a", prompt_tokens=10, output_tokens=5)
        tracker.add_step("b", prompt_tokens=20, output_tokens=10)
        assert repr(tracker) == "TokenTracker(agents=2, steps=2)"

    def test_repr_empty(self) -> None:
        tracker = TokenTracker()
        assert repr(tracker) == "TokenTracker(agents=0, steps=0)"


# ── TokenTracker — trajectory ────────────────────────────────────────


class TestTokenTrackerTrajectory:
    def test_get_trajectory_single_agent(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        trajectory = tracker.get_trajectory("agent-a")
        assert len(trajectory) == 2
        assert trajectory[0].step == 0
        assert trajectory[1].step == 1

    def test_get_trajectory_unknown_agent(self) -> None:
        tracker = TokenTracker()
        assert tracker.get_trajectory("unknown") == []

    def test_get_trajectory_filters_by_agent(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        trajectory_a = tracker.get_trajectory("agent-a")
        trajectory_b = tracker.get_trajectory("agent-b")
        assert len(trajectory_a) == 2
        assert len(trajectory_b) == 1
        assert trajectory_a[0].prompt_tokens == 100
        assert trajectory_a[1].prompt_tokens == 120
        assert trajectory_b[0].prompt_tokens == 200


# ── TokenTracker — total_usage ───────────────────────────────────────


class TestTokenTrackerTotalUsage:
    def test_total_usage_empty(self) -> None:
        tracker = TokenTracker()
        usage = tracker.total_usage()
        assert usage.prompt_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.step_count == 0

    def test_total_usage_single_agent(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        usage = tracker.total_usage()
        assert usage.prompt_tokens == 220
        assert usage.output_tokens == 110
        assert usage.total_tokens == 330
        assert usage.step_count == 2

    def test_total_usage_multi_agent(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        usage = tracker.total_usage()
        assert usage.prompt_tokens == 420
        assert usage.output_tokens == 190
        assert usage.total_tokens == 610
        assert usage.step_count == 3


# ── TokenTracker — agent_usage ───────────────────────────────────────


class TestTokenTrackerAgentUsage:
    def test_agent_usage(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-a", prompt_tokens=100, output_tokens=50)
        tracker.add_step("agent-b", prompt_tokens=200, output_tokens=80)
        tracker.add_step("agent-a", prompt_tokens=120, output_tokens=60)
        usage_a = tracker.agent_usage("agent-a")
        assert usage_a.prompt_tokens == 220
        assert usage_a.output_tokens == 110
        assert usage_a.total_tokens == 330
        assert usage_a.step_count == 2

    def test_agent_usage_unknown(self) -> None:
        tracker = TokenTracker()
        usage = tracker.agent_usage("unknown")
        assert usage.prompt_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.step_count == 0


# ── TokenTracker — agent_ids ─────────────────────────────────────────


class TestTokenTrackerAgentIds:
    def test_agent_ids_order(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("agent-b", prompt_tokens=10, output_tokens=5)
        tracker.add_step("agent-a", prompt_tokens=10, output_tokens=5)
        tracker.add_step("agent-b", prompt_tokens=10, output_tokens=5)
        # First-seen order, no duplicates
        assert tracker.agent_ids == ["agent-b", "agent-a"]

    def test_agent_ids_returns_copy(self) -> None:
        tracker = TokenTracker()
        tracker.add_step("a", prompt_tokens=10, output_tokens=5)
        ids = tracker.agent_ids
        ids.clear()
        assert tracker.agent_ids == ["a"]

    def test_many_agents(self) -> None:
        tracker = TokenTracker()
        for i in range(10):
            tracker.add_step(f"agent-{i}", prompt_tokens=10, output_tokens=5)
        assert len(tracker.agent_ids) == 10
        assert len(tracker) == 10
        usage = tracker.total_usage()
        assert usage.total_tokens == 150
        assert usage.step_count == 10


# ── TokenTracker — add_usage ─────────────────────────────────────────


class TestTokenTrackerAddUsage:
    def test_add_usage_from_duck_typed_object(self) -> None:
        """add_usage() accepts any object with input_tokens and output_tokens."""
        from types import SimpleNamespace

        tracker = TokenTracker()
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150)
        step = tracker.add_usage("agent-a", usage)
        assert isinstance(step, TokenStep)
        assert step.prompt_tokens == 100
        assert step.output_tokens == 50
        assert step.total_tokens == 150

    def test_add_usage_from_exo_usage(self) -> None:
        """add_usage() works with exo.types.Usage."""
        from exo.types import Usage  # pyright: ignore[reportMissingImports]

        tracker = TokenTracker()
        usage = Usage(input_tokens=200, output_tokens=80, total_tokens=280)
        step = tracker.add_usage("agent-b", usage)
        assert step.prompt_tokens == 200
        assert step.output_tokens == 80

    def test_add_usage_increments_step_count(self) -> None:
        from types import SimpleNamespace

        tracker = TokenTracker()
        u1 = SimpleNamespace(input_tokens=100, output_tokens=40, total_tokens=140)
        u2 = SimpleNamespace(input_tokens=120, output_tokens=60, total_tokens=180)
        tracker.add_usage("agent-a", u1)
        tracker.add_usage("agent-a", u2)
        assert len(tracker) == 2
        assert tracker.agent_usage("agent-a").total_tokens == 320

    def test_add_usage_missing_fields_defaults_to_zero(self) -> None:
        """Gracefully handles objects without expected fields."""
        from types import SimpleNamespace

        tracker = TokenTracker()
        usage = SimpleNamespace()  # no input_tokens / output_tokens
        step = tracker.add_usage("agent-a", usage)
        assert step.prompt_tokens == 0
        assert step.output_tokens == 0
