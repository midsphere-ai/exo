"""Tests for exo.observability.cost — cost estimation and tracking."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from exo.observability.cost import (  # pyright: ignore[reportMissingImports]
    CostEntry,
    CostTracker,
    ModelPricing,
    get_tracker,
    reset,
)

# ---------------------------------------------------------------------------
# ModelPricing
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_creation(self) -> None:
        p = ModelPricing(model_pattern=r"gpt-4o", input_cost_per_1k=0.0025, output_cost_per_1k=0.01)
        assert p.model_pattern == r"gpt-4o"
        assert p.input_cost_per_1k == 0.0025
        assert p.output_cost_per_1k == 0.01

    def test_frozen(self) -> None:
        p = ModelPricing(model_pattern=r"gpt-4o", input_cost_per_1k=0.0025, output_cost_per_1k=0.01)
        with pytest.raises(AttributeError):
            p.model_pattern = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CostEntry
# ---------------------------------------------------------------------------


class TestCostEntry:
    def test_creation(self) -> None:
        e = CostEntry(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.0025,
            output_cost=0.005,
            total_cost=0.0075,
        )
        assert e.model == "gpt-4o"
        assert e.input_tokens == 1000
        assert e.output_tokens == 500
        assert e.input_cost == 0.0025
        assert e.output_cost == 0.005
        assert e.total_cost == 0.0075

    def test_has_utc_timestamp(self) -> None:
        e = CostEntry(
            model="x", input_tokens=0, output_tokens=0, input_cost=0, output_cost=0, total_cost=0
        )
        assert e.timestamp.tzinfo is not None

    def test_frozen(self) -> None:
        e = CostEntry(
            model="x", input_tokens=0, output_tokens=0, input_cost=0, output_cost=0, total_cost=0
        )
        with pytest.raises(AttributeError):
            e.model = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CostTracker — pricing lookup
# ---------------------------------------------------------------------------


class TestPricingLookup:
    def test_gpt4o(self) -> None:
        t = CostTracker()
        p = t._find_pricing("gpt-4o")
        assert p is not None
        assert p.input_cost_per_1k == 0.0025

    def test_gpt4o_mini(self) -> None:
        t = CostTracker()
        p = t._find_pricing("gpt-4o-mini")
        assert p is not None
        assert p.input_cost_per_1k == 0.00015

    def test_claude_sonnet(self) -> None:
        t = CostTracker()
        p = t._find_pricing("claude-sonnet-4-5-20250514")
        assert p is not None
        assert p.input_cost_per_1k == 0.003

    def test_claude_haiku(self) -> None:
        t = CostTracker()
        p = t._find_pricing("claude-haiku-3-5-20241022")
        assert p is not None
        assert p.input_cost_per_1k == 0.0008

    def test_gemini_flash(self) -> None:
        t = CostTracker()
        p = t._find_pricing("gemini-2.0-flash")
        assert p is not None
        assert p.input_cost_per_1k == 0.0001

    def test_unknown_model_returns_none(self) -> None:
        t = CostTracker()
        assert t._find_pricing("totally-unknown-model") is None

    def test_register_pricing_takes_priority(self) -> None:
        t = CostTracker()
        custom = ModelPricing(
            model_pattern=r"gpt-4o", input_cost_per_1k=0.999, output_cost_per_1k=0.999
        )
        t.register_pricing(custom)
        p = t._find_pricing("gpt-4o")
        assert p is not None
        assert p.input_cost_per_1k == 0.999


# ---------------------------------------------------------------------------
# CostTracker — recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_known_model(self) -> None:
        t = CostTracker()
        entry = t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        assert entry.model == "gpt-4o"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.input_cost == pytest.approx(0.0025)
        assert entry.output_cost == pytest.approx(0.005)
        assert entry.total_cost == pytest.approx(0.0075)

    def test_record_unknown_model_zero_cost(self, caplog: pytest.LogCaptureFixture) -> None:
        t = CostTracker()
        with caplog.at_level("WARNING", logger="exo.cost"):
            entry = t.record("unknown-model", input_tokens=1000, output_tokens=500)
        assert entry.input_cost == 0.0
        assert entry.output_cost == 0.0
        assert entry.total_cost == 0.0
        assert "No pricing found" in caplog.text

    def test_record_stores_entry(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert len(t.get_entries()) == 1

    def test_multiple_records(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        t.record("gpt-4o-mini", input_tokens=2000, output_tokens=1000)
        assert len(t.get_entries()) == 2


# ---------------------------------------------------------------------------
# CostTracker — queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_get_total(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        assert t.get_total() == pytest.approx(0.015)

    def test_get_total_empty(self) -> None:
        t = CostTracker()
        assert t.get_total() == 0.0

    def test_get_breakdown(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        t.record("gpt-4o-mini", input_tokens=2000, output_tokens=1000)
        breakdown = t.get_breakdown()
        assert "gpt-4o" in breakdown
        assert "gpt-4o-mini" in breakdown
        assert breakdown["gpt-4o"] == pytest.approx(0.0075)

    def test_get_breakdown_aggregates_same_model(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        t.record("gpt-4o", input_tokens=1000, output_tokens=500)
        breakdown = t.get_breakdown()
        assert breakdown["gpt-4o"] == pytest.approx(0.015)

    def test_get_entries_all(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        t.record("gpt-4o-mini", input_tokens=200, output_tokens=100)
        entries = t.get_entries()
        assert len(entries) == 2

    def test_get_entries_since(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        entries = t.get_entries(since=cutoff)
        assert len(entries) == 0

    def test_get_entries_since_includes_recent(self) -> None:
        t = CostTracker()
        cutoff = datetime.now(UTC) - timedelta(seconds=1)
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        entries = t.get_entries(since=cutoff)
        assert len(entries) == 1

    def test_get_entries_returns_copy(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        entries = t.get_entries()
        entries.clear()
        assert len(t.get_entries()) == 1


# ---------------------------------------------------------------------------
# CostTracker — reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_entries(self) -> None:
        t = CostTracker()
        t.record("gpt-4o", input_tokens=100, output_tokens=50)
        t.reset()
        assert len(t.get_entries()) == 0
        assert t.get_total() == 0.0

    def test_reset_preserves_pricing(self) -> None:
        t = CostTracker()
        custom = ModelPricing(
            model_pattern=r"custom-model", input_cost_per_1k=1.0, output_cost_per_1k=2.0
        )
        t.register_pricing(custom)
        t.reset()
        assert t._find_pricing("custom-model") is not None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_recording(self) -> None:
        t = CostTracker()
        num_threads = 10
        records_per_thread = 100

        def record_many() -> None:
            for _ in range(records_per_thread):
                t.record("gpt-4o", input_tokens=100, output_tokens=50)

        threads = [threading.Thread(target=record_many) for _ in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(t.get_entries()) == num_threads * records_per_thread


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_get_tracker_singleton(self) -> None:
        assert get_tracker() is get_tracker()

    def test_reset_clears_global(self) -> None:
        tracker = get_tracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        reset()
        assert tracker.get_total() == 0.0

    def test_full_flow(self) -> None:
        reset()
        tracker = get_tracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record("claude-sonnet-4-5-20250514", input_tokens=2000, output_tokens=1000)
        assert tracker.get_total() > 0
        breakdown = tracker.get_breakdown()
        assert len(breakdown) == 2
        reset()
