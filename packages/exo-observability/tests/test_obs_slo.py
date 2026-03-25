"""Tests for exo.observability.slo — SLO tracking."""

from __future__ import annotations

import time

from exo.observability.slo import (  # pyright: ignore[reportMissingImports]
    SLO,
    SLOReport,
    SLOTracker,
    get_tracker,
    reset,
)

# ---------------------------------------------------------------------------
# SLO dataclass
# ---------------------------------------------------------------------------


class TestSLO:
    def test_creation_defaults(self) -> None:
        slo = SLO(name="latency", metric_name="agent.latency", target=2.0)
        assert slo.name == "latency"
        assert slo.metric_name == "agent.latency"
        assert slo.target == 2.0
        assert slo.window_seconds == 3600
        assert slo.comparator == "gt"

    def test_creation_custom(self) -> None:
        slo = SLO(
            name="p99",
            metric_name="agent.latency",
            target=5.0,
            window_seconds=1800,
            comparator="lt",
        )
        assert slo.window_seconds == 1800
        assert slo.comparator == "lt"

    def test_frozen(self) -> None:
        slo = SLO(name="x", metric_name="m", target=0.9)
        try:
            slo.name = "y"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# SLOReport dataclass
# ---------------------------------------------------------------------------


class TestSLOReport:
    def test_creation(self) -> None:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        r = SLOReport(
            slo_name="latency",
            target=2.0,
            actual=1.5,
            budget_remaining=0.9,
            compliant=True,
            window_start=now,
            total_samples=100,
            violating_samples=10,
        )
        assert r.slo_name == "latency"
        assert r.compliant is True
        assert r.budget_remaining == 0.9
        assert r.total_samples == 100
        assert r.violating_samples == 10

    def test_frozen(self) -> None:
        from datetime import UTC, datetime

        r = SLOReport(
            slo_name="x",
            target=1.0,
            actual=0.5,
            budget_remaining=1.0,
            compliant=True,
            window_start=datetime.now(UTC),
            total_samples=0,
            violating_samples=0,
        )
        try:
            r.slo_name = "y"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# SLO registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_list(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="a", metric_name="m", target=0.99)
        tracker.register(slo)
        assert len(tracker.list_slos()) == 1
        assert tracker.list_slos()[0].name == "a"

    def test_register_overwrite(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="a", metric_name="m", target=0.99))
        tracker.register(SLO(name="a", metric_name="m", target=0.95))
        assert len(tracker.list_slos()) == 1
        assert tracker.list_slos()[0].target == 0.95

    def test_unregister(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="a", metric_name="m", target=0.99))
        tracker.unregister("a")
        assert len(tracker.list_slos()) == 0

    def test_unregister_missing(self) -> None:
        tracker = SLOTracker()
        tracker.unregister("nonexistent")  # no error

    def test_clear(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="a", metric_name="m", target=0.99))
        tracker.record("m", 1.0)
        tracker.clear()
        assert len(tracker.list_slos()) == 0


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_record_creates_samples(self) -> None:
        tracker = SLOTracker()
        tracker.record("latency", 1.5)
        tracker.record("latency", 2.0)
        assert len(tracker._samples["latency"]) == 2

    def test_record_multiple_metrics(self) -> None:
        tracker = SLOTracker()
        tracker.record("latency", 1.0)
        tracker.record("success_rate", 1.0)
        assert "latency" in tracker._samples
        assert "success_rate" in tracker._samples


# ---------------------------------------------------------------------------
# Reporting — GT comparator (success rate style)
# ---------------------------------------------------------------------------


class TestReportGT:
    def test_compliant(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.9, comparator="gt")
        tracker.register(slo)
        for _ in range(10):
            tracker.record("success", 1.0)
        r = tracker.report("uptime")
        assert r is not None
        assert r.compliant is True
        assert r.actual == 1.0
        assert r.violating_samples == 0
        assert r.budget_remaining == 1.0

    def test_violation(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.9, comparator="gt")
        tracker.register(slo)
        # 5 successes + 5 failures => average 0.5, below target
        for _ in range(5):
            tracker.record("success", 1.0)
        for _ in range(5):
            tracker.record("success", 0.0)
        r = tracker.report("uptime")
        assert r is not None
        assert r.compliant is False
        assert r.actual == 0.5
        assert r.violating_samples == 5
        assert r.total_samples == 10

    def test_budget_calculation(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.5, comparator="gt")
        tracker.register(slo)
        # 8 good + 2 bad => 20% violating => budget = 0.8
        for _ in range(8):
            tracker.record("success", 1.0)
        for _ in range(2):
            tracker.record("success", 0.0)
        r = tracker.report("uptime")
        assert r is not None
        assert r.budget_remaining == 0.8

    def test_no_samples(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.99)
        tracker.register(slo)
        r = tracker.report("uptime")
        assert r is not None
        assert r.compliant is True
        assert r.actual == 0.0
        assert r.total_samples == 0
        assert r.budget_remaining == 1.0


# ---------------------------------------------------------------------------
# Reporting — LT comparator (latency style)
# ---------------------------------------------------------------------------


class TestReportLT:
    def test_compliant(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="fast", metric_name="latency", target=2.0, comparator="lt")
        tracker.register(slo)
        for v in [1.0, 1.5, 0.8]:
            tracker.record("latency", v)
        r = tracker.report("fast")
        assert r is not None
        assert r.compliant is True
        assert r.violating_samples == 0

    def test_violation(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="fast", metric_name="latency", target=2.0, comparator="lt")
        tracker.register(slo)
        for v in [1.0, 3.0, 4.0]:
            tracker.record("latency", v)
        r = tracker.report("fast")
        assert r is not None
        assert r.compliant is False
        assert r.violating_samples == 2

    def test_exact_threshold_violates(self) -> None:
        """Values equal to threshold count as violating for lt."""
        tracker = SLOTracker()
        slo = SLO(name="fast", metric_name="latency", target=2.0, comparator="lt")
        tracker.register(slo)
        tracker.record("latency", 2.0)
        r = tracker.report("fast")
        assert r is not None
        assert r.violating_samples == 1


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_expired_samples_excluded(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.9, window_seconds=60)
        tracker.register(slo)

        # Insert old samples with timestamps in the past
        old_ts = time.monotonic() - 120  # 2 minutes ago
        tracker._samples.setdefault("success", []).append((old_ts, 0.0))

        # Insert new sample
        tracker.record("success", 1.0)

        r = tracker.report("uptime")
        assert r is not None
        assert r.total_samples == 1  # only the recent sample
        assert r.actual == 1.0
        assert r.compliant is True

    def test_all_expired(self) -> None:
        tracker = SLOTracker()
        slo = SLO(name="uptime", metric_name="success", target=0.9, window_seconds=60)
        tracker.register(slo)

        old_ts = time.monotonic() - 120
        tracker._samples.setdefault("success", []).append((old_ts, 0.0))

        r = tracker.report("uptime")
        assert r is not None
        assert r.total_samples == 0
        assert r.compliant is True


# ---------------------------------------------------------------------------
# report_all
# ---------------------------------------------------------------------------


class TestReportAll:
    def test_multiple_slos(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="uptime", metric_name="success", target=0.9))
        tracker.register(SLO(name="fast", metric_name="latency", target=2.0, comparator="lt"))
        tracker.record("success", 1.0)
        tracker.record("latency", 1.0)
        reports = tracker.report_all()
        assert len(reports) == 2
        names = {r.slo_name for r in reports}
        assert names == {"uptime", "fast"}

    def test_empty(self) -> None:
        tracker = SLOTracker()
        assert tracker.report_all() == []


# ---------------------------------------------------------------------------
# Report for unknown SLO
# ---------------------------------------------------------------------------


class TestReportUnknown:
    def test_returns_none(self) -> None:
        tracker = SLOTracker()
        assert tracker.report("nonexistent") is None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_samples(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="a", metric_name="m", target=0.99))
        tracker.record("m", 1.0)
        tracker.reset()
        r = tracker.report("a")
        assert r is not None
        assert r.total_samples == 0

    def test_reset_preserves_slos(self) -> None:
        tracker = SLOTracker()
        tracker.register(SLO(name="a", metric_name="m", target=0.99))
        tracker.reset()
        assert len(tracker.list_slos()) == 1


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_get_tracker_singleton(self) -> None:
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_reset_clears_global(self) -> None:
        t = get_tracker()
        t.register(SLO(name="x", metric_name="m", target=0.5))
        t.record("m", 1.0)
        reset()
        assert len(t.list_slos()) == 0
        assert t.report("x") is None

    def test_full_flow(self) -> None:
        reset()
        t = get_tracker()
        slo = SLO(name="api_success", metric_name="api.success_rate", target=0.95)
        t.register(slo)

        # 19 successes + 1 failure
        for _ in range(19):
            t.record("api.success_rate", 1.0)
        t.record("api.success_rate", 0.0)

        r = t.report("api_success")
        assert r is not None
        assert r.total_samples == 20
        assert r.actual == 0.95
        assert r.compliant is True
        assert r.violating_samples == 1
        assert r.budget_remaining == 0.95

        reset()
