"""Tests for exo.observability.metrics — both OTel and in-memory paths."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    METRIC_AGENT_RUN_COUNTER,
    METRIC_AGENT_RUN_DURATION,
    METRIC_AGENT_TOKEN_USAGE,
    METRIC_TOOL_STEP_COUNTER,
    METRIC_TOOL_STEP_DURATION,
    MetricsCollector,
    Timer,
    build_agent_attributes,
    build_tool_attributes,
    create_counter,
    create_gauge,
    create_histogram,
    get_collector,
    get_metrics_snapshot,
    record_agent_run,
    record_tool_step,
    reset_metrics,
    timer,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    AGENT_NAME,
    AGENT_RUN_SUCCESS,
    AGENT_STEP,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    SESSION_ID,
    TASK_ID,
    TOOL_NAME,
    TOOL_STEP_SUCCESS,
    USER_ID,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_collector() -> None:
    """Reset the in-memory collector before each test."""
    reset_metrics()


# ---------------------------------------------------------------------------
# Metric name constant tests
# ---------------------------------------------------------------------------


class TestMetricNames:
    """Verify metric name constants."""

    def test_agent_run_duration_name(self) -> None:
        assert METRIC_AGENT_RUN_DURATION == "agent_run_duration"

    def test_agent_run_counter_name(self) -> None:
        assert METRIC_AGENT_RUN_COUNTER == "agent_run_counter"

    def test_agent_token_usage_name(self) -> None:
        assert METRIC_AGENT_TOKEN_USAGE == "agent_token_usage"

    def test_tool_step_duration_name(self) -> None:
        assert METRIC_TOOL_STEP_DURATION == "tool_step_duration"

    def test_tool_step_counter_name(self) -> None:
        assert METRIC_TOOL_STEP_COUNTER == "tool_step_counter"


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    """Test the in-memory MetricsCollector."""

    def test_add_counter(self) -> None:
        mc = MetricsCollector()
        mc.add_counter("my_counter", 1.0)
        mc.add_counter("my_counter", 2.0)
        snap = mc.get_snapshot()
        assert snap["counters"]["my_counter"] == 3.0

    def test_record_histogram(self) -> None:
        mc = MetricsCollector()
        mc.record_histogram("my_hist", 1.5, {"a": "b"})
        mc.record_histogram("my_hist", 2.5)
        snap = mc.get_snapshot()
        assert len(snap["histograms"]["my_hist"]) == 2
        assert snap["histograms"]["my_hist"][0]["value"] == 1.5
        assert snap["histograms"]["my_hist"][0]["attributes"] == {"a": "b"}
        assert snap["histograms"]["my_hist"][1]["value"] == 2.5

    def test_set_gauge(self) -> None:
        mc = MetricsCollector()
        mc.set_gauge("my_gauge", 42.0)
        snap = mc.get_snapshot()
        assert snap["gauges"]["my_gauge"] == 42.0
        mc.set_gauge("my_gauge", 99.0)
        snap = mc.get_snapshot()
        assert snap["gauges"]["my_gauge"] == 99.0

    def test_reset(self) -> None:
        mc = MetricsCollector()
        mc.add_counter("c", 1.0)
        mc.record_histogram("h", 1.0)
        mc.set_gauge("g", 1.0)
        mc.reset()
        snap = mc.get_snapshot()
        assert snap["counters"] == {}
        assert snap["histograms"] == {}
        assert snap["gauges"] == {}

    def test_snapshot_is_independent(self) -> None:
        mc = MetricsCollector()
        mc.add_counter("x", 1.0)
        snap1 = mc.get_snapshot()
        mc.add_counter("x", 1.0)
        snap2 = mc.get_snapshot()
        assert snap1["counters"]["x"] == 1.0
        assert snap2["counters"]["x"] == 2.0

    def test_empty_snapshot(self) -> None:
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert snap == {"counters": {}, "histograms": {}, "gauges": {}}

    def test_get_collector_returns_singleton(self) -> None:
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2

    def test_get_metrics_snapshot(self) -> None:
        get_collector().add_counter("test_snap", 5.0)
        snap = get_metrics_snapshot()
        assert snap["counters"]["test_snap"] == 5.0


# ---------------------------------------------------------------------------
# Attribute builder tests
# ---------------------------------------------------------------------------


class TestBuildAgentAttributes:
    """Test build_agent_attributes helper."""

    def test_defaults(self) -> None:
        attrs = build_agent_attributes(agent_name="test-agent")
        assert attrs[AGENT_NAME] == "test-agent"
        assert attrs[TASK_ID] == ""
        assert attrs[SESSION_ID] == ""
        assert attrs[USER_ID] == ""
        assert AGENT_STEP not in attrs

    def test_all_fields(self) -> None:
        attrs = build_agent_attributes(
            agent_name="my-agent",
            task_id="task-1",
            session_id="sess-1",
            user_id="user-1",
            step=3,
        )
        assert attrs[AGENT_NAME] == "my-agent"
        assert attrs[TASK_ID] == "task-1"
        assert attrs[SESSION_ID] == "sess-1"
        assert attrs[USER_ID] == "user-1"
        assert attrs[AGENT_STEP] == 3

    def test_none_values_become_empty_string(self) -> None:
        attrs = build_agent_attributes(agent_name="a", task_id=None)  # type: ignore[arg-type]
        assert attrs[TASK_ID] == ""


class TestBuildToolAttributes:
    """Test build_tool_attributes helper."""

    def test_defaults(self) -> None:
        attrs = build_tool_attributes(tool_name="search")
        assert attrs[TOOL_NAME] == "search"
        assert attrs[AGENT_NAME] == ""
        assert attrs[TASK_ID] == ""

    def test_all_fields(self) -> None:
        attrs = build_tool_attributes(
            tool_name="search",
            agent_name="my-agent",
            task_id="task-1",
        )
        assert attrs[TOOL_NAME] == "search"
        assert attrs[AGENT_NAME] == "my-agent"
        assert attrs[TASK_ID] == "task-1"


# ---------------------------------------------------------------------------
# Recording tests — in-memory path (no OTel)
# ---------------------------------------------------------------------------


class TestRecordAgentRunInMemory:
    """Test record_agent_run with the in-memory fallback."""

    def test_success(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            attrs = build_agent_attributes(agent_name="agent-a")
            record_agent_run(
                duration=1.5,
                success=True,
                attributes=attrs,
                input_tokens=100,
                output_tokens=50,
            )
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_AGENT_RUN_COUNTER] == 1.0
        dur_recs = snap["histograms"][METRIC_AGENT_RUN_DURATION]
        assert len(dur_recs) > 0
        assert dur_recs[0]["value"] == 1.5
        assert dur_recs[0]["attributes"][AGENT_RUN_SUCCESS] == "1"

    def test_failure_sets_success_zero(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_agent_run(duration=0.3, success=False)
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_AGENT_RUN_DURATION]
        assert dur_recs[0]["attributes"][AGENT_RUN_SUCCESS] == "0"

    def test_no_tokens_skips_token_metric(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_agent_run(duration=0.1, success=True)
        snap = get_metrics_snapshot()
        assert METRIC_AGENT_TOKEN_USAGE not in snap["histograms"]

    def test_token_attributes(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_agent_run(
                duration=1.0,
                success=True,
                input_tokens=200,
                output_tokens=100,
            )
        snap = get_metrics_snapshot()
        tok_recs = snap["histograms"][METRIC_AGENT_TOKEN_USAGE]
        assert len(tok_recs) > 0
        tok_attrs = tok_recs[0]["attributes"]
        assert tok_attrs[GEN_AI_USAGE_INPUT_TOKENS] == 200
        assert tok_attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 100
        assert tok_attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 300

    def test_empty_attributes(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_agent_run(duration=0.5, success=True)
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_AGENT_RUN_DURATION]
        assert dur_recs[0]["attributes"][AGENT_RUN_SUCCESS] == "1"


class TestRecordToolStepInMemory:
    """Test record_tool_step with the in-memory fallback."""

    def test_success(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            attrs = build_tool_attributes(tool_name="search", agent_name="agent-a")
            record_tool_step(duration=0.2, success=True, attributes=attrs)
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_TOOL_STEP_DURATION]
        assert len(dur_recs) > 0
        assert snap["counters"][METRIC_TOOL_STEP_COUNTER] == 1.0

    def test_failure(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            attrs = build_tool_attributes(tool_name="fetch")
            record_tool_step(duration=0.5, success=False, attributes=attrs)
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_TOOL_STEP_DURATION]
        assert dur_recs[0]["attributes"][TOOL_STEP_SUCCESS] == "0"

    def test_empty_attributes(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_tool_step(duration=0.1, success=True)
        snap = get_metrics_snapshot()
        cnt = snap["counters"][METRIC_TOOL_STEP_COUNTER]
        assert cnt == 1.0

    def test_does_not_mutate_input(self) -> None:
        attrs = build_tool_attributes(tool_name="search")
        original = dict(attrs)
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_tool_step(duration=0.1, success=True, attributes=attrs)
        assert attrs == original


# ---------------------------------------------------------------------------
# Recording tests — OTel path
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_meter_provider() -> None:
    """Reset the global MeterProvider before each test so set_meter_provider works."""
    from opentelemetry import metrics

    metrics._internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    metrics._internal._METER_PROVIDER = None  # type: ignore[attr-defined]


@pytest.fixture()
def metric_reader(_reset_meter_provider: None) -> Any:
    """Create an in-memory metric reader and set up a fresh MeterProvider."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return reader


def _get_metric_data(reader: Any, name: str) -> list[dict[str, Any]]:
    """Extract data points for a named metric from the reader."""
    data = reader.get_metrics_data()
    results: list[dict[str, Any]] = []
    if data is None:
        return results
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        results.append(
                            {
                                "value": getattr(dp, "value", None) or getattr(dp, "sum", None),
                                "attributes": dict(dp.attributes) if dp.attributes else {},
                            }
                        )
    return results


class TestRecordAgentRunOTel:
    """Test record_agent_run with real OTel."""

    def test_success(self, metric_reader: Any) -> None:
        attrs = build_agent_attributes(agent_name="agent-a")
        record_agent_run(
            duration=1.5,
            success=True,
            attributes=attrs,
            input_tokens=100,
            output_tokens=50,
        )
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0

        cnt_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_COUNTER)
        assert len(cnt_data) > 0

        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) > 0

    def test_failure_sets_success_zero(self, metric_reader: Any) -> None:
        attrs = build_agent_attributes(agent_name="agent-b")
        record_agent_run(duration=0.3, success=False, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][AGENT_RUN_SUCCESS] == "0"

    def test_no_tokens_skips_token_metric(self, metric_reader: Any) -> None:
        record_agent_run(duration=0.1, success=True)
        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) == 0

    def test_token_attributes(self, metric_reader: Any) -> None:
        record_agent_run(
            duration=1.0,
            success=True,
            input_tokens=200,
            output_tokens=100,
        )
        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) > 0
        attrs = tok_data[0]["attributes"]
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 200
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 100
        assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 300


class TestRecordToolStepOTel:
    """Test record_tool_step with real OTel."""

    def test_success(self, metric_reader: Any) -> None:
        attrs = build_tool_attributes(tool_name="search", agent_name="agent-a")
        record_tool_step(duration=0.2, success=True, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(dur_data) > 0
        cnt_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_COUNTER)
        assert len(cnt_data) > 0

    def test_failure(self, metric_reader: Any) -> None:
        attrs = build_tool_attributes(tool_name="fetch")
        record_tool_step(duration=0.5, success=False, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert dur_data[0]["attributes"][TOOL_STEP_SUCCESS] == "0"


# ---------------------------------------------------------------------------
# Instrument factory tests (OTel)
# ---------------------------------------------------------------------------


class TestInstrumentFactory:
    """Verify instrument factory functions return valid instruments."""

    def test_create_counter(self, _reset_meter_provider: None) -> None:
        c = create_counter("test_counter", "1", "A test counter")
        assert c is not None

    def test_create_histogram(self, _reset_meter_provider: None) -> None:
        h = create_histogram("test_histogram", "s", "A test histogram")
        assert h is not None

    def test_create_gauge(self, _reset_meter_provider: None) -> None:
        g = create_gauge("test_gauge", "1", "A test gauge")
        assert g is not None


# ---------------------------------------------------------------------------
# Timer tests
# ---------------------------------------------------------------------------


class TestTimer:
    """Test the Timer helper."""

    def test_basic_timing(self) -> None:
        t = Timer()
        t.start()
        time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed >= 0.005

    def test_start_returns_self(self) -> None:
        t = Timer()
        assert t.start() is t

    def test_elapsed_property(self) -> None:
        t = Timer()
        assert t.elapsed == 0.0
        t.start()
        t.stop()
        assert t.elapsed > 0.0

    def test_elapsed_matches_stop_return(self) -> None:
        t = Timer()
        t.start()
        result = t.stop()
        assert t.elapsed == result

    def test_initial_state(self) -> None:
        t = Timer()
        assert t.elapsed == 0.0

    def test_context_manager(self) -> None:
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.005

    def test_timer_context_manager_function(self) -> None:
        with timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.005


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationInMemory:
    """Integration tests using in-memory collector."""

    def test_agent_full_flow(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            attrs = build_agent_attributes(
                agent_name="planner",
                task_id="t-1",
                session_id="s-1",
                step=2,
            )
            with Timer() as t:
                pass  # simulated work
            record_agent_run(
                duration=t.elapsed,
                success=True,
                attributes=attrs,
                input_tokens=1000,
                output_tokens=500,
            )
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_AGENT_RUN_DURATION]
        assert len(dur_recs) > 0
        assert dur_recs[0]["attributes"][AGENT_NAME] == "planner"

    def test_tool_full_flow(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            attrs = build_tool_attributes(
                tool_name="web_search",
                agent_name="researcher",
                task_id="t-1",
            )
            with Timer() as t:
                pass  # simulated work
            record_tool_step(duration=t.elapsed, success=True, attributes=attrs)
        snap = get_metrics_snapshot()
        dur_recs = snap["histograms"][METRIC_TOOL_STEP_DURATION]
        assert len(dur_recs) > 0
        assert dur_recs[0]["attributes"][TOOL_NAME] == "web_search"

    def test_multiple_recordings(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            for i in range(3):
                record_agent_run(
                    duration=float(i),
                    success=True,
                    attributes=build_agent_attributes(agent_name=f"agent-{i}"),
                )
        snap = get_metrics_snapshot()
        assert snap["counters"][METRIC_AGENT_RUN_COUNTER] == 3.0

    def test_snapshot_reset(self) -> None:
        with patch("exo.observability.metrics.HAS_OTEL", False):
            record_agent_run(duration=1.0, success=True)
        snap = get_metrics_snapshot()
        assert METRIC_AGENT_RUN_DURATION in snap["histograms"]
        reset_metrics()
        snap2 = get_metrics_snapshot()
        assert snap2["histograms"] == {}


class TestIntegrationOTel:
    """Integration tests with real OTel."""

    def test_agent_full_flow(self, metric_reader: Any) -> None:
        attrs = build_agent_attributes(
            agent_name="planner",
            task_id="t-1",
            session_id="s-1",
            step=2,
        )
        with Timer() as t:
            pass
        record_agent_run(
            duration=t.elapsed,
            success=True,
            attributes=attrs,
            input_tokens=1000,
            output_tokens=500,
        )
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][AGENT_NAME] == "planner"

    def test_tool_full_flow(self, metric_reader: Any) -> None:
        attrs = build_tool_attributes(
            tool_name="web_search",
            agent_name="researcher",
            task_id="t-1",
        )
        with Timer() as t:
            pass
        record_tool_step(duration=t.elapsed, success=True, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][TOOL_NAME] == "web_search"


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test MetricsCollector thread safety."""

    def test_concurrent_counter_increments(self) -> None:
        import threading

        mc = MetricsCollector()
        n = 100

        def inc() -> None:
            for _ in range(n):
                mc.add_counter("x", 1.0)

        threads = [threading.Thread(target=inc) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = mc.get_snapshot()
        assert snap["counters"]["x"] == 400.0

    def test_concurrent_histogram_records(self) -> None:
        import threading

        mc = MetricsCollector()
        n = 50

        def rec() -> None:
            for i in range(n):
                mc.record_histogram("h", float(i))

        threads = [threading.Thread(target=rec) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = mc.get_snapshot()
        assert len(snap["histograms"]["h"]) == 200
