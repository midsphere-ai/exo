"""Tests for orbiter.context.processor — ContextProcessor pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from orbiter.context.context import Context  # pyright: ignore[reportMissingImports]
from orbiter.context.processor import (  # pyright: ignore[reportMissingImports]
    ContextProcessor,
    MessageOffloader,
    ProcessorError,
    ProcessorPipeline,
    SummarizeProcessor,
    ToolResultOffloader,
)

# ── Helpers ──────────────────────────────────────────────────────────


class RecordingProcessor(ContextProcessor):
    """Test processor that records calls."""

    def __init__(self, event: str, *, name: str | None = None) -> None:
        super().__init__(event, name=name)
        self.calls: list[tuple[Context, dict[str, Any]]] = []

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        self.calls.append((ctx, dict(payload)))


class MutatingProcessor(ContextProcessor):
    """Test processor that sets a state key."""

    def __init__(self, event: str, key: str, value: Any) -> None:
        super().__init__(event)
        self._key = key
        self._value = value

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        ctx.state.set(self._key, self._value)


class FailingProcessor(ContextProcessor):
    """Test processor that raises on process."""

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        msg = "processor failed"
        raise RuntimeError(msg)


def _make_ctx(task_id: str = "test-task", **state_data: Any) -> Context:
    ctx = Context(task_id)
    for k, v in state_data.items():
        ctx.state.set(k, v)
    return ctx


# ── ContextProcessor ABC tests ──────────────────────────────────────


class TestContextProcessorABC:
    def test_create_concrete_processor(self) -> None:
        proc = RecordingProcessor("pre_llm_call")
        assert proc.event == "pre_llm_call"
        assert proc.name == "RecordingProcessor"

    def test_custom_name(self) -> None:
        proc = RecordingProcessor("post_tool_call", name="my-proc")
        assert proc.name == "my-proc"

    def test_empty_event_raises(self) -> None:
        with pytest.raises(ProcessorError, match="non-empty"):
            RecordingProcessor("")

    def test_repr(self) -> None:
        proc = RecordingProcessor("pre_llm_call", name="test")
        result = repr(proc)
        assert "RecordingProcessor" in result
        assert "pre_llm_call" in result
        assert "test" in result

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ContextProcessor("event")  # type: ignore[abstract]


# ── ProcessorPipeline tests ─────────────────────────────────────────


class TestPipelineRegistration:
    def test_register_and_has_processors(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("pre_llm_call")
        pipeline.register(proc)
        assert pipeline.has_processors("pre_llm_call")
        assert not pipeline.has_processors("post_tool_call")

    def test_register_multiple_same_event(self) -> None:
        pipeline = ProcessorPipeline()
        p1 = RecordingProcessor("pre_llm_call", name="first")
        p2 = RecordingProcessor("pre_llm_call", name="second")
        pipeline.register(p1)
        pipeline.register(p2)
        assert len(pipeline) == 2
        procs = pipeline.list_processors("pre_llm_call")
        assert len(procs) == 2
        assert procs[0].name == "first"
        assert procs[1].name == "second"

    def test_register_different_events(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(RecordingProcessor("pre_llm_call"))
        pipeline.register(RecordingProcessor("post_tool_call"))
        assert len(pipeline) == 2
        assert pipeline.has_processors("pre_llm_call")
        assert pipeline.has_processors("post_tool_call")

    def test_unregister(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("pre_llm_call")
        pipeline.register(proc)
        pipeline.unregister(proc)
        assert not pipeline.has_processors("pre_llm_call")
        assert len(pipeline) == 0

    def test_unregister_not_registered(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("pre_llm_call")
        # Should not raise
        pipeline.unregister(proc)

    def test_method_chaining(self) -> None:
        pipeline = ProcessorPipeline()
        result = pipeline.register(RecordingProcessor("a")).register(RecordingProcessor("b"))
        assert result is pipeline
        assert len(pipeline) == 2

    def test_clear(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(RecordingProcessor("a"))
        pipeline.register(RecordingProcessor("b"))
        pipeline.clear()
        assert len(pipeline) == 0

    def test_list_processors_all(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(RecordingProcessor("a"))
        pipeline.register(RecordingProcessor("b"))
        pipeline.register(RecordingProcessor("a"))
        all_procs = pipeline.list_processors()
        assert len(all_procs) == 3

    def test_list_processors_filtered(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(RecordingProcessor("a"))
        pipeline.register(RecordingProcessor("b"))
        assert len(pipeline.list_processors("a")) == 1
        assert len(pipeline.list_processors("c")) == 0

    def test_repr(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(RecordingProcessor("a"))
        r = repr(pipeline)
        assert "ProcessorPipeline" in r
        assert "total=1" in r


class TestPipelineFire:
    async def test_fire_calls_processor(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("pre_llm_call")
        pipeline.register(proc)
        ctx = _make_ctx()
        await pipeline.fire("pre_llm_call", ctx, {"key": "value"})
        assert len(proc.calls) == 1
        assert proc.calls[0][0] is ctx
        assert proc.calls[0][1] == {"key": "value"}

    async def test_fire_no_matching_event(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("pre_llm_call")
        pipeline.register(proc)
        ctx = _make_ctx()
        await pipeline.fire("post_tool_call", ctx)
        assert len(proc.calls) == 0

    async def test_fire_default_empty_payload(self) -> None:
        pipeline = ProcessorPipeline()
        proc = RecordingProcessor("event")
        pipeline.register(proc)
        ctx = _make_ctx()
        await pipeline.fire("event", ctx)
        assert proc.calls[0][1] == {}

    async def test_fire_sequential_order(self) -> None:
        pipeline = ProcessorPipeline()
        order: list[str] = []

        class OrderProc(ContextProcessor):
            def __init__(self, label: str) -> None:
                super().__init__("test")
                self._label = label

            async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
                order.append(self._label)

        pipeline.register(OrderProc("first"))
        pipeline.register(OrderProc("second"))
        pipeline.register(OrderProc("third"))
        await pipeline.fire("test", _make_ctx())
        assert order == ["first", "second", "third"]

    async def test_fire_mutates_state(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(MutatingProcessor("prep", "status", "ready"))
        ctx = _make_ctx()
        await pipeline.fire("prep", ctx)
        assert ctx.state.get("status") == "ready"

    async def test_fire_error_propagates(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(FailingProcessor("boom"))
        with pytest.raises(RuntimeError, match="processor failed"):
            await pipeline.fire("boom", _make_ctx())

    async def test_fire_stops_on_first_error(self) -> None:
        pipeline = ProcessorPipeline()
        proc_before = RecordingProcessor("test")
        proc_after = RecordingProcessor("test")
        pipeline.register(proc_before)
        pipeline.register(FailingProcessor("test"))
        pipeline.register(proc_after)
        with pytest.raises(RuntimeError):
            await pipeline.fire("test", _make_ctx())
        assert len(proc_before.calls) == 1
        assert len(proc_after.calls) == 0


# ── SummarizeProcessor tests ────────────────────────────────────────


class TestSummarizeProcessor:
    def test_defaults(self) -> None:
        proc = SummarizeProcessor()
        assert proc.event == "pre_llm_call"
        assert proc.name == "summarize"

    async def test_no_history(self) -> None:
        proc = SummarizeProcessor()
        ctx = _make_ctx()
        await proc.process(ctx, {})
        assert ctx.state.get("needs_summary") is None

    async def test_history_below_threshold(self) -> None:
        proc = SummarizeProcessor()
        config = ContextConfig(summary_threshold=10)
        ctx = Context("test", config=config)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        ctx.state.set("history", history)
        await proc.process(ctx, {})
        assert ctx.state.get("needs_summary") is None

    async def test_history_at_threshold(self) -> None:
        proc = SummarizeProcessor()
        config = ContextConfig(summary_threshold=5)
        ctx = Context("test", config=config)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        ctx.state.set("history", history)
        await proc.process(ctx, {})
        assert ctx.state.get("needs_summary") is None

    async def test_history_exceeds_threshold(self) -> None:
        proc = SummarizeProcessor()
        config = ContextConfig(summary_threshold=3)
        ctx = Context("test", config=config)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(7)]
        ctx.state.set("history", history)
        await proc.process(ctx, {})
        assert ctx.state.get("needs_summary") is True
        candidates = ctx.state.get("summary_candidates")
        assert len(candidates) == 4  # 7 - 3 = 4 excess
        assert candidates[0]["content"] == "msg 0"
        assert candidates[3]["content"] == "msg 3"

    async def test_with_pipeline(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        config = ContextConfig(summary_threshold=2)
        ctx = Context("test", config=config)
        ctx.state.set(
            "history",
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        )
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("needs_summary") is True


# ── ToolResultOffloader tests ────────────────────────────────────────


class TestToolResultOffloader:
    def test_defaults(self) -> None:
        proc = ToolResultOffloader()
        assert proc.event == "post_tool_call"
        assert proc.name == "tool_result_offloader"
        assert proc.max_size == 5000

    def test_custom_max_size(self) -> None:
        proc = ToolResultOffloader(max_size=100)
        assert proc.max_size == 100

    async def test_no_tool_result(self) -> None:
        proc = ToolResultOffloader(max_size=10)
        ctx = _make_ctx()
        await proc.process(ctx, {})
        assert ctx.state.get("offloaded_results") is None

    async def test_small_result_not_offloaded(self) -> None:
        proc = ToolResultOffloader(max_size=100)
        ctx = _make_ctx()
        payload: dict[str, Any] = {"tool_result": "short result", "tool_name": "test"}
        await proc.process(ctx, payload)
        assert ctx.state.get("offloaded_results") is None
        assert payload["tool_result"] == "short result"

    async def test_large_result_offloaded(self) -> None:
        proc = ToolResultOffloader(max_size=20)
        ctx = _make_ctx()
        large = "x" * 50
        payload: dict[str, Any] = {
            "tool_result": large,
            "tool_name": "my_tool",
            "tool_call_id": "call_123",
        }
        await proc.process(ctx, payload)

        offloaded = ctx.state.get("offloaded_results")
        assert offloaded is not None
        assert len(offloaded) == 1
        assert offloaded[0]["tool_name"] == "my_tool"
        assert offloaded[0]["tool_call_id"] == "call_123"
        assert offloaded[0]["content"] == large
        assert offloaded[0]["size"] == 50

        # Payload should have been replaced with a reference
        assert "offloaded" in payload["tool_result"].lower()
        assert "50 chars" in payload["tool_result"]
        # The first half of the content is preserved as preview
        assert payload["tool_result"].startswith("x" * 10)

    async def test_multiple_offloads_accumulate(self) -> None:
        proc = ToolResultOffloader(max_size=10)
        ctx = _make_ctx()
        for i in range(3):
            payload: dict[str, Any] = {
                "tool_result": "a" * 20,
                "tool_name": f"tool_{i}",
                "tool_call_id": f"call_{i}",
            }
            await proc.process(ctx, payload)

        offloaded = ctx.state.get("offloaded_results")
        assert len(offloaded) == 3
        assert offloaded[0]["tool_name"] == "tool_0"
        assert offloaded[2]["tool_name"] == "tool_2"

    async def test_offload_default_tool_name(self) -> None:
        proc = ToolResultOffloader(max_size=5)
        ctx = _make_ctx()
        payload: dict[str, Any] = {"tool_result": "x" * 10}
        await proc.process(ctx, payload)
        offloaded = ctx.state.get("offloaded_results")
        assert offloaded[0]["tool_name"] == "unknown"
        assert offloaded[0]["tool_call_id"] == "unknown"

    async def test_with_pipeline(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(ToolResultOffloader(max_size=10))
        ctx = _make_ctx()
        payload: dict[str, Any] = {
            "tool_result": "y" * 20,
            "tool_name": "search",
            "tool_call_id": "call_abc",
        }
        await pipeline.fire("post_tool_call", ctx, payload)
        assert ctx.state.get("offloaded_results") is not None


# ── Integration tests ────────────────────────────────────────────────


class TestProcessorIntegration:
    async def test_multiple_events_same_pipeline(self) -> None:
        pipeline = ProcessorPipeline()
        pre_proc = RecordingProcessor("pre_llm_call")
        post_proc = RecordingProcessor("post_tool_call")
        pipeline.register(pre_proc)
        pipeline.register(post_proc)

        ctx = _make_ctx()
        await pipeline.fire("pre_llm_call", ctx, {"step": 1})
        await pipeline.fire("post_tool_call", ctx, {"step": 2})

        assert len(pre_proc.calls) == 1
        assert pre_proc.calls[0][1]["step"] == 1
        assert len(post_proc.calls) == 1
        assert post_proc.calls[0][1]["step"] == 2

    async def test_processor_chain_mutates_state(self) -> None:
        """Two processors for the same event both mutate state."""
        pipeline = ProcessorPipeline()
        pipeline.register(MutatingProcessor("setup", "a", 1))
        pipeline.register(MutatingProcessor("setup", "b", 2))
        ctx = _make_ctx()
        await pipeline.fire("setup", ctx)
        assert ctx.state.get("a") == 1
        assert ctx.state.get("b") == 2

    async def test_full_lifecycle(self) -> None:
        """Simulate a full lifecycle with both built-in processors."""
        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        pipeline.register(ToolResultOffloader(max_size=15))

        config = ContextConfig(summary_threshold=2)
        ctx = Context("lifecycle-test", config=config)
        ctx.state.set(
            "history",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "search for X"},
            ],
        )

        # Pre-LLM: should trigger summarization
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("needs_summary") is True

        # Post-tool: should trigger offloading
        payload: dict[str, Any] = {
            "tool_result": "a" * 30,
            "tool_name": "search",
            "tool_call_id": "c1",
        }
        await pipeline.fire("post_tool_call", ctx, payload)
        assert ctx.state.get("offloaded_results") is not None


# ── MessageOffloader tests ─────────────────────────────────────────


class TestMessageOffloader:
    def test_defaults(self) -> None:
        proc = MessageOffloader()
        assert proc.event == "pre_llm_call"
        assert proc.name == "message_offloader"
        assert proc.max_message_size == 10000

    def test_custom_max_size(self) -> None:
        proc = MessageOffloader(max_message_size=500)
        assert proc.max_message_size == 500

    async def test_no_history(self) -> None:
        proc = MessageOffloader(max_message_size=10)
        ctx = _make_ctx()
        await proc.process(ctx, {})
        assert ctx.state.get("offloaded_messages") is None

    async def test_empty_history(self) -> None:
        proc = MessageOffloader(max_message_size=10)
        ctx = _make_ctx(history=[])
        await proc.process(ctx, {})
        # Empty list is falsy, so offloaded_messages should not be set
        assert ctx.state.get("offloaded_messages") is None

    async def test_message_within_limit_untouched(self) -> None:
        proc = MessageOffloader(max_message_size=100)
        ctx = _make_ctx(
            history=[
                {"role": "user", "content": "short message"},
                {"role": "assistant", "content": "short reply"},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        assert history[0]["content"] == "short message"
        assert history[1]["content"] == "short reply"
        # offloaded_messages dict should be empty
        offloaded = ctx.state.get("offloaded_messages")
        assert offloaded == {}

    async def test_oversized_user_message_replaced(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        large_content = "x" * 50
        ctx = _make_ctx(
            history=[
                {"role": "user", "content": large_content},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        # Content should be replaced with an offload marker
        assert history[0]["content"].startswith("[[OFFLOAD: handle=")
        assert history[0]["content"].endswith("]]")
        # Original should be stored
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 1
        handle_id = list(offloaded.keys())[0]
        assert offloaded[handle_id] == large_content
        assert f"handle={handle_id}" in history[0]["content"]

    async def test_oversized_assistant_message_replaced(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        large_content = "y" * 50
        ctx = _make_ctx(
            history=[
                {"role": "assistant", "content": large_content},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        assert "[[OFFLOAD:" in history[0]["content"]
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 1
        assert list(offloaded.values())[0] == large_content

    async def test_oversized_tool_message_replaced(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        large_content = "z" * 50
        ctx = _make_ctx(
            history=[
                {"role": "tool", "content": large_content},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        assert "[[OFFLOAD:" in history[0]["content"]
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 1

    async def test_system_message_not_offloaded(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        large_system = "s" * 50
        ctx = _make_ctx(
            history=[
                {"role": "system", "content": large_system},
                {"role": "user", "content": "short"},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        # System message should be untouched
        assert history[0]["content"] == large_system
        # User message within limit also untouched
        assert history[1]["content"] == "short"
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 0

    async def test_mixed_messages_only_oversized_replaced(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        ctx = _make_ctx(
            history=[
                {"role": "system", "content": "a" * 50},
                {"role": "user", "content": "small"},
                {"role": "assistant", "content": "b" * 50},
                {"role": "user", "content": "c" * 50},
            ]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        # System: untouched (skipped)
        assert history[0]["content"] == "a" * 50
        # Short user: untouched
        assert history[1]["content"] == "small"
        # Large assistant: offloaded
        assert "[[OFFLOAD:" in history[2]["content"]
        # Large user: offloaded
        assert "[[OFFLOAD:" in history[3]["content"]

        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 2

    async def test_multiple_offloads_have_unique_handles(self) -> None:
        proc = MessageOffloader(max_message_size=10)
        ctx = _make_ctx(
            history=[
                {"role": "user", "content": "x" * 20},
                {"role": "assistant", "content": "y" * 20},
                {"role": "user", "content": "z" * 20},
            ]
        )
        await proc.process(ctx, {})
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 3
        # All handles should be unique
        handles = list(offloaded.keys())
        assert len(set(handles)) == 3

    async def test_message_at_exact_limit_not_offloaded(self) -> None:
        proc = MessageOffloader(max_message_size=20)
        exact_content = "a" * 20
        ctx = _make_ctx(
            history=[{"role": "user", "content": exact_content}]
        )
        await proc.process(ctx, {})
        history = ctx.state.get("history")
        assert history[0]["content"] == exact_content

    async def test_original_stored_correctly(self) -> None:
        proc = MessageOffloader(max_message_size=10)
        original = "hello world this is a long message"
        ctx = _make_ctx(
            history=[{"role": "user", "content": original}]
        )
        await proc.process(ctx, {})
        offloaded = ctx.state.get("offloaded_messages")
        assert list(offloaded.values())[0] == original

    async def test_with_pipeline(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(MessageOffloader(max_message_size=15))
        ctx = _make_ctx(
            history=[
                {"role": "user", "content": "a" * 30},
                {"role": "assistant", "content": "ok"},
            ]
        )
        await pipeline.fire("pre_llm_call", ctx)
        history = ctx.state.get("history")
        assert "[[OFFLOAD:" in history[0]["content"]
        assert history[1]["content"] == "ok"
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 1

    async def test_none_content_skipped(self) -> None:
        proc = MessageOffloader(max_message_size=10)
        ctx = _make_ctx(
            history=[{"role": "user", "content": None}]
        )
        await proc.process(ctx, {})
        offloaded = ctx.state.get("offloaded_messages")
        assert len(offloaded) == 0
