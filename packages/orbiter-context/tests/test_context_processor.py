"""Tests for orbiter.context.processor — ContextProcessor pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from orbiter.context.context import Context  # pyright: ignore[reportMissingImports]
from orbiter.context.processor import (  # pyright: ignore[reportMissingImports]
    ContextProcessor,
    DialogueCompressor,
    MessageOffloader,
    ProcessorError,
    ProcessorPipeline,
    RoundWindowProcessor,
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


# ── DialogueCompressor tests ──────────────────────────────────────


def _tool_call_msg(name: str = "search", call_id: str = "c1") -> dict[str, Any]:
    """Helper: assistant message with a tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": "{}"}],
    }


def _tool_result_msg(
    name: str = "search", call_id: str = "c1", content: str = "result"
) -> dict[str, Any]:
    """Helper: tool result message."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": name,
        "content": content,
    }


class TestDialogueCompressor:
    def test_defaults(self) -> None:
        proc = DialogueCompressor()
        assert proc.event == "pre_llm_call"
        assert proc.name == "dialogue_compressor"
        assert proc.min_tool_chain_length == 3
        assert proc.model is None

    def test_custom_params(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=5, model="gpt-4")
        assert proc.min_tool_chain_length == 5
        assert proc.model == "gpt-4"

    async def test_no_history(self) -> None:
        proc = DialogueCompressor()
        ctx = _make_ctx()
        await proc.process(ctx, {})
        # No error, no state change
        assert ctx.state.get("history") is None

    async def test_empty_history(self) -> None:
        proc = DialogueCompressor()
        ctx = _make_ctx(history=[])
        await proc.process(ctx, {})
        assert ctx.state.get("history") == []

    async def test_no_tool_chains(self) -> None:
        proc = DialogueCompressor()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        assert len(ctx.state.get("history")) == 3

    async def test_chain_below_threshold_untouched(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            {"role": "user", "content": "search for X"},
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "found X"),
            {"role": "assistant", "content": "I found X"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        # 2 tool messages < 3, so no compression
        assert len(ctx.state.get("history")) == 4

    async def test_chain_at_threshold_compressed(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            {"role": "user", "content": "do research"},
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "result 1"),
            _tool_result_msg("lookup", "c2", "result 2"),
            {"role": "assistant", "content": "done"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # 3 tool messages compressed to 1 system message
        assert len(result) == 3  # user + system_summary + assistant
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "system"
        assert result[2]["role"] == "assistant"
        assert "compressed" in result[1]["content"].lower()

    async def test_long_chain_compressed(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            {"role": "user", "content": "find everything"},
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "result A"),
            _tool_call_msg("lookup", "c2"),
            _tool_result_msg("lookup", "c2", "result B"),
            _tool_call_msg("fetch", "c3"),
            _tool_result_msg("fetch", "c3", "result C"),
            {"role": "assistant", "content": "here's what I found"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # 6 tool messages -> 1 summary
        assert len(result) == 3
        summary = result[1]["content"]
        assert "search" in summary
        assert "lookup" in summary
        assert "fetch" in summary

    async def test_summary_contains_tool_names_and_results(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            _tool_call_msg("web_search", "c1"),
            _tool_result_msg("web_search", "c1", "Found 10 results"),
            _tool_call_msg("read_file", "c2"),
            _tool_result_msg("read_file", "c2", "File contents here"),
            _tool_call_msg("analyze", "c3"),
            _tool_result_msg("analyze", "c3", "Analysis complete"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 1
        summary = result[0]["content"]
        assert "web_search" in summary
        assert "read_file" in summary
        assert "analyze" in summary
        assert "3 calls" in summary
        assert "Found 10 results" in summary

    async def test_multiple_separate_chains(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            # First chain (3 tool msgs)
            _tool_call_msg("a", "c1"),
            _tool_result_msg("a", "c1", "r1"),
            _tool_result_msg("b", "c2", "r2"),
            # Break
            {"role": "user", "content": "now do more"},
            # Second chain (3 tool msgs)
            _tool_call_msg("x", "c3"),
            _tool_result_msg("x", "c3", "r3"),
            _tool_result_msg("y", "c4", "r4"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # Both chains compressed: 2 summaries + 1 user msg
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "system"

    async def test_mixed_chains_and_regular_messages(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "r1"),
            _tool_call_msg("lookup", "c2"),
            _tool_result_msg("lookup", "c2", "r2"),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # Chain of 4 tool msgs -> 1 summary; rest untouched
        assert len(result) == 5  # user, assistant, summary, assistant, user
        assert result[2]["role"] == "system"

    async def test_chain_only_tool_results(self) -> None:
        """Consecutive tool results (no assistant tool_call) still count."""
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            _tool_result_msg("a", "c1", "r1"),
            _tool_result_msg("b", "c2", "r2"),
            _tool_result_msg("c", "c3", "r3"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 1
        assert result[0]["role"] == "system"

    async def test_assistant_without_tool_calls_not_counted(self) -> None:
        """Regular assistant messages don't count as tool chain."""
        proc = DialogueCompressor(min_tool_chain_length=3)
        history = [
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "r1"),
            {"role": "assistant", "content": "thinking..."},  # breaks chain
            _tool_call_msg("lookup", "c2"),
            _tool_result_msg("lookup", "c2", "r2"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # Two chains of 2 each, both below threshold of 3
        assert len(result) == 5

    async def test_truncates_long_tool_results_in_summary(self) -> None:
        proc = DialogueCompressor(min_tool_chain_length=3)
        long_result = "x" * 200
        history = [
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", long_result),
            _tool_result_msg("fetch", "c2", "short"),
            _tool_result_msg("analyze", "c3", "done"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        summary = result[0]["content"]
        # Long result should be truncated to 100 chars
        assert "x" * 100 in summary
        assert "x" * 200 not in summary

    async def test_with_pipeline(self) -> None:
        pipeline = ProcessorPipeline()
        pipeline.register(DialogueCompressor(min_tool_chain_length=3))
        history = [
            {"role": "user", "content": "go"},
            _tool_call_msg("a", "c1"),
            _tool_result_msg("a", "c1", "r1"),
            _tool_call_msg("b", "c2"),
            _tool_result_msg("b", "c2", "r2"),
            {"role": "assistant", "content": "done"},
        ]
        ctx = _make_ctx(history=history)
        await pipeline.fire("pre_llm_call", ctx)
        result = ctx.state.get("history")
        # 4 tool messages -> 1 summary
        assert len(result) == 3
        assert result[1]["role"] == "system"

    async def test_fallback_no_model(self) -> None:
        """When model is None, uses simple concatenation fallback."""
        proc = DialogueCompressor(min_tool_chain_length=3, model=None)
        history = [
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "found it"),
            _tool_result_msg("validate", "c2", "valid"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 1
        summary = result[0]["content"]
        # Falls back to concatenation — still produces valid summary
        assert "search" in summary
        assert "found it" in summary

    async def test_with_mock_summarizer_model(self) -> None:
        """When model is set, the summary still works (no actual LLM call)."""
        proc = DialogueCompressor(min_tool_chain_length=3, model="test-model")
        history = [
            _tool_call_msg("search", "c1"),
            _tool_result_msg("search", "c1", "data"),
            _tool_result_msg("process", "c2", "processed"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # Still compresses (model param is stored but not used for actual LLM call yet)
        assert len(result) == 1
        assert result[0]["role"] == "system"


# ── RoundWindowProcessor tests ─────────────────────────────────────


class TestRoundWindowProcessor:
    def test_defaults(self) -> None:
        proc = RoundWindowProcessor()
        assert proc.event == "pre_llm_call"
        assert proc.name == "round_window"
        assert proc.max_rounds == 20

    def test_custom_max_rounds(self) -> None:
        proc = RoundWindowProcessor(max_rounds=5)
        assert proc.max_rounds == 5

    def test_custom_name(self) -> None:
        proc = RoundWindowProcessor(name="my-window")
        assert proc.name == "my-window"

    @pytest.mark.asyncio
    async def test_no_history(self) -> None:
        """No history — no-op."""
        proc = RoundWindowProcessor(max_rounds=2)
        ctx = _make_ctx()
        await proc.process(ctx, {})
        assert ctx.state.get("history") is None

    @pytest.mark.asyncio
    async def test_empty_history(self) -> None:
        """Empty history list — no-op."""
        proc = RoundWindowProcessor(max_rounds=2)
        ctx = _make_ctx(history=[])
        await proc.process(ctx, {})
        assert ctx.state.get("history") == []

    @pytest.mark.asyncio
    async def test_within_limit_no_change(self) -> None:
        """Fewer rounds than max — history unchanged."""
        proc = RoundWindowProcessor(max_rounds=3)
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "good"},
        ]
        original = list(history)
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        assert ctx.state.get("history") == original

    @pytest.mark.asyncio
    async def test_exact_limit_no_change(self) -> None:
        """Exactly max_rounds rounds — no trimming."""
        proc = RoundWindowProcessor(max_rounds=2)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        original = list(history)
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        assert ctx.state.get("history") == original

    @pytest.mark.asyncio
    async def test_trims_oldest_rounds(self) -> None:
        """Three rounds with max_rounds=2 — first round removed."""
        proc = RoundWindowProcessor(max_rounds=2)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 4
        assert result[0]["content"] == "q2"
        assert result[1]["content"] == "a2"
        assert result[2]["content"] == "q3"
        assert result[3]["content"] == "a3"

    @pytest.mark.asyncio
    async def test_system_messages_preserved(self) -> None:
        """System messages are always kept, placed before rounds."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1]["content"] == "q2"
        assert result[2]["content"] == "a2"

    @pytest.mark.asyncio
    async def test_multiple_system_messages(self) -> None:
        """Multiple system messages are all preserved."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 4
        assert result[0]["content"] == "sys1"
        assert result[1]["content"] == "sys2"
        assert result[2]["content"] == "q2"
        assert result[3]["content"] == "a2"

    @pytest.mark.asyncio
    async def test_rounds_with_tool_calls(self) -> None:
        """Tool-call messages between user/assistant are part of the round."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": None, "tool_calls": [{"name": "search"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 4
        assert result[0]["content"] == "q2"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"
        assert result[3]["content"] == "a2"

    @pytest.mark.asyncio
    async def test_system_messages_interleaved(self) -> None:
        """System messages in the middle of history are still preserved."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "updated instructions"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 3
        assert result[0]["content"] == "updated instructions"
        assert result[1]["content"] == "q2"
        assert result[2]["content"] == "a2"

    @pytest.mark.asyncio
    async def test_single_round_max_one(self) -> None:
        """Single round with max_rounds=1 — unchanged."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_incomplete_round_at_end(self) -> None:
        """Trailing user message without assistant reply counts as a round."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 1
        assert result[0]["content"] == "q2"

    @pytest.mark.asyncio
    async def test_only_system_messages(self) -> None:
        """Only system messages — no rounds, no change."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        # No rounds at all, so len(rounds)=0 <= max_rounds=1, no change
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mutates_history_in_place(self) -> None:
        """History list is mutated in-place (same object reference)."""
        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        # Same list object
        assert ctx.state.get("history") is history
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_pydantic_message_objects(self) -> None:
        """Works with Pydantic-style message objects (attribute access)."""

        class FakeMsg:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        proc = RoundWindowProcessor(max_rounds=1)
        history = [
            FakeMsg("user", "q1"),
            FakeMsg("assistant", "a1"),
            FakeMsg("user", "q2"),
            FakeMsg("assistant", "a2"),
        ]
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 2
        assert result[0].content == "q2"
        assert result[1].content == "a2"

    @pytest.mark.asyncio
    async def test_with_pipeline(self) -> None:
        """Works when registered in a ProcessorPipeline."""
        proc = RoundWindowProcessor(max_rounds=1)
        pipeline = ProcessorPipeline()
        pipeline.register(proc)
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ctx = _make_ctx(history=history)
        await pipeline.fire("pre_llm_call", ctx)
        result = ctx.state.get("history")
        assert len(result) == 2
        assert result[0]["content"] == "q2"

    @pytest.mark.asyncio
    async def test_many_rounds_trimmed(self) -> None:
        """Large number of rounds trimmed correctly."""
        proc = RoundWindowProcessor(max_rounds=2)
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"q{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        ctx = _make_ctx(history=history)
        await proc.process(ctx, {})
        result = ctx.state.get("history")
        assert len(result) == 4
        assert result[0]["content"] == "q8"
        assert result[1]["content"] == "a8"
        assert result[2]["content"] == "q9"
        assert result[3]["content"] == "a9"
