"""Tests for TrajectoryExtractor — DAG extraction from session traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orbiter.train.trajectory import (  # pyright: ignore[reportMissingImports]
    DefaultStrategy,
    TrajectoryDataset,
    TrajectoryExtractor,
    TrajectoryItem,
    TrajectoryStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_messages() -> list[dict[str, Any]]:
    """User asks a question, assistant answers."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def _tool_call_messages() -> list[dict[str, Any]]:
    """User asks, assistant calls a tool, tool responds, assistant answers."""
    return [
        {"role": "user", "content": "Search for Python"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "name": "web_search", "arguments": '{"q": "Python"}'}],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "Python is a programming language."},
        {"role": "assistant", "content": "Python is a programming language."},
    ]


def _multi_tool_chain_messages() -> list[dict[str, Any]]:
    """User asks, assistant calls two tools in sequence, then answers."""
    return [
        {"role": "user", "content": "Research and summarize AI"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "name": "search", "arguments": '{"q": "AI"}'}],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "AI search results..."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc2", "name": "fetch", "arguments": '{"url": "https://example.com"}'}],
        },
        {"role": "tool", "tool_call_id": "tc2", "content": "Page content about AI..."},
        {"role": "assistant", "content": "AI is a field of computer science."},
    ]


def _multi_turn_messages() -> list[dict[str, Any]]:
    """Two separate user turns."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well."},
    ]


def _system_plus_messages() -> list[dict[str, Any]]:
    """System message followed by a normal exchange."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


# ---------------------------------------------------------------------------
# TrajectoryExtractor — init & properties
# ---------------------------------------------------------------------------


class TestTrajectoryExtractorInit:
    def test_creates_with_dataset(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        assert ext.dataset is ds

    def test_custom_strategy_preserved(self) -> None:
        ds = TrajectoryDataset(strategy=DefaultStrategy())
        ext = TrajectoryExtractor(ds)
        assert isinstance(ext.dataset.strategy, DefaultStrategy)


# ---------------------------------------------------------------------------
# extract — empty / simple cases
# ---------------------------------------------------------------------------


class TestExtractBasic:
    def test_empty_messages(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract([])
        assert items == []
        assert len(ds) == 0

    def test_simple_exchange(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_simple_messages(), task_id="t1", agent_id="a1")
        assert len(items) == 1
        assert items[0].input == "What is 2+2?"
        assert items[0].output == "4"
        assert items[0].task_id == "t1"
        assert items[0].agent_id == "a1"
        assert items[0].step == 0

    def test_items_appended_to_dataset(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_simple_messages(), task_id="t1")
        assert len(ds) == 1
        assert ds.items[0] is items[0]

    def test_multi_turn(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_multi_turn_messages(), task_id="t1")
        assert len(items) == 2
        assert items[0].input == "Hello"
        assert items[0].output == "Hi there!"
        assert items[0].step == 0
        assert items[1].input == "How are you?"
        assert items[1].output == "I'm doing well."
        assert items[1].step == 1

    def test_system_message_creates_separate_segment(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_system_plus_messages())
        # System message forms its own segment; user+assistant is a second.
        assert len(items) == 2
        assert items[1].input == "Hi"
        assert items[1].output == "Hello!"


# ---------------------------------------------------------------------------
# extract — tool call handling (include_tool_calls=True)
# ---------------------------------------------------------------------------


class TestExtractWithToolCalls:
    def test_single_tool_call_produces_two_steps(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_tool_call_messages(), task_id="t1")
        # Two assistant messages → two incremental segments.
        assert len(items) == 2
        # Step 0: LLM call that decided to use a tool.
        assert items[0].step == 0
        assert items[0].input == "Search for Python"
        assert len(items[0].tool_calls) == 1
        assert items[0].tool_calls[0]["name"] == "web_search"
        # Step 1: Complete exchange including final answer.
        assert items[1].step == 1
        assert items[1].input == "Search for Python"
        assert items[1].output == "Python is a programming language."

    def test_multi_tool_chain_produces_three_steps(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_multi_tool_chain_messages(), task_id="t1")
        # Three assistant messages → three incremental segments.
        assert len(items) == 3
        # Step 0: First LLM call with search tool.
        assert items[0].step == 0
        assert len(items[0].tool_calls) == 1
        assert items[0].tool_calls[0]["name"] == "search"
        # Step 1: Second LLM call with fetch tool.
        assert items[1].step == 1
        assert len(items[1].tool_calls) >= 1
        # Step 2: Final answer.
        assert items[2].step == 2
        assert items[2].output == "AI is a field of computer science."

    def test_tool_calls_in_dataset(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        ext.extract(_tool_call_messages(), task_id="t1")
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# extract — include_tool_calls=False
# ---------------------------------------------------------------------------


class TestExtractWithoutToolCalls:
    def test_single_tool_call_collapsed(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(
            _tool_call_messages(),
            include_tool_calls=False,
            task_id="t1",
        )
        # Tool messages stripped; assistant with only tool_calls skipped.
        # Result: user + final assistant = 1 segment.
        assert len(items) == 1
        assert items[0].input == "Search for Python"
        assert items[0].output == "Python is a programming language."
        assert items[0].tool_calls == ()  # No tool calls extracted.

    def test_multi_tool_chain_collapsed(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(
            _multi_tool_chain_messages(),
            include_tool_calls=False,
            task_id="t1",
        )
        assert len(items) == 1
        assert items[0].output == "AI is a field of computer science."
        assert items[0].tool_calls == ()

    def test_simple_exchange_unchanged(self) -> None:
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(
            _simple_messages(),
            include_tool_calls=False,
        )
        assert len(items) == 1
        assert items[0].input == "What is 2+2?"
        assert items[0].output == "4"

    def test_assistant_with_tool_calls_and_content_kept(self) -> None:
        """If an assistant message has both tool_calls AND content,
        the content is preserved but tool_calls are stripped."""
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [{"id": "tc1", "name": "search"}],
            },
        ]
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(msgs, include_tool_calls=False)
        assert len(items) == 1
        assert items[0].output == "I'll search for that."
        assert items[0].tool_calls == ()


# ---------------------------------------------------------------------------
# extract — compatibility with DefaultStrategy
# ---------------------------------------------------------------------------


class TestDefaultStrategyCompatibility:
    def test_build_item_produces_same_result(self) -> None:
        """Verify that extracted items use DefaultStrategy.build_item()."""
        strategy = DefaultStrategy()
        ds = TrajectoryDataset(strategy=strategy)
        ext = TrajectoryExtractor(ds)

        msgs = _simple_messages()
        items = ext.extract(msgs, task_id="t1", agent_id="a1")

        # Build directly with strategy for comparison.
        direct = strategy.build_item(msgs, task_id="t1", agent_id="a1", step=0)
        assert items[0].input == direct.input
        assert items[0].output == direct.output
        assert items[0].task_id == direct.task_id
        assert items[0].agent_id == direct.agent_id
        assert items[0].step == direct.step

    def test_extracted_items_serialisable(self) -> None:
        """Items from extract() must be serialisable via to_dict()."""
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_tool_call_messages(), task_id="t1")
        for item in items:
            d = item.to_dict()
            assert isinstance(d, dict)
            assert "input" in d
            assert "output" in d
            restored = TrajectoryItem.from_dict(d)
            assert restored.task_id == item.task_id


# ---------------------------------------------------------------------------
# extract — custom strategy
# ---------------------------------------------------------------------------


class _CountingStrategy(TrajectoryStrategy):
    """Test strategy that counts how many times build_item is called."""

    def __init__(self) -> None:
        self.call_count = 0

    def build_item(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        task_id: str = "",
        agent_id: str = "",
        step: int = 0,
        **kwargs: Any,
    ) -> TrajectoryItem:
        self.call_count += 1
        return TrajectoryItem(
            task_id=task_id,
            agent_id=agent_id,
            step=step,
            input=f"call-{self.call_count}",
            messages=tuple(dict(m) for m in messages),
        )


class TestExtractCustomStrategy:
    def test_uses_dataset_strategy(self) -> None:
        strategy = _CountingStrategy()
        ds = TrajectoryDataset(strategy=strategy)
        ext = TrajectoryExtractor(ds)
        items = ext.extract(_multi_turn_messages())
        assert strategy.call_count == 2
        assert items[0].input == "call-1"
        assert items[1].input == "call-2"


# ---------------------------------------------------------------------------
# _segment_messages — unit tests
# ---------------------------------------------------------------------------


class TestSegmentMessages:
    def test_empty(self) -> None:
        assert TrajectoryExtractor._segment_messages([]) == []

    def test_simple(self) -> None:
        msgs = _simple_messages()
        segs = TrajectoryExtractor._segment_messages(msgs)
        assert len(segs) == 1
        assert len(segs[0]) == 2

    def test_multi_turn_split(self) -> None:
        msgs = _multi_turn_messages()
        segs = TrajectoryExtractor._segment_messages(msgs)
        assert len(segs) == 2

    def test_tool_call_incremental(self) -> None:
        msgs = _tool_call_messages()
        segs = TrajectoryExtractor._segment_messages(msgs, include_tool_calls=True)
        assert len(segs) == 2
        # First segment: up to first assistant.
        assert len(segs[0]) == 2
        # Second segment: up to second assistant (all 4 messages).
        assert len(segs[1]) == 4

    def test_tool_call_filtered(self) -> None:
        msgs = _tool_call_messages()
        segs = TrajectoryExtractor._segment_messages(msgs, include_tool_calls=False)
        assert len(segs) == 1
        # Only user + final assistant remain.
        roles = [m.get("role") for m in segs[0]]
        assert "tool" not in roles
        # No tool_calls key on any message.
        for m in segs[0]:
            assert "tool_calls" not in m

    def test_user_only_messages(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        segs = TrajectoryExtractor._segment_messages(msgs)
        assert len(segs) == 1
        assert segs[0][0]["content"] == "hello"

    def test_no_user_messages(self) -> None:
        """System + assistant with no user message — still one segment."""
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "assistant", "content": "Hello!"},
        ]
        segs = TrajectoryExtractor._segment_messages(msgs)
        assert len(segs) == 1


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestExtractorIntegration:
    def test_full_lifecycle(self) -> None:
        """Extract from a complex multi-turn conversation."""
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)

        # Turn 1: simple Q&A.
        # Turn 2: tool-calling exchange.
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Search for AI"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "search"}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "AI results"},
            {"role": "assistant", "content": "AI is fascinating."},
        ]

        items = ext.extract(msgs, task_id="session-1", agent_id="agent-1")

        # Turn 1: 1 item, Turn 2: 2 items (tool call produces 2 steps).
        assert len(items) == 3
        assert len(ds) == 3

        # Verify task/agent propagated.
        for item in items:
            assert item.task_id == "session-1"
            assert item.agent_id == "agent-1"

        # Verify step numbering.
        assert [it.step for it in items] == [0, 1, 2]

        # Verify content.
        assert items[0].input == "Hi"
        assert items[0].output == "Hello!"
        assert items[2].output == "AI is fascinating."

        # Dataset export works.
        import json
        exported = json.loads(ds.to_json())
        assert len(exported) == 3

    def test_extract_then_filter(self) -> None:
        """Extract with tools, then without, from the same messages."""
        msgs = _tool_call_messages()

        ds1 = TrajectoryDataset()
        ext1 = TrajectoryExtractor(ds1)
        with_tools = ext1.extract(msgs, task_id="t1")

        ds2 = TrajectoryDataset()
        ext2 = TrajectoryExtractor(ds2)
        without_tools = ext2.extract(msgs, include_tool_calls=False, task_id="t1")

        assert len(with_tools) == 2
        assert len(without_tools) == 1
        # Both capture the same final output.
        assert with_tools[-1].output == without_tools[0].output

    def test_multiple_extractions_accumulate(self) -> None:
        """Multiple extract() calls on the same extractor accumulate items."""
        ds = TrajectoryDataset()
        ext = TrajectoryExtractor(ds)
        ext.extract(_simple_messages(), task_id="t1")
        ext.extract(_simple_messages(), task_id="t2")
        assert len(ds) == 2
        assert ds.items[0].task_id == "t1"
        assert ds.items[1].task_id == "t2"
