"""Tests for trajectory capture, export, and strategy patterns."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any

from exo.train.trajectory import (  # pyright: ignore[reportMissingImports]
    DefaultStrategy,
    TrajectoryDataset,
    TrajectoryError,
    TrajectoryItem,
    TrajectoryStrategy,
)

# ---------------------------------------------------------------------------
# TrajectoryItem
# ---------------------------------------------------------------------------


class TestTrajectoryItemCreation:
    def test_defaults(self) -> None:
        item = TrajectoryItem()
        assert item.task_id == ""
        assert item.agent_id == ""
        assert item.step == 0
        assert item.input == ""
        assert item.output == ""
        assert item.tool_calls == ()
        assert item.messages == ()
        assert item.score is None
        assert item.status == "success"
        assert len(item.id) == 12

    def test_custom_fields(self) -> None:
        item = TrajectoryItem(
            id="abc",
            task_id="t1",
            agent_id="a1",
            step=3,
            input="hello",
            output="world",
            score=0.9,
            status="failed",
        )
        assert item.id == "abc"
        assert item.task_id == "t1"
        assert item.step == 3
        assert item.score == 0.9

    def test_frozen(self) -> None:
        item = TrajectoryItem()
        try:
            item.step = 5  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except AttributeError:
            pass

    def test_with_messages_and_tool_calls(self) -> None:
        msgs = ({"role": "user", "content": "hi"},)
        tcs = ({"id": "tc1", "name": "foo", "arguments": "{}"},)
        item = TrajectoryItem(messages=msgs, tool_calls=tcs)
        assert len(item.messages) == 1
        assert len(item.tool_calls) == 1


class TestTrajectoryItemSerialization:
    def test_to_dict(self) -> None:
        item = TrajectoryItem(id="x", task_id="t1", agent_id="a1", step=1)
        d = item.to_dict()
        assert d["id"] == "x"
        assert d["task_id"] == "t1"
        assert isinstance(d["messages"], list)
        assert isinstance(d["tool_calls"], list)

    def test_from_dict(self) -> None:
        d = {"id": "y", "task_id": "t2", "agent_id": "a2", "step": 2, "score": 0.5}
        item = TrajectoryItem.from_dict(d)
        assert item.id == "y"
        assert item.task_id == "t2"
        assert item.score == 0.5

    def test_roundtrip(self) -> None:
        original = TrajectoryItem(
            id="rt",
            task_id="t1",
            agent_id="a1",
            step=5,
            input="q",
            output="a",
            messages=({"role": "user", "content": "q"},),
            tool_calls=({"id": "tc", "name": "fn", "arguments": "{}"},),
            score=0.8,
            status="success",
            metadata={"key": "val"},
        )
        restored = TrajectoryItem.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.task_id == original.task_id
        assert restored.score == original.score
        assert len(restored.messages) == 1

    def test_from_dict_defaults(self) -> None:
        item = TrajectoryItem.from_dict({})
        assert item.task_id == ""
        assert item.step == 0
        assert item.score is None


# ---------------------------------------------------------------------------
# TrajectoryStrategy
# ---------------------------------------------------------------------------


class TestDefaultStrategy:
    def test_build_item_basic(self) -> None:
        strategy = DefaultStrategy()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        item = strategy.build_item(msgs, task_id="t1", agent_id="a1", step=0)
        assert item.input == "What is 2+2?"
        assert item.output == "4"
        assert item.task_id == "t1"
        assert item.agent_id == "a1"

    def test_build_item_with_tool_calls(self) -> None:
        strategy = DefaultStrategy()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "calc"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "add"}],
            },
        ]
        item = strategy.build_item(msgs)
        assert len(item.tool_calls) == 1
        assert item.tool_calls[0]["name"] == "add"

    def test_build_item_empty_messages(self) -> None:
        strategy = DefaultStrategy()
        item = strategy.build_item([])
        assert item.input == ""
        assert item.output == ""

    def test_validate_non_empty(self) -> None:
        strategy = DefaultStrategy()
        assert strategy.validate([TrajectoryItem()])
        assert not strategy.validate([])


class _AlwaysFailStrategy(TrajectoryStrategy):
    """Test strategy that always marks items as failed."""

    def build_item(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        task_id: str = "",
        agent_id: str = "",
        step: int = 0,
        **kwargs: Any,
    ) -> TrajectoryItem:
        return TrajectoryItem(task_id=task_id, agent_id=agent_id, step=step, status="custom")

    def validate(self, items: Sequence[TrajectoryItem]) -> bool:
        return all(it.status == "custom" for it in items)


class TestCustomStrategy:
    def test_custom_build(self) -> None:
        strategy = _AlwaysFailStrategy()
        item = strategy.build_item([], task_id="t")
        assert item.status == "custom"

    def test_custom_validate(self) -> None:
        strategy = _AlwaysFailStrategy()
        items = [TrajectoryItem(status="custom")]
        assert strategy.validate(items)
        assert not strategy.validate([TrajectoryItem(status="success")])


# ---------------------------------------------------------------------------
# TrajectoryDataset
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetInit:
    def test_defaults(self) -> None:
        ds = TrajectoryDataset()
        assert len(ds) == 0
        assert isinstance(ds.strategy, DefaultStrategy)

    def test_custom_strategy(self) -> None:
        s = _AlwaysFailStrategy()
        ds = TrajectoryDataset(strategy=s)
        assert ds.strategy is s

    def test_repr(self) -> None:
        ds = TrajectoryDataset()
        assert "TrajectoryDataset" in repr(ds)
        assert "items=0" in repr(ds)


class TestTrajectoryDatasetCapture:
    def test_append_trajectory(self) -> None:
        ds = TrajectoryDataset()
        item = TrajectoryItem(id="a1", task_id="t1")
        ds.append_trajectory(item)
        assert len(ds) == 1
        assert ds.items[0].id == "a1"

    def test_from_messages(self) -> None:
        ds = TrajectoryDataset()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        item = ds.from_messages(msgs, task_id="t1", agent_id="a1")
        assert item.input == "hi"
        assert item.output == "hello"
        assert item.step == 0
        assert len(ds) == 1

    def test_from_messages_step_counter(self) -> None:
        ds = TrajectoryDataset()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        ds.from_messages(msgs, task_id="t1", agent_id="a1")
        item2 = ds.from_messages(msgs, task_id="t1", agent_id="a1")
        assert item2.step == 1
        assert len(ds) == 2

    def test_from_messages_separate_agents(self) -> None:
        ds = TrajectoryDataset()
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
        ds.from_messages(msgs, task_id="t1", agent_id="a1")
        item2 = ds.from_messages(msgs, task_id="t1", agent_id="a2")
        assert item2.step == 0  # separate counter for different agent

    def test_save_task_trajectory(self) -> None:
        ds = TrajectoryDataset()
        items = [
            TrajectoryItem(task_id="t1", step=0),
            TrajectoryItem(task_id="t1", step=1),
        ]
        count = ds.save_task_trajectory("t1", items)
        assert count == 2
        assert len(ds) == 2

    def test_get_task_trajectory(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(task_id="t1"))
        ds.append_trajectory(TrajectoryItem(task_id="t2"))
        ds.append_trajectory(TrajectoryItem(task_id="t1"))
        result = ds.get_task_trajectory("t1")
        assert len(result) == 2

    def test_get_task_trajectory_empty(self) -> None:
        ds = TrajectoryDataset()
        assert ds.get_task_trajectory("missing") == []

    def test_validate_delegates(self) -> None:
        ds = TrajectoryDataset()
        assert not ds.validate()  # empty
        ds.append_trajectory(TrajectoryItem())
        assert ds.validate()

    def test_clear(self) -> None:
        ds = TrajectoryDataset()
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
        ds.from_messages(msgs, task_id="t1", agent_id="a1")
        ds.clear()
        assert len(ds) == 0
        # step counters also reset
        item = ds.from_messages(msgs, task_id="t1", agent_id="a1")
        assert item.step == 0


class TestTrajectoryDatasetExportJSON:
    def test_to_json_empty(self) -> None:
        ds = TrajectoryDataset()
        result = json.loads(ds.to_json())
        assert result == []

    def test_to_json_single(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(id="j1", task_id="t1", output="hello"))
        result = json.loads(ds.to_json())
        assert len(result) == 1
        assert result[0]["id"] == "j1"
        assert result[0]["output"] == "hello"

    def test_to_json_multiple(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(id="a"))
        ds.append_trajectory(TrajectoryItem(id="b"))
        result = json.loads(ds.to_json())
        assert len(result) == 2

    def test_to_json_preserves_complex_fields(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(
            TrajectoryItem(
                messages=({"role": "user", "content": "q"},),
                tool_calls=({"id": "tc1", "name": "fn"},),
                metadata={"k": "v"},
            )
        )
        result = json.loads(ds.to_json())
        assert len(result[0]["messages"]) == 1
        assert result[0]["metadata"]["k"] == "v"


class TestTrajectoryDatasetExportCSV:
    def test_to_csv_empty(self) -> None:
        ds = TrajectoryDataset()
        assert ds.to_csv() == ""

    def test_to_csv_single(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(id="c1", task_id="t1", output="hi"))
        text = ds.to_csv()
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "c1"
        assert rows[0]["output"] == "hi"

    def test_to_csv_complex_fields_json_encoded(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(
            TrajectoryItem(
                id="c2",
                messages=({"role": "user", "content": "q"},),
                metadata={"k": 1},
            )
        )
        text = ds.to_csv()
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        msgs = json.loads(rows[0]["messages"])
        assert len(msgs) == 1
        meta = json.loads(rows[0]["metadata"])
        assert meta["k"] == 1

    def test_to_csv_multiple(self) -> None:
        ds = TrajectoryDataset()
        ds.append_trajectory(TrajectoryItem(id="x"))
        ds.append_trajectory(TrajectoryItem(id="y"))
        text = ds.to_csv()
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2


class TestTrajectoryError:
    def test_exception(self) -> None:
        err = TrajectoryError("bad")
        assert str(err) == "bad"
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        ds = TrajectoryDataset()
        # Capture from messages
        msgs1: list[dict[str, Any]] = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]
        ds.from_messages(msgs1, task_id="t1", agent_id="agent-1")

        # Capture with tool calls
        msgs2: list[dict[str, Any]] = [
            {"role": "user", "content": "Search web"},
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [{"id": "tc1", "name": "web_search"}],
            },
        ]
        ds.from_messages(msgs2, task_id="t1", agent_id="agent-1", score=0.95)

        assert len(ds) == 2
        assert ds.items[1].step == 1
        assert ds.items[1].score == 0.95

        # Export JSON
        j = json.loads(ds.to_json())
        assert len(j) == 2

        # Export CSV
        c = ds.to_csv()
        reader = csv.DictReader(io.StringIO(c))
        assert len(list(reader)) == 2

        # Task filter
        assert len(ds.get_task_trajectory("t1")) == 2
        assert len(ds.get_task_trajectory("t2")) == 0

        # Validate
        assert ds.validate()

    def test_custom_strategy_lifecycle(self) -> None:
        ds = TrajectoryDataset(strategy=_AlwaysFailStrategy())
        ds.from_messages([], task_id="t")
        assert ds.items[0].status == "custom"
        assert ds.validate()

    def test_save_then_export(self) -> None:
        ds = TrajectoryDataset()
        items = [
            TrajectoryItem(id="s1", task_id="t1", output="a"),
            TrajectoryItem(id="s2", task_id="t1", output="b"),
        ]
        ds.save_task_trajectory("t1", items)
        j = json.loads(ds.to_json())
        assert j[0]["output"] == "a"
        assert j[1]["output"] == "b"
