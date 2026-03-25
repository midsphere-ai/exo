"""Tests for TaskLoopEvent, TaskLoopQueue, and task loop tools."""

from __future__ import annotations

import pytest

from orbiter.task_controller import (
    TaskLoopEvent,
    TaskLoopEventType,
    TaskLoopQueue,
    abort_agent_tool,
    get_task_loop_tools,
    steer_agent_tool,
)


# ---------------------------------------------------------------------------
# TaskLoopEventType tests
# ---------------------------------------------------------------------------


class TestTaskLoopEventType:
    def test_priority_ordering(self) -> None:
        assert TaskLoopEventType.ABORT < TaskLoopEventType.STEER
        assert TaskLoopEventType.STEER < TaskLoopEventType.FOLLOWUP

    def test_integer_values(self) -> None:
        assert TaskLoopEventType.ABORT == 0
        assert TaskLoopEventType.STEER == 1
        assert TaskLoopEventType.FOLLOWUP == 2


# ---------------------------------------------------------------------------
# TaskLoopEvent tests
# ---------------------------------------------------------------------------


class TestTaskLoopEvent:
    def test_defaults(self) -> None:
        evt = TaskLoopEvent(type=TaskLoopEventType.STEER)
        assert evt.type == TaskLoopEventType.STEER
        assert evt.content == ""
        assert evt.metadata == {}

    def test_with_content_and_metadata(self) -> None:
        evt = TaskLoopEvent(
            type=TaskLoopEventType.ABORT,
            content="stop now",
            metadata={"source": "monitor"},
        )
        assert evt.content == "stop now"
        assert evt.metadata == {"source": "monitor"}

    def test_ordering_by_type(self) -> None:
        abort = TaskLoopEvent(type=TaskLoopEventType.ABORT, _seq=1)
        steer = TaskLoopEvent(type=TaskLoopEventType.STEER, _seq=0)
        assert abort < steer

    def test_ordering_same_type_by_seq(self) -> None:
        first = TaskLoopEvent(type=TaskLoopEventType.STEER, _seq=0)
        second = TaskLoopEvent(type=TaskLoopEventType.STEER, _seq=1)
        assert first < second


# ---------------------------------------------------------------------------
# TaskLoopQueue tests
# ---------------------------------------------------------------------------


class TestTaskLoopQueue:
    def test_empty_queue(self) -> None:
        q = TaskLoopQueue()
        assert q.pop() is None
        assert q.peek() is None
        assert len(q) == 0
        assert not q

    def test_push_pop(self) -> None:
        q = TaskLoopQueue()
        evt = TaskLoopEvent(type=TaskLoopEventType.STEER, content="go left")
        q.push(evt)
        assert len(q) == 1
        assert q
        result = q.pop()
        assert result is not None
        assert result.content == "go left"
        assert len(q) == 0

    def test_peek_does_not_remove(self) -> None:
        q = TaskLoopQueue()
        evt = TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="check later")
        q.push(evt)
        peeked = q.peek()
        assert peeked is not None
        assert peeked.content == "check later"
        assert len(q) == 1  # still there

    def test_priority_ordering_abort_before_steer(self) -> None:
        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="steer"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="abort"))
        result = q.pop()
        assert result is not None
        assert result.type == TaskLoopEventType.ABORT

    def test_priority_ordering_steer_before_followup(self) -> None:
        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="follow"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="steer"))
        result = q.pop()
        assert result is not None
        assert result.type == TaskLoopEventType.STEER

    def test_priority_ordering_all_three(self) -> None:
        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="f"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="a"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="s"))
        results = []
        while q:
            evt = q.pop()
            assert evt is not None
            results.append(evt.type)
        assert results == [
            TaskLoopEventType.ABORT,
            TaskLoopEventType.STEER,
            TaskLoopEventType.FOLLOWUP,
        ]

    def test_fifo_within_same_priority(self) -> None:
        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="first"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="second"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="third"))
        results = []
        while q:
            evt = q.pop()
            assert evt is not None
            results.append(evt.content)
        assert results == ["first", "second", "third"]

    def test_multiple_abort_fifo(self) -> None:
        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="a1"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="a2"))
        first = q.pop()
        second = q.pop()
        assert first is not None and first.content == "a1"
        assert second is not None and second.content == "a2"


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestTaskLoopTools:
    @pytest.mark.asyncio()
    async def test_steer_agent_pushes_event(self) -> None:
        q = TaskLoopQueue()
        tool = steer_agent_tool.bind(q)
        result = await tool.execute(content="change direction")
        assert "change direction" in result
        evt = q.pop()
        assert evt is not None
        assert evt.type == TaskLoopEventType.STEER
        assert evt.content == "change direction"

    @pytest.mark.asyncio()
    async def test_abort_agent_pushes_event(self) -> None:
        q = TaskLoopQueue()
        tool = abort_agent_tool.bind(q)
        result = await tool.execute(reason="too slow")
        assert "too slow" in result
        evt = q.pop()
        assert evt is not None
        assert evt.type == TaskLoopEventType.ABORT
        assert evt.content == "too slow"

    @pytest.mark.asyncio()
    async def test_unbound_tool_raises(self) -> None:
        from orbiter._internal.task_controller.tools import _QueueTool, _steer_agent
        from orbiter.tool import ToolError

        fresh_tool = _QueueTool(_steer_agent, name="steer_agent")
        with pytest.raises(ToolError, match="requires a bound queue"):
            await fresh_tool.execute(content="oops")

    def test_get_task_loop_tools(self) -> None:
        tools = get_task_loop_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"steer_agent", "abort_agent"}

    def test_tool_schemas_exclude_queue(self) -> None:
        schema = steer_agent_tool.to_schema()
        params = schema["function"]["parameters"]
        assert "queue" not in params.get("properties", {})
        assert "content" in params.get("properties", {})

        schema = abort_agent_tool.to_schema()
        params = schema["function"]["parameters"]
        assert "queue" not in params.get("properties", {})
        assert "reason" in params.get("properties", {})


# ---------------------------------------------------------------------------
# Agent queue drain integration tests
# ---------------------------------------------------------------------------


class TestAgentQueueDrain:
    def test_drain_abort_raises(self) -> None:
        from orbiter.agent import TaskLoopAbort, _drain_task_loop_queue
        from orbiter.types import UserMessage

        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="stop"))
        msgs: list = []
        with pytest.raises(TaskLoopAbort, match="stop"):
            _drain_task_loop_queue(q, msgs)

    def test_drain_steer_injects_message(self) -> None:
        from orbiter.agent import _drain_task_loop_queue
        from orbiter.types import UserMessage

        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="go left"))
        msgs: list = []
        _drain_task_loop_queue(q, msgs)
        assert len(msgs) == 1
        assert isinstance(msgs[0], UserMessage)
        assert "STEER" in msgs[0].content
        assert "go left" in msgs[0].content

    def test_drain_followup_injects_message(self) -> None:
        from orbiter.agent import _drain_task_loop_queue
        from orbiter.types import UserMessage

        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="also check X"))
        msgs: list = []
        _drain_task_loop_queue(q, msgs)
        assert len(msgs) == 1
        assert "FOLLOWUP" in msgs[0].content

    def test_drain_abort_takes_priority_over_steer(self) -> None:
        """If both ABORT and STEER are queued, ABORT fires first."""
        from orbiter.agent import TaskLoopAbort, _drain_task_loop_queue

        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="redirect"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.ABORT, content="halt"))
        msgs: list = []
        with pytest.raises(TaskLoopAbort, match="halt"):
            _drain_task_loop_queue(q, msgs)

    def test_drain_empty_queue_is_noop(self) -> None:
        from orbiter.agent import _drain_task_loop_queue

        q = TaskLoopQueue()
        msgs: list = []
        _drain_task_loop_queue(q, msgs)
        assert msgs == []

    def test_drain_multiple_non_abort_events(self) -> None:
        from orbiter.agent import _drain_task_loop_queue
        from orbiter.types import UserMessage

        q = TaskLoopQueue()
        q.push(TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="f1"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.STEER, content="s1"))
        q.push(TaskLoopEvent(type=TaskLoopEventType.FOLLOWUP, content="f2"))
        msgs: list = []
        _drain_task_loop_queue(q, msgs)
        assert len(msgs) == 3
        # STEER first (priority 1), then two FOLLOWUPs (priority 2) in FIFO
        assert "STEER" in msgs[0].content
        assert "FOLLOWUP" in msgs[1].content
        assert "f1" in msgs[1].content
        assert "FOLLOWUP" in msgs[2].content
        assert "f2" in msgs[2].content
