"""Tests for exo._internal.state — run state tracking."""

from __future__ import annotations

import time

from exo._internal.state import RunNode, RunNodeStatus, RunState
from exo.types import ToolCall, Usage, UserMessage

# ── RunNodeStatus ────────────────────────────────────────────────────


class TestRunNodeStatus:
    def test_enum_values(self) -> None:
        assert RunNodeStatus.INIT == "init"
        assert RunNodeStatus.RUNNING == "running"
        assert RunNodeStatus.SUCCESS == "success"
        assert RunNodeStatus.FAILED == "failed"
        assert RunNodeStatus.TIMEOUT == "timeout"

    def test_all_statuses(self) -> None:
        assert len(RunNodeStatus) == 5


# ── RunNode ──────────────────────────────────────────────────────────


class TestRunNode:
    def test_creation_defaults(self) -> None:
        node = RunNode(agent_name="test-agent")
        assert node.agent_name == "test-agent"
        assert node.step_index == 0
        assert node.status == RunNodeStatus.INIT
        assert node.group_id is None
        assert node.started_at is None
        assert node.ended_at is None
        assert node.tool_calls == []
        assert node.usage == Usage()
        assert node.error is None
        assert node.metadata == {}
        assert node.duration is None

    def test_start_transition(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        assert node.status == RunNodeStatus.RUNNING
        assert node.started_at is not None

    def test_succeed_transition(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        node.succeed(usage)
        assert node.status == RunNodeStatus.SUCCESS
        assert node.ended_at is not None
        assert node.usage == usage

    def test_succeed_without_usage(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        node.succeed()
        assert node.status == RunNodeStatus.SUCCESS
        assert node.usage == Usage()

    def test_fail_transition(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        node.fail("something broke")
        assert node.status == RunNodeStatus.FAILED
        assert node.ended_at is not None
        assert node.error == "something broke"

    def test_timeout_transition(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        node.timeout()
        assert node.status == RunNodeStatus.TIMEOUT
        assert node.ended_at is not None

    def test_duration_calculated(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        # Simulate a brief duration
        node.started_at = 100.0
        node.ended_at = 100.5
        assert node.duration == 0.5

    def test_duration_none_when_not_started(self) -> None:
        node = RunNode(agent_name="a")
        assert node.duration is None

    def test_duration_none_when_not_ended(self) -> None:
        node = RunNode(agent_name="a")
        node.start()
        assert node.duration is None

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="get_weather", arguments='{"city": "SF"}')
        node = RunNode(agent_name="a", tool_calls=[tc])
        assert len(node.tool_calls) == 1
        assert node.tool_calls[0].name == "get_weather"

    def test_with_group_id(self) -> None:
        node = RunNode(agent_name="a", group_id="parallel-1")
        assert node.group_id == "parallel-1"

    def test_with_metadata(self) -> None:
        node = RunNode(agent_name="a", metadata={"key": "value"})
        assert node.metadata["key"] == "value"

    def test_created_at_auto_set(self) -> None:
        before = time.time()
        node = RunNode(agent_name="a")
        after = time.time()
        assert before <= node.created_at <= after


# ── RunState ─────────────────────────────────────────────────────────


class TestRunState:
    def test_initial_state(self) -> None:
        state = RunState(agent_name="main-agent")
        assert state.agent_name == "main-agent"
        assert state.status == RunNodeStatus.INIT
        assert state.messages == []
        assert state.nodes == []
        assert state.iterations == 0
        assert state.total_usage == Usage()
        assert state.current_node is None
        assert not state.is_running
        assert not state.is_terminal

    def test_start(self) -> None:
        state = RunState(agent_name="a")
        state.start()
        assert state.status == RunNodeStatus.RUNNING
        assert state.is_running
        assert not state.is_terminal

    def test_add_message(self) -> None:
        state = RunState(agent_name="a")
        msg = UserMessage(content="hello")
        state.add_message(msg)
        assert len(state.messages) == 1
        assert state.messages[0].content == "hello"

    def test_add_messages(self) -> None:
        state = RunState(agent_name="a")
        msgs = [UserMessage(content="hi"), UserMessage(content="there")]
        state.add_messages(msgs)
        assert len(state.messages) == 2

    def test_new_node(self) -> None:
        state = RunState(agent_name="main")
        node = state.new_node()
        assert node.agent_name == "main"
        assert node.step_index == 0
        assert state.iterations == 1
        assert len(state.nodes) == 1
        assert state.current_node is node

    def test_new_node_custom_agent(self) -> None:
        state = RunState(agent_name="main")
        node = state.new_node(agent_name="helper")
        assert node.agent_name == "helper"

    def test_new_node_with_group(self) -> None:
        state = RunState(agent_name="main")
        node = state.new_node(group_id="group-1")
        assert node.group_id == "group-1"

    def test_multiple_nodes_increment(self) -> None:
        state = RunState(agent_name="main")
        n0 = state.new_node()
        n1 = state.new_node()
        n2 = state.new_node()
        assert n0.step_index == 0
        assert n1.step_index == 1
        assert n2.step_index == 2
        assert state.iterations == 3
        assert state.current_node is n2

    def test_record_usage(self) -> None:
        state = RunState(agent_name="a")
        state.record_usage(Usage(input_tokens=10, output_tokens=5, total_tokens=15))
        state.record_usage(Usage(input_tokens=20, output_tokens=10, total_tokens=30))
        assert state.total_usage.input_tokens == 30
        assert state.total_usage.output_tokens == 15
        assert state.total_usage.total_tokens == 45

    def test_succeed(self) -> None:
        state = RunState(agent_name="a")
        state.start()
        state.succeed()
        assert state.status == RunNodeStatus.SUCCESS
        assert state.is_terminal
        assert not state.is_running

    def test_fail(self) -> None:
        state = RunState(agent_name="a")
        state.start()
        state.fail("error")
        assert state.status == RunNodeStatus.FAILED
        assert state.is_terminal

    def test_timeout(self) -> None:
        state = RunState(agent_name="a")
        state.start()
        state.timeout()
        assert state.status == RunNodeStatus.TIMEOUT
        assert state.is_terminal

    def test_message_accumulation(self) -> None:
        """Messages accumulate correctly over multiple steps."""
        state = RunState(agent_name="a")
        state.start()
        state.add_message(UserMessage(content="q1"))
        state.add_message(UserMessage(content="q2"))
        state.add_message(UserMessage(content="q3"))
        assert len(state.messages) == 3

    def test_full_lifecycle(self) -> None:
        """End-to-end: init -> start -> nodes -> usage -> succeed."""
        state = RunState(agent_name="orchestrator")
        assert state.status == RunNodeStatus.INIT

        state.start()
        assert state.is_running

        # Step 1: LLM call
        node1 = state.new_node()
        node1.start()
        usage1 = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        node1.succeed(usage1)
        state.record_usage(usage1)

        # Step 2: tool call
        node2 = state.new_node()
        node2.start()
        usage2 = Usage(input_tokens=200, output_tokens=80, total_tokens=280)
        node2.succeed(usage2)
        state.record_usage(usage2)

        # Finish
        state.add_message(UserMessage(content="result"))
        state.succeed()

        assert state.is_terminal
        assert state.iterations == 2
        assert state.total_usage.total_tokens == 430
        assert len(state.messages) == 1
        assert state.nodes[0].status == RunNodeStatus.SUCCESS
        assert state.nodes[1].status == RunNodeStatus.SUCCESS
