"""Tests for MemoryPersistence — hook-based auto-persistence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from orbiter.hooks import HookPoint
from orbiter.memory.base import AIMemory, HumanMemory, MemoryMetadata, SystemMemory, ToolMemory
from orbiter.memory.persistence import MemoryPersistence
from orbiter.memory.short_term import ShortTermMemory
from orbiter.types import AssistantMessage, SystemMessage, ToolResult, UserMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Any:
    """Create a minimal Agent for testing (memory=None to avoid auto-attached hooks)."""
    from orbiter.agent import Agent

    return Agent(name="test-agent", memory=None)


def _make_response(
    content: str = "Hello",
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    """Create a synthetic LLM response object."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reason="stop",
    )


def _make_tool_result(
    content: str = "result",
    error: str | None = None,
    tool_call_id: str = "tc-1",
    tool_name: str = "search",
) -> SimpleNamespace:
    """Create a synthetic tool result object."""
    return SimpleNamespace(
        content=content,
        error=error,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
    )


# ---------------------------------------------------------------------------
# Attach / Detach
# ---------------------------------------------------------------------------


class TestAttachDetach:
    def test_attach_registers_hooks(self) -> None:
        agent = _make_agent()
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        persistence.attach(agent)

        assert agent.hook_manager.has_hooks(HookPoint.POST_LLM_CALL)
        assert agent.hook_manager.has_hooks(HookPoint.POST_TOOL_CALL)

    def test_detach_removes_hooks(self) -> None:
        agent = _make_agent()
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        persistence.attach(agent)
        persistence.detach(agent)

        assert not agent.hook_manager.has_hooks(HookPoint.POST_LLM_CALL)
        assert not agent.hook_manager.has_hooks(HookPoint.POST_TOOL_CALL)

    def test_detach_idempotent(self) -> None:
        """Detaching twice does not raise."""
        agent = _make_agent()
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        persistence.attach(agent)
        persistence.detach(agent)
        persistence.detach(agent)  # should not raise

    def test_attach_idempotent(self) -> None:
        """Attaching twice registers hooks only once (second call is a no-op)."""
        agent = _make_agent()
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        persistence.attach(agent)
        hook_count_after_first = len(agent.hook_manager._hooks.get(HookPoint.POST_LLM_CALL, []))

        persistence.attach(agent)  # second call should be a no-op
        hook_count_after_second = len(agent.hook_manager._hooks.get(HookPoint.POST_LLM_CALL, []))

        assert hook_count_after_first == hook_count_after_second

    def test_attach_idempotent_different_agents(self) -> None:
        """Two different agents can each be attached once."""
        agent1 = _make_agent()
        agent2 = _make_agent()
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        persistence.attach(agent1)
        persistence.attach(agent2)
        persistence.attach(agent1)  # no-op for agent1

        # Both agents should have hooks registered exactly once
        assert agent1.hook_manager.has_hooks(HookPoint.POST_LLM_CALL)
        assert agent2.hook_manager.has_hooks(HookPoint.POST_LLM_CALL)


# ---------------------------------------------------------------------------
# POST_LLM_CALL → AIMemory
# ---------------------------------------------------------------------------


class TestSaveLLMResponse:
    async def test_saves_ai_memory(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        agent = _make_agent()
        response = _make_response(content="Hello world")

        await persistence._save_llm_response(agent=agent, response=response)

        items = await store.search(limit=100)
        assert len(items) == 1
        assert isinstance(items[0], AIMemory)
        assert items[0].content == "Hello world"
        assert items[0].tool_calls == []

    async def test_saves_tool_calls_in_ai_memory(self) -> None:
        store = ShortTermMemory(scope="user")
        persistence = MemoryPersistence(store)
        agent = _make_agent()

        tc = SimpleNamespace(id="tc-1", name="search", arguments='{"q":"test"}')
        response = _make_response(content="Calling tool", tool_calls=[tc])

        await persistence._save_llm_response(agent=agent, response=response)

        # Access _items directly — search() applies integrity filtering
        # which strips trailing AIMemory with unmatched tool_calls
        assert len(store._items) == 1
        assert isinstance(store._items[0], AIMemory)
        assert store._items[0].tool_calls == [
            {"id": "tc-1", "name": "search", "arguments": '{"q":"test"}'}
        ]

    async def test_respects_metadata(self) -> None:
        meta = MemoryMetadata(user_id="u1", session_id="s1")
        store = ShortTermMemory()
        persistence = MemoryPersistence(store, metadata=meta)
        agent = _make_agent()

        await persistence._save_llm_response(agent=agent, response=_make_response())

        items = await store.search(limit=100)
        assert items[0].metadata.user_id == "u1"
        assert items[0].metadata.session_id == "s1"

    async def test_handles_none_content(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        agent = _make_agent()

        response = SimpleNamespace(content=None, tool_calls=[], usage=None)
        await persistence._save_llm_response(agent=agent, response=response)

        items = await store.search(limit=100)
        assert items[0].content == ""


# ---------------------------------------------------------------------------
# POST_TOOL_CALL → ToolMemory
# ---------------------------------------------------------------------------


class TestSaveToolResult:
    async def test_saves_tool_memory(self) -> None:
        store = ShortTermMemory(scope="user")
        persistence = MemoryPersistence(store)
        agent = _make_agent()

        result = _make_tool_result(content="found it", tool_call_id="tc-1", tool_name="search")
        await persistence._save_tool_result(agent=agent, tool_name="search", result=result)

        # Access _items directly — search() applies integrity filtering
        # which strips orphan ToolMemory without matching AIMemory
        assert len(store._items) == 1
        item = store._items[0]
        assert isinstance(item, ToolMemory)
        assert item.content == "found it"
        assert item.tool_call_id == "tc-1"
        assert item.tool_name == "search"
        assert item.is_error is False

    async def test_saves_error_tool_memory(self) -> None:
        store = ShortTermMemory(scope="user")
        persistence = MemoryPersistence(store)
        agent = _make_agent()

        result = _make_tool_result(content=None, error="timeout", tool_name="fetch")
        await persistence._save_tool_result(agent=agent, tool_name="fetch", result=result)

        assert len(store._items) == 1
        assert store._items[0].is_error is True
        assert store._items[0].content == "timeout"

    async def test_respects_metadata(self) -> None:
        meta = MemoryMetadata(task_id="t1")
        store = ShortTermMemory(scope="user")
        persistence = MemoryPersistence(store, metadata=meta)
        agent = _make_agent()

        await persistence._save_tool_result(
            agent=agent,
            tool_name="test",
            result=_make_tool_result(),
        )

        assert len(store._items) == 1
        assert store._items[0].metadata.task_id == "t1"


# ---------------------------------------------------------------------------
# Integration: attach + stream
# ---------------------------------------------------------------------------


class TestPersistenceIntegration:
    async def test_hooks_fire_during_stream(self) -> None:
        """Verify persistence hooks work end-to-end with run.stream()."""
        from orbiter.runner import run
        from orbiter.tool import tool

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = _make_agent()
        agent._tools = {"greet": greet}

        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        persistence.attach(agent)

        # Use the stream test helpers from test_runner
        from collections.abc import AsyncIterator
        from unittest.mock import AsyncMock

        call_count = 0

        class _Chunk:
            def __init__(
                self,
                delta: str = "",
                tool_call_deltas: list[Any] | None = None,
                usage: Any = None,
            ):
                self.delta = delta
                self.tool_call_deltas = tool_call_deltas or []
                from orbiter.types import Usage

                self.usage = usage or Usage()

        class _TCD:
            def __init__(
                self,
                index: int = 0,
                id: str | None = None,
                name: str | None = None,
                arguments: str = "",
            ):
                self.index = index
                self.id = id
                self.name = name
                self.arguments = arguments

        round1 = [
            _Chunk(tool_call_deltas=[_TCD(index=0, id="tc1", name="greet")]),
            _Chunk(
                tool_call_deltas=[_TCD(index=0, arguments='{"name":"Alice"}')],
            ),
        ]
        round2 = [_Chunk(delta="Done!")]
        stream_rounds = [round1, round2]

        async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            nonlocal call_count
            chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
            call_count += 1
            for c in chunks:
                yield c

        mock_provider = AsyncMock()
        mock_provider.stream = stream

        _ = [ev async for ev in run.stream(agent, "Greet Alice", provider=mock_provider)]

        # Should have saved: AIMemory (tool call step), ToolMemory, AIMemory (text step)
        items = await store.search(limit=100)
        types = [type(i).__name__ for i in items]
        assert "AIMemory" in types

        persistence.detach(agent)


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------


class TestLoadHistory:
    async def test_returns_empty_when_no_history(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)
        assert messages == []

    async def test_converts_human_and_ai_to_messages(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        meta = MemoryMetadata(agent_id="agent-a", task_id="conv-1")

        await store.add(HumanMemory(content="hello", metadata=meta))
        await store.add(AIMemory(content="hi there", metadata=meta))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)

        assert len(messages) == 2
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "hello"
        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].content == "hi there"

    async def test_converts_system_memory(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        meta = MemoryMetadata(agent_id="agent-a", task_id="conv-1")

        await store.add(SystemMemory(content="You are a helpful assistant.", metadata=meta))
        await store.add(HumanMemory(content="hello", metadata=meta))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are a helpful assistant."

    async def test_converts_tool_memory_to_tool_result(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        meta = MemoryMetadata(agent_id="agent-a", task_id="conv-1")

        tc_item = AIMemory(
            content="calling tool",
            metadata=meta,
            tool_calls=[{"id": "tc1", "name": "search", "arguments": "{}"}],
        )
        await store.add(tc_item)
        await store.add(
            ToolMemory(
                content="result",
                metadata=meta,
                tool_call_id="tc1",
                tool_name="search",
                is_error=False,
            )
        )
        await store.add(AIMemory(content="done", metadata=meta))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)
        tool_results = [m for m in messages if isinstance(m, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_call_id == "tc1"
        assert tool_results[0].tool_name == "search"
        assert tool_results[0].content == "result"

    async def test_error_tool_memory_sets_error_field(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        meta = MemoryMetadata(agent_id="agent-a", task_id="conv-1")

        tc_item = AIMemory(
            content="calling tool",
            metadata=meta,
            tool_calls=[{"id": "tc2", "name": "fetch", "arguments": "{}"}],
        )
        await store.add(tc_item)
        await store.add(
            ToolMemory(
                content="timeout",
                metadata=meta,
                tool_call_id="tc2",
                tool_name="fetch",
                is_error=True,
            )
        )
        await store.add(AIMemory(content="sorry", metadata=meta))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)
        tool_results = [m for m in messages if isinstance(m, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].error == "timeout"
        assert tool_results[0].content == ""

    async def test_windowing_limits_rounds(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)
        meta = MemoryMetadata(agent_id="agent-a", task_id="conv-1")

        # Add 3 conversation rounds
        for i in range(3):
            await store.add(HumanMemory(content=f"human {i}", metadata=meta))
            await store.add(AIMemory(content=f"ai {i}", metadata=meta))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=2)

        user_msgs = [m for m in messages if isinstance(m, UserMessage)]
        assert len(user_msgs) == 2
        assert user_msgs[0].content == "human 1"
        assert user_msgs[1].content == "human 2"

    async def test_scoping_by_agent_and_conversation(self) -> None:
        store = ShortTermMemory()
        persistence = MemoryPersistence(store)

        meta_a = MemoryMetadata(agent_id="agent-a", task_id="conv-1")
        meta_b = MemoryMetadata(agent_id="agent-b", task_id="conv-2")

        await store.add(HumanMemory(content="agent-a message", metadata=meta_a))
        await store.add(HumanMemory(content="agent-b message", metadata=meta_b))

        messages = await persistence.load_history("agent-a", "conv-1", rounds=5)
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "agent-a message"
