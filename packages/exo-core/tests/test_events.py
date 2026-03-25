"""Tests for exo.events — async event bus."""

from exo.events import EventBus

# --- Basic emit ---


class TestEmit:
    async def test_emit_calls_handler(self) -> None:
        bus = EventBus()
        calls: list[dict] = []

        async def handler(**data: object) -> None:
            calls.append(dict(data))

        bus.on("test", handler)
        await bus.emit("test", value=42)
        assert calls == [{"value": 42}]

    async def test_emit_multiple_handlers_in_order(self) -> None:
        bus = EventBus()
        order: list[str] = []

        async def first(**data: object) -> None:
            order.append("first")

        async def second(**data: object) -> None:
            order.append("second")

        bus.on("test", first)
        bus.on("test", second)
        await bus.emit("test")
        assert order == ["first", "second"]

    async def test_emit_no_handlers(self) -> None:
        bus = EventBus()
        # Should not raise
        await bus.emit("nonexistent", value=1)


# --- Event isolation ---


class TestEventIsolation:
    async def test_different_events_are_isolated(self) -> None:
        bus = EventBus()
        a_calls: list[str] = []
        b_calls: list[str] = []

        async def handler_a(**data: object) -> None:
            a_calls.append("a")

        async def handler_b(**data: object) -> None:
            b_calls.append("b")

        bus.on("event_a", handler_a)
        bus.on("event_b", handler_b)

        await bus.emit("event_a")
        assert a_calls == ["a"]
        assert b_calls == []

        await bus.emit("event_b")
        assert b_calls == ["b"]
        assert a_calls == ["a"]  # still just one call


# --- Off (unsubscribe) ---


class TestOff:
    async def test_off_removes_handler(self) -> None:
        bus = EventBus()
        calls: list[int] = []

        async def handler(**data: object) -> None:
            calls.append(1)

        bus.on("test", handler)
        await bus.emit("test")
        assert len(calls) == 1

        bus.off("test", handler)
        await bus.emit("test")
        assert len(calls) == 1  # not called again

    async def test_off_idempotent(self) -> None:
        bus = EventBus()

        async def handler(**data: object) -> None:
            pass

        # Should not raise even if handler was never registered
        bus.off("test", handler)

    async def test_off_removes_first_occurrence_only(self) -> None:
        bus = EventBus()
        calls: list[int] = []

        async def handler(**data: object) -> None:
            calls.append(1)

        bus.on("test", handler)
        bus.on("test", handler)  # registered twice
        bus.off("test", handler)  # remove first
        await bus.emit("test")
        assert len(calls) == 1  # second registration still fires


# --- has_handlers ---


class TestHasHandlers:
    def test_no_handlers(self) -> None:
        bus = EventBus()
        assert bus.has_handlers("test") is False

    def test_with_handler(self) -> None:
        bus = EventBus()

        async def handler(**data: object) -> None:
            pass

        bus.on("test", handler)
        assert bus.has_handlers("test") is True

    def test_after_off(self) -> None:
        bus = EventBus()

        async def handler(**data: object) -> None:
            pass

        bus.on("test", handler)
        bus.off("test", handler)
        assert bus.has_handlers("test") is False


# --- Clear ---


class TestClear:
    async def test_clear_removes_all(self) -> None:
        bus = EventBus()
        calls: list[str] = []

        async def handler_a(**data: object) -> None:
            calls.append("a")

        async def handler_b(**data: object) -> None:
            calls.append("b")

        bus.on("event1", handler_a)
        bus.on("event2", handler_b)
        bus.clear()

        await bus.emit("event1")
        await bus.emit("event2")
        assert calls == []
        assert bus.has_handlers("event1") is False
        assert bus.has_handlers("event2") is False


# --- Async behavior ---


class TestAsyncBehavior:
    async def test_handlers_receive_kwargs(self) -> None:
        bus = EventBus()
        received: dict = {}

        async def handler(**data: object) -> None:
            received.update(data)

        bus.on("test", handler)
        await bus.emit("test", agent="bot", step=3)
        assert received == {"agent": "bot", "step": 3}

    async def test_handler_with_no_kwargs(self) -> None:
        bus = EventBus()
        called = False

        async def handler(**data: object) -> None:
            nonlocal called
            called = True

        bus.on("test", handler)
        await bus.emit("test")
        assert called is True
