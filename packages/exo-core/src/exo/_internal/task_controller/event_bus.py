"""In-memory event bus for task lifecycle events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskEventType(StrEnum):
    """Lifecycle event types emitted by the task controller."""

    CREATED = "task.created"
    STARTED = "task.started"
    COMPLETED = "task.completed"
    FAILED = "task.failed"
    PAUSED = "task.paused"
    CANCELED = "task.canceled"


def _now() -> datetime:
    return datetime.now()


@dataclass
class TaskEvent:
    """A single task lifecycle event.

    Args:
        event_type: The lifecycle event type.
        task_id: ID of the task that triggered the event.
        data: Arbitrary event payload.
        timestamp: When the event occurred.
    """

    event_type: TaskEventType
    task_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)


# Handler type: async callable that receives a TaskEvent.
TaskEventHandler = Callable[[TaskEvent], Coroutine[Any, Any, None]]


class TaskEventBus:
    """Pub/sub event bus for task lifecycle events.

    Handlers subscribe to specific event types and are called
    asynchronously when matching events are emitted.
    """

    def __init__(self) -> None:
        self._handlers: dict[TaskEventType, list[TaskEventHandler]] = defaultdict(list)

    def subscribe(self, event_type: TaskEventType, handler: TaskEventHandler) -> None:
        """Register *handler* to be called when *event_type* is emitted.

        Args:
            event_type: The event type to listen for.
            handler: Async callable ``(TaskEvent) -> None``.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: TaskEventType, handler: TaskEventHandler) -> None:
        """Remove *handler* from *event_type* subscriptions.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.
        """
        import contextlib

        handlers = self._handlers.get(event_type)
        if handlers:
            with contextlib.suppress(ValueError):
                handlers.remove(handler)

    async def emit(self, event: TaskEvent) -> None:
        """Emit *event*, calling all subscribed handlers.

        Handlers are called sequentially in subscription order.

        Args:
            event: The event to broadcast.
        """
        for handler in self._handlers.get(event.event_type, []):
            await handler(event)
