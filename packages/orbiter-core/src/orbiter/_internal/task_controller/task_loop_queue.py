"""Priority event queue for agent task loop iterations.

Provides a priority queue that processes ABORT events before STEER events
before FOLLOWUP events, allowing external callers (tools, hooks) to
influence an agent's tool loop mid-execution.
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TaskLoopEventType(IntEnum):
    """Event types ordered by priority (lower = higher priority)."""

    ABORT = 0
    STEER = 1
    FOLLOWUP = 2


@dataclass(order=True)
class TaskLoopEvent:
    """A single event destined for the agent task loop.

    Events are ordered by ``(type, _seq)`` so that ABORT events are always
    processed before STEER events, which are processed before FOLLOWUP
    events.  Within the same type, events are ordered by insertion order.

    Args:
        type: The event type (determines priority).
        content: Textual content for the event (e.g. steering instruction).
        metadata: Arbitrary key-value payload.
    """

    type: TaskLoopEventType
    content: str = field(compare=False, default="")
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)
    _seq: int = field(default=0, repr=False)


class TaskLoopQueue:
    """Thread-safe priority queue for :class:`TaskLoopEvent` instances.

    Lower ``TaskLoopEventType`` values are dequeued first:
    ABORT (0) > STEER (1) > FOLLOWUP (2).

    The queue is safe to push from any thread (e.g. an external monitoring
    process) and pop from the asyncio event loop.
    """

    def __init__(self) -> None:
        self._heap: list[TaskLoopEvent] = []
        self._counter = 0
        self._lock = threading.Lock()

    def push(self, event: TaskLoopEvent) -> None:
        """Add an event to the queue.

        Args:
            event: The event to enqueue.
        """
        with self._lock:
            event._seq = self._counter
            self._counter += 1
            heapq.heappush(self._heap, event)

    def pop(self) -> TaskLoopEvent | None:
        """Remove and return the highest-priority event, or ``None`` if empty."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def peek(self) -> TaskLoopEvent | None:
        """Return the highest-priority event without removing it, or ``None``."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def __bool__(self) -> bool:
        with self._lock:
            return len(self._heap) > 0
