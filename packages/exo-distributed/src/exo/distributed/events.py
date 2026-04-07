"""Redis-backed event publishing and subscription for distributed streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _collector,
    _get_meter,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_STREAM_EVENT_PUBLISH_DURATION,
    METRIC_STREAM_EVENTS_EMITTED,
    STREAM_EVENT_TYPE,
)
from exo.types import (  # pyright: ignore[reportMissingImports]
    ErrorEvent,
    ReasoningEvent,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

# Mapping from event type discriminator to the concrete class.
_EVENT_TYPE_MAP: dict[str, type[Any]] = {
    "text": TextEvent,
    "tool_call": ToolCallEvent,
    "tool_call_delta": ToolCallDeltaEvent,
    "step": StepEvent,
    "tool_result": ToolResultEvent,
    "reasoning": ReasoningEvent,
    "error": ErrorEvent,
    "status": StatusEvent,
    "usage": UsageEvent,
}


def _record_event_published(event_type: str, duration: float) -> None:
    """Record metrics for a single event publish operation."""
    attrs: dict[str, str] = {STREAM_EVENT_TYPE: event_type}
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_counter(
            name=METRIC_STREAM_EVENTS_EMITTED,
            unit="1",
            description="Number of streaming events emitted",
        ).add(1, attrs)
        if duration > 0:
            meter.create_histogram(
                name=METRIC_STREAM_EVENT_PUBLISH_DURATION,
                unit="s",
                description="Duration of event publish operations",
            ).record(duration, attrs)
    else:
        _collector.add_counter(METRIC_STREAM_EVENTS_EMITTED, 1.0, attrs)
        if duration > 0:
            _collector.record_histogram(METRIC_STREAM_EVENT_PUBLISH_DURATION, duration, attrs)


def _deserialize_event(data: dict[str, Any]) -> StreamEvent:
    """Reconstruct a ``StreamEvent`` from a JSON-decoded dict.

    Uses the ``type`` field as a discriminator to pick the correct class.
    """
    event_type = data.get("type")
    cls = _EVENT_TYPE_MAP.get(event_type)  # type: ignore[arg-type]
    if cls is None:
        msg = f"Unknown event type: {event_type!r}"
        raise ValueError(msg)
    return cls(**data)  # type: ignore[return-value]


class EventPublisher:
    """Publishes streaming events to Redis Pub/Sub and Streams.

    Each event is published to:
    - A Pub/Sub channel ``exo:events:{task_id}`` for live consumption.
    - A Redis Stream ``exo:stream:{task_id}`` for persistent replay.

    Args:
        redis_url: Redis connection URL.
        stream_ttl_seconds: TTL for the persistent stream key (default 3600 = 1 hour).
    """

    def __init__(
        self,
        redis_url: str,
        *,
        stream_ttl_seconds: int = 3600,
    ) -> None:
        self._redis_url = redis_url
        self._stream_ttl_seconds = stream_ttl_seconds
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        logger.debug("EventPublisher connecting to Redis")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.debug("EventPublisher disconnected")

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            msg = "EventPublisher is not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._redis

    def _pubsub_channel(self, task_id: str) -> str:
        return f"exo:events:{task_id}"

    def _stream_key(self, task_id: str) -> str:
        return f"exo:stream:{task_id}"

    async def publish(self, task_id: str, event: StreamEvent) -> None:
        """Publish *event* to both Pub/Sub and the persistent Stream."""
        r = self._client()
        payload = json.dumps(event.model_dump())

        start = time.monotonic()

        # Publish to Pub/Sub for live subscribers.
        await r.publish(self._pubsub_channel(task_id), payload)  # type: ignore[misc]

        # Append to Stream for replay.
        stream_key = self._stream_key(task_id)
        await r.xadd(stream_key, {"event": payload})
        await r.expire(stream_key, self._stream_ttl_seconds)  # type: ignore[misc]

        duration = time.monotonic() - start
        logger.debug(
            "EventPublisher published %s event for task %s (%.3fs)",
            event.type,
            task_id,
            duration,
        )
        _record_event_published(event.type, duration)


class EventSubscriber:
    """Subscribes to streaming events from Redis.

    Provides two consumption modes:

    - :meth:`subscribe` — live Pub/Sub for real-time events.
    - :meth:`replay` — reads persisted events from a Redis Stream.

    Args:
        redis_url: Redis connection URL.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        logger.debug("EventSubscriber connecting to Redis")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.debug("EventSubscriber disconnected")

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            msg = "EventSubscriber is not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._redis

    async def subscribe(self, task_id: str) -> AsyncIterator[StreamEvent]:
        """Yield live events via Redis Pub/Sub.

        Listens on channel ``exo:events:{task_id}`` and yields
        deserialized ``StreamEvent`` instances.  The iterator ends when
        a ``StatusEvent`` with status ``"completed"`` or ``"error"`` or
        ``"cancelled"`` is received.
        """
        r = self._client()
        channel_name = f"exo:events:{task_id}"
        pubsub = r.pubsub()
        await pubsub.subscribe(channel_name)  # type: ignore[misc]
        logger.debug("EventSubscriber subscribed to %s", channel_name)
        terminal_statuses = {"completed", "error", "cancelled"}
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                if msg["type"] != "message":
                    continue
                data = json.loads(msg["data"])
                event = _deserialize_event(data)
                yield event
                # Stop on terminal status events.
                if isinstance(event, StatusEvent) and event.status in terminal_statuses:
                    logger.debug(
                        "EventSubscriber received terminal event for task %s (status=%s)",
                        task_id,
                        event.status,
                    )
                    break
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.aclose()  # type: ignore[misc]

    async def replay(self, task_id: str, from_id: str = "0") -> AsyncIterator[StreamEvent]:
        """Yield persisted events from the Redis Stream.

        Reads all entries in ``exo:stream:{task_id}`` starting from
        *from_id* (default ``"0"`` = beginning).
        """
        r = self._client()
        stream_key = f"exo:stream:{task_id}"
        entries = await r.xrange(stream_key, min=from_id)
        logger.debug(
            "EventSubscriber replaying %d events for task %s from %s",
            len(entries),
            task_id,
            from_id,
        )
        for _msg_id, fields in entries:
            data = json.loads(fields["event"])
            yield _deserialize_event(data)
