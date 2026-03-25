"""W3C Baggage propagation and span consumer plugin system."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote_plus, unquote_plus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carrier protocol — abstraction over header containers
# ---------------------------------------------------------------------------

BAGGAGE_HEADER = "baggage"

# W3C Baggage size limits (RFC 9110 / W3C Baggage spec)
MAX_HEADER_LENGTH = 8192
MAX_PAIR_LENGTH = 4096
MAX_PAIRS = 180


@runtime_checkable
class Carrier(Protocol):
    """Minimal protocol for reading/writing propagation headers."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...


class DictCarrier:
    """Carrier backed by a plain dict."""

    __slots__ = ("_headers",)

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers: dict[str, str] = headers if headers is not None else {}

    def get(self, key: str) -> str | None:
        return self._headers.get(key)

    def set(self, key: str, value: str) -> None:
        self._headers[key] = value

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    def __repr__(self) -> str:
        return f"DictCarrier({self._headers!r})"


# ---------------------------------------------------------------------------
# Baggage context — async-safe storage via ContextVar
# ---------------------------------------------------------------------------

_BAGGAGE_VAR: ContextVar[dict[str, str] | None] = ContextVar("exo.baggage", default=None)


def get_baggage() -> dict[str, str]:
    """Return the current baggage as a read-only copy."""
    val = _BAGGAGE_VAR.get()
    return dict(val) if val is not None else {}


def get_baggage_value(key: str) -> str | None:
    """Return a single baggage value, or None if not set."""
    val = _BAGGAGE_VAR.get()
    if val is None:
        return None
    return val.get(key)


def set_baggage(key: str, value: str) -> None:
    """Set a single baggage key-value pair in the current context."""
    val = _BAGGAGE_VAR.get()
    current = dict(val) if val is not None else {}
    current[key] = value
    _BAGGAGE_VAR.set(current)


def clear_baggage() -> None:
    """Remove all baggage entries from the current context."""
    _BAGGAGE_VAR.set(None)


# ---------------------------------------------------------------------------
# W3C Baggage propagator
# ---------------------------------------------------------------------------

_PAIR_SEPARATOR = re.compile(r"[ \t]*,[ \t]*")


class BaggagePropagator:
    """Extract and inject W3C Baggage headers (RFC 9110).

    Handles URL-encoding of keys/values and enforces size limits.
    """

    def extract(self, carrier: Carrier) -> dict[str, str]:
        """Extract baggage from a carrier into the current context.

        Returns the extracted key-value pairs.
        """
        raw = carrier.get(BAGGAGE_HEADER)
        if not raw:
            return {}

        if len(raw) > MAX_HEADER_LENGTH:
            logger.warning(
                "baggage header exceeds max length (%d > %d), ignoring", len(raw), MAX_HEADER_LENGTH
            )
            return {}

        pairs: dict[str, str] = {}
        entries = _PAIR_SEPARATOR.split(raw)
        for entry in entries[:MAX_PAIRS]:
            entry = entry.strip()
            if not entry or len(entry) > MAX_PAIR_LENGTH:
                continue
            if "=" not in entry:
                continue
            key_part, _, value_part = entry.partition("=")
            key = unquote_plus(key_part.strip())
            value = unquote_plus(value_part.strip())
            if key:
                pairs[key] = value
                set_baggage(key, value)

        logger.debug("extracted %d baggage entries", len(pairs))
        return pairs

    def inject(self, carrier: Carrier, baggage: dict[str, str] | None = None) -> None:
        """Inject baggage into a carrier.

        Uses the provided *baggage* dict, or falls back to the current
        context baggage if *baggage* is ``None``.
        """
        items = baggage if baggage is not None else get_baggage()
        if not items:
            return

        encoded_pairs: list[str] = []
        for key, value in items.items():
            pair = f"{quote_plus(key)}={quote_plus(value)}"
            if len(pair) <= MAX_PAIR_LENGTH:
                encoded_pairs.append(pair)
            if len(encoded_pairs) >= MAX_PAIRS:
                break

        if encoded_pairs:
            header_value = ",".join(encoded_pairs)
            if len(header_value) <= MAX_HEADER_LENGTH:
                carrier.set(BAGGAGE_HEADER, header_value)

    def __repr__(self) -> str:
        return "BaggagePropagator()"


# ---------------------------------------------------------------------------
# Span consumer plugin system
# ---------------------------------------------------------------------------


class SpanConsumer(ABC):
    """Abstract base class for span consumers.

    Span consumers receive completed spans for processing (e.g. logging,
    exporting to external systems, analytics).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this consumer."""

    @abstractmethod
    def consume(self, spans: Sequence[Any]) -> None:
        """Process a batch of completed spans."""


_CONSUMER_REGISTRY: dict[str, SpanConsumer] = {}


def register_span_consumer(consumer: SpanConsumer | None = None) -> Any:
    """Register a span consumer, directly or as a decorator.

    Direct usage::

        register_span_consumer(MyConsumer())

    Decorator usage::

        @register_span_consumer
        class MyConsumer(SpanConsumer):
            ...

    When used as a decorator on a class, the class is instantiated with no
    arguments and the resulting instance is registered.
    """
    if consumer is not None:
        if isinstance(consumer, SpanConsumer):
            _CONSUMER_REGISTRY[consumer.name] = consumer
            logger.debug("registered span consumer %r", consumer.name)
            return consumer
        # Used as a bare decorator on a class: @register_span_consumer
        cls = consumer
        instance = cls()
        if isinstance(instance, SpanConsumer):
            _CONSUMER_REGISTRY[instance.name] = instance
        return cls

    # Should not reach here — register_span_consumer is not called with None.
    def _decorator(cls: Any) -> Any:
        instance = cls()
        if isinstance(instance, SpanConsumer):
            _CONSUMER_REGISTRY[instance.name] = instance
        return cls

    return _decorator


def get_span_consumer(name: str) -> SpanConsumer | None:
    """Look up a registered span consumer by name."""
    return _CONSUMER_REGISTRY.get(name)


def list_span_consumers() -> list[str]:
    """Return the names of all registered span consumers."""
    return list(_CONSUMER_REGISTRY.keys())


def dispatch_spans(spans: Sequence[Any]) -> None:
    """Send a batch of spans to all registered consumers."""
    for consumer in _CONSUMER_REGISTRY.values():
        try:
            consumer.consume(spans)
        except Exception:
            logger.error(
                "span consumer %r failed to process %d spans",
                consumer.name,
                len(spans),
                exc_info=True,
            )


def clear_span_consumers() -> None:
    """Remove all registered span consumers (useful for testing)."""
    _CONSUMER_REGISTRY.clear()
