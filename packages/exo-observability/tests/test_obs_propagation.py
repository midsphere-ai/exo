"""Tests for exo.observability.propagation — W3C Baggage propagation and span consumer system."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from exo.observability.propagation import (  # pyright: ignore[reportMissingImports]
    BAGGAGE_HEADER,
    MAX_HEADER_LENGTH,
    MAX_PAIR_LENGTH,
    MAX_PAIRS,
    BaggagePropagator,
    Carrier,
    DictCarrier,
    SpanConsumer,
    clear_baggage,
    clear_span_consumers,
    dispatch_spans,
    get_baggage,
    get_baggage_value,
    get_span_consumer,
    list_span_consumers,
    register_span_consumer,
    set_baggage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_baggage() -> Any:
    """Reset baggage and consumer registries before each test."""
    clear_baggage()
    clear_span_consumers()
    yield
    clear_baggage()
    clear_span_consumers()


# ---------------------------------------------------------------------------
# DictCarrier
# ---------------------------------------------------------------------------


class TestDictCarrier:
    def test_creation_empty(self) -> None:
        c = DictCarrier()
        assert c.headers == {}

    def test_creation_with_headers(self) -> None:
        c = DictCarrier({"key": "value"})
        assert c.get("key") == "value"

    def test_get_missing(self) -> None:
        c = DictCarrier()
        assert c.get("nope") is None

    def test_set(self) -> None:
        c = DictCarrier()
        c.set("x", "y")
        assert c.get("x") == "y"

    def test_implements_carrier_protocol(self) -> None:
        c = DictCarrier()
        assert isinstance(c, Carrier)

    def test_repr(self) -> None:
        c = DictCarrier({"a": "1"})
        assert "DictCarrier" in repr(c)


# ---------------------------------------------------------------------------
# Baggage context functions
# ---------------------------------------------------------------------------


class TestBaggageContext:
    def test_default_empty(self) -> None:
        assert get_baggage() == {}

    def test_set_and_get(self) -> None:
        set_baggage("user.id", "u123")
        assert get_baggage_value("user.id") == "u123"

    def test_get_missing(self) -> None:
        assert get_baggage_value("nope") is None

    def test_set_multiple(self) -> None:
        set_baggage("a", "1")
        set_baggage("b", "2")
        bag = get_baggage()
        assert bag == {"a": "1", "b": "2"}

    def test_overwrite(self) -> None:
        set_baggage("key", "old")
        set_baggage("key", "new")
        assert get_baggage_value("key") == "new"

    def test_clear(self) -> None:
        set_baggage("x", "y")
        clear_baggage()
        assert get_baggage() == {}

    def test_get_returns_copy(self) -> None:
        set_baggage("a", "1")
        bag = get_baggage()
        bag["a"] = "modified"
        assert get_baggage_value("a") == "1"


# ---------------------------------------------------------------------------
# BaggagePropagator — extract
# ---------------------------------------------------------------------------


class TestBaggagePropagatorExtract:
    def test_extract_single_pair(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "user=alice"})
        prop = BaggagePropagator()
        pairs = prop.extract(carrier)
        assert pairs == {"user": "alice"}
        assert get_baggage_value("user") == "alice"

    def test_extract_multiple_pairs(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "a=1,b=2,c=3"})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {"a": "1", "b": "2", "c": "3"}

    def test_extract_url_encoded(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "user+name=hello+world"})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {"user name": "hello world"}

    def test_extract_empty_header(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: ""})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {}

    def test_extract_no_header(self) -> None:
        carrier = DictCarrier()
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {}

    def test_extract_whitespace_around_comma(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "a=1 , b=2 , c=3"})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {"a": "1", "b": "2", "c": "3"}

    def test_extract_skips_oversized_header(self) -> None:
        huge = "x=" + "a" * (MAX_HEADER_LENGTH + 1)
        carrier = DictCarrier({BAGGAGE_HEADER: huge})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {}

    def test_extract_skips_oversized_pair(self) -> None:
        big_pair = "big=" + "v" * MAX_PAIR_LENGTH
        carrier = DictCarrier({BAGGAGE_HEADER: f"ok=1,{big_pair}"})
        pairs = BaggagePropagator().extract(carrier)
        assert "ok" in pairs
        assert "big" not in pairs

    def test_extract_skips_no_equals(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "valid=1,invalid_entry,ok=2"})
        pairs = BaggagePropagator().extract(carrier)
        assert pairs == {"valid": "1", "ok": "2"}

    def test_extract_max_pairs_limit(self) -> None:
        entries = [f"k{i}=v{i}" for i in range(MAX_PAIRS + 10)]
        carrier = DictCarrier({BAGGAGE_HEADER: ",".join(entries)})
        pairs = BaggagePropagator().extract(carrier)
        assert len(pairs) == MAX_PAIRS

    def test_extract_stores_in_context(self) -> None:
        carrier = DictCarrier({BAGGAGE_HEADER: "session=s1,task=t1"})
        BaggagePropagator().extract(carrier)
        assert get_baggage_value("session") == "s1"
        assert get_baggage_value("task") == "t1"


# ---------------------------------------------------------------------------
# BaggagePropagator — inject
# ---------------------------------------------------------------------------


class TestBaggagePropagatorInject:
    def test_inject_single_pair(self) -> None:
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier, {"user": "alice"})
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw == "user=alice"

    def test_inject_multiple_pairs(self) -> None:
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier, {"a": "1", "b": "2"})
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw is not None
        assert "a=1" in raw
        assert "b=2" in raw

    def test_inject_url_encodes(self) -> None:
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier, {"user name": "hello world"})
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw is not None
        assert "user+name" in raw
        assert "hello+world" in raw

    def test_inject_empty_baggage(self) -> None:
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier, {})
        assert carrier.get(BAGGAGE_HEADER) is None

    def test_inject_from_context(self) -> None:
        set_baggage("ctx_key", "ctx_val")
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier)
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw is not None
        assert "ctx_key" in raw
        assert "ctx_val" in raw

    def test_inject_skips_oversized_pair(self) -> None:
        carrier = DictCarrier()
        baggage = {"ok": "fine", "big": "v" * MAX_PAIR_LENGTH}
        BaggagePropagator().inject(carrier, baggage)
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw is not None
        assert "ok=fine" in raw
        assert "big" not in raw

    def test_inject_none_baggage_uses_context(self) -> None:
        set_baggage("from_ctx", "yes")
        carrier = DictCarrier()
        BaggagePropagator().inject(carrier, None)
        raw = carrier.get(BAGGAGE_HEADER)
        assert raw is not None
        assert "from_ctx" in raw


# ---------------------------------------------------------------------------
# BaggagePropagator — roundtrip
# ---------------------------------------------------------------------------


class TestBaggagePropagatorRoundtrip:
    def test_inject_then_extract(self) -> None:
        prop = BaggagePropagator()
        original = {"user": "alice", "session": "s-123", "task": "t-456"}
        carrier = DictCarrier()
        prop.inject(carrier, original)
        clear_baggage()
        extracted = prop.extract(carrier)
        assert extracted == original

    def test_roundtrip_with_special_chars(self) -> None:
        prop = BaggagePropagator()
        original = {"path/key": "val=ue", "sp ace": "wo&rd"}
        carrier = DictCarrier()
        prop.inject(carrier, original)
        clear_baggage()
        extracted = prop.extract(carrier)
        assert extracted == original

    def test_repr(self) -> None:
        assert "BaggagePropagator" in repr(BaggagePropagator())


# ---------------------------------------------------------------------------
# SpanConsumer ABC
# ---------------------------------------------------------------------------


class _TestConsumer(SpanConsumer):
    """Concrete consumer for testing."""

    def __init__(self, consumer_name: str = "test") -> None:
        self._name = consumer_name
        self.received: list[Sequence[Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def consume(self, spans: Sequence[Any]) -> None:
        self.received.append(spans)


class TestSpanConsumerABC:
    def test_concrete_implementation(self) -> None:
        c = _TestConsumer("my-consumer")
        assert c.name == "my-consumer"

    def test_consume_receives_spans(self) -> None:
        c = _TestConsumer()
        c.consume(["span1", "span2"])
        assert len(c.received) == 1
        assert c.received[0] == ["span1", "span2"]

    def test_abstract_enforcement(self) -> None:
        with pytest.raises(TypeError):
            SpanConsumer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Consumer registration — direct
# ---------------------------------------------------------------------------


class TestConsumerRegistrationDirect:
    def test_register_instance(self) -> None:
        c = _TestConsumer("direct")
        register_span_consumer(c)
        assert get_span_consumer("direct") is c

    def test_list_consumers(self) -> None:
        register_span_consumer(_TestConsumer("a"))
        register_span_consumer(_TestConsumer("b"))
        names = list_span_consumers()
        assert "a" in names
        assert "b" in names

    def test_get_missing(self) -> None:
        assert get_span_consumer("nonexistent") is None

    def test_clear_consumers(self) -> None:
        register_span_consumer(_TestConsumer("temp"))
        clear_span_consumers()
        assert list_span_consumers() == []

    def test_overwrite(self) -> None:
        c1 = _TestConsumer("same")
        c2 = _TestConsumer("same")
        register_span_consumer(c1)
        register_span_consumer(c2)
        assert get_span_consumer("same") is c2


# ---------------------------------------------------------------------------
# Consumer registration — decorator
# ---------------------------------------------------------------------------


class TestConsumerRegistrationDecorator:
    def test_decorator_on_class(self) -> None:
        @register_span_consumer
        class MyConsumer(SpanConsumer):
            @property
            def name(self) -> str:
                return "decorated"

            def consume(self, spans: Sequence[Any]) -> None:
                pass

        assert get_span_consumer("decorated") is not None
        assert isinstance(MyConsumer, type)

    def test_decorated_class_still_usable(self) -> None:
        @register_span_consumer
        class Another(SpanConsumer):
            @property
            def name(self) -> str:
                return "another"

            def consume(self, spans: Sequence[Any]) -> None:
                pass

        instance = Another()
        assert instance.name == "another"


# ---------------------------------------------------------------------------
# dispatch_spans
# ---------------------------------------------------------------------------


class TestDispatchSpans:
    def test_dispatch_to_single_consumer(self) -> None:
        c = _TestConsumer("sink")
        register_span_consumer(c)
        dispatch_spans(["s1", "s2"])
        assert len(c.received) == 1
        assert c.received[0] == ["s1", "s2"]

    def test_dispatch_to_multiple_consumers(self) -> None:
        c1 = _TestConsumer("c1")
        c2 = _TestConsumer("c2")
        register_span_consumer(c1)
        register_span_consumer(c2)
        dispatch_spans(["span"])
        assert len(c1.received) == 1
        assert len(c2.received) == 1

    def test_dispatch_empty_spans(self) -> None:
        c = _TestConsumer("empty")
        register_span_consumer(c)
        dispatch_spans([])
        assert len(c.received) == 1
        assert c.received[0] == []

    def test_dispatch_no_consumers(self) -> None:
        dispatch_spans(["s1"])  # should not raise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_baggage_header(self) -> None:
        assert BAGGAGE_HEADER == "baggage"

    def test_max_header_length(self) -> None:
        assert MAX_HEADER_LENGTH == 8192

    def test_max_pair_length(self) -> None:
        assert MAX_PAIR_LENGTH == 4096

    def test_max_pairs(self) -> None:
        assert MAX_PAIRS == 180
