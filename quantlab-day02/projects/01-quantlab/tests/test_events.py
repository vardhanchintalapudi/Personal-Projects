from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from quantlab.events import (
    Bar,
    EventQueue,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
)

TS = pd.Timestamp("2024-01-02")


def test_market_event_is_immutable_and_copies_input():
    source = {"AAA": Bar(1, 2, 0.5, 1.5, 100)}
    event = MarketEvent(TS, source)
    source["BBB"] = Bar(1, 1, 1, 1, 1)
    assert set(event.bars) == {"AAA"}
    assert event.close("AAA") == 1.5
    with pytest.raises(TypeError):
        event.bars["CCC"] = Bar(1, 1, 1, 1, 1)
    with pytest.raises(FrozenInstanceError):
        event.timestamp = TS
    assert event.type is EventType.MARKET


def test_signal_direction_and_validation():
    assert SignalEvent(TS, "A", 0.4).direction == 1
    assert SignalEvent(TS, "A", -0.4).direction == -1
    assert SignalEvent(TS, "A", 0.0).direction == 0
    with pytest.raises(ValueError):
        SignalEvent(TS, "A", float("nan"))
    with pytest.raises(ValueError):
        SignalEvent(TS, "A", float("inf"))


def test_order_validation_and_side():
    buy = OrderEvent(TS, "A", 10, order_id=1)
    sell = OrderEvent(TS, "A", -3, order_id=2)
    assert (buy.side, sell.side) == ("BUY", "SELL")
    assert buy.order_type is OrderType.MARKET
    with pytest.raises(ValueError):
        OrderEvent(TS, "A", 0, order_id=3)
    with pytest.raises(ValueError):
        OrderEvent(TS, "A", 1, order_id=4, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        OrderEvent(TS, "A", 1, order_id=5, order_type=OrderType.STOP, stop_price=-1)
    ok = OrderEvent(TS, "A", 1, order_id=6, order_type=OrderType.LIMIT, limit_price=10.0)
    assert ok.limit_price == 10.0


def test_fill_cash_flow_signs():
    buy = FillEvent(TS, "A", 10, 50.0, commission=2.0)
    sell = FillEvent(TS, "A", -10, 55.0, commission=2.0)
    assert buy.notional == 500.0
    assert buy.cash_flow == -502.0
    assert sell.notional == -550.0
    assert sell.cash_flow == 548.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantity": 0, "price": 10.0},
        {"quantity": 1, "price": 0.0},
        {"quantity": 1, "price": float("nan")},
        {"quantity": 1, "price": 10.0, "commission": -1.0},
    ],
)
def test_fill_validation(kwargs):
    with pytest.raises(ValueError):
        FillEvent(TS, "A", **kwargs)


def test_queue_fifo_and_type_checking():
    q = EventQueue()
    assert q.empty() and not q and len(q) == 0
    first, second = SignalEvent(TS, "A", 1.0), SignalEvent(TS, "B", 1.0)
    q.put(first)
    q.put(second)
    assert len(q) == 2 and q
    assert q.get() is first
    assert q.get() is second
    with pytest.raises(IndexError):
        q.get()
    with pytest.raises(TypeError):
        q.put("not an event")  # type: ignore[arg-type]


def test_queue_drain_includes_events_added_while_draining():
    q = EventQueue()
    q.put(SignalEvent(TS, "A", 1.0))
    seen = []
    for event in q.drain():
        seen.append(event.type)
        if event.type is EventType.SIGNAL:
            q.put(OrderEvent(TS, "A", 5, order_id=1))
        elif event.type is EventType.ORDER:
            q.put(FillEvent(TS, "A", 5, 10.0, order_id=1))
    assert seen == [EventType.SIGNAL, EventType.ORDER, EventType.FILL]
    assert q.empty()
    q.put(SignalEvent(TS, "A", 0.0))
    q.clear()
    assert q.empty()
