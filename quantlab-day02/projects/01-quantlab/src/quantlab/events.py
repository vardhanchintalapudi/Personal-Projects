"""Event types and the event queue that connect QuantLab's components.

Flow of information in a backtest::

    DataFeed --MarketEvent--> Strategy --SignalEvent--> Portfolio
    Portfolio --OrderEvent--> ExecutionHandler --FillEvent--> Portfolio

All events are immutable, so a component can never alter what another
component has already seen.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar

import pandas as pd


class EventType(str, Enum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV observation for a single instrument."""

    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """All bars that became available at ``timestamp``.

    Instruments with no bar at this timestamp (holidays, late listings) are
    simply absent from ``bars``.
    """

    timestamp: pd.Timestamp
    bars: Mapping[str, Bar]
    type: ClassVar[EventType] = EventType.MARKET

    def __post_init__(self) -> None:
        object.__setattr__(self, "bars", MappingProxyType(dict(self.bars)))

    def close(self, symbol: str) -> float:
        return self.bars[symbol].close


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A strategy's desired exposure to one instrument.

    ``target_weight`` is a fraction of portfolio equity: ``+0.5`` means "hold
    half of equity long", ``-0.25`` means "hold a quarter short" and ``0``
    means "be flat". The portfolio decides how many shares that implies.
    """

    timestamp: pd.Timestamp
    symbol: str
    target_weight: float
    strategy_id: str = ""
    reason: str = ""
    type: ClassVar[EventType] = EventType.SIGNAL

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_weight):
            raise ValueError("target_weight must be finite")

    @property
    def direction(self) -> int:
        """``+1`` long, ``-1`` short, ``0`` flat."""
        if abs(self.target_weight) < 1e-12:
            return 0
        return 1 if self.target_weight > 0 else -1


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A request to trade. ``quantity`` is signed: positive buys, negative sells."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    order_id: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    strategy_id: str = ""
    type: ClassVar[EventType] = EventType.ORDER

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity) or self.quantity == 0:
            raise ValueError("order quantity must be finite and non-zero")
        if self.order_type is OrderType.LIMIT and not (self.limit_price and self.limit_price > 0):
            raise ValueError("limit orders require a positive limit_price")
        if self.order_type is OrderType.STOP and not (self.stop_price and self.stop_price > 0):
            raise ValueError("stop orders require a positive stop_price")

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"


@dataclass(frozen=True, slots=True)
class FillEvent:
    """An executed trade. ``price`` already includes any slippage."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0
    order_id: int | None = None
    strategy_id: str = ""
    type: ClassVar[EventType] = EventType.FILL

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantity) or self.quantity == 0:
            raise ValueError("fill quantity must be finite and non-zero")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError("fill price must be positive")
        if not math.isfinite(self.commission) or self.commission < 0:
            raise ValueError("commission must be non-negative")

    @property
    def notional(self) -> float:
        """Signed trade value: positive for buys, negative for sells."""
        return self.quantity * self.price

    @property
    def cash_flow(self) -> float:
        """Change in cash caused by this fill (buys are negative)."""
        return -self.notional - self.commission


Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
_EVENT_TYPES = (MarketEvent, SignalEvent, OrderEvent, FillEvent)


class EventQueue:
    """First-in-first-out queue of events.

    ``drain`` keeps yielding until the queue is empty, including events that
    handlers add while it is being drained. That is what lets a chain such as
    market -> signal -> order -> fill resolve within a single bar.
    """

    def __init__(self) -> None:
        self._items: deque[Event] = deque()

    def put(self, event: Event) -> None:
        if not isinstance(event, _EVENT_TYPES):
            raise TypeError(f"not an event: {type(event).__name__}")
        self._items.append(event)

    def get(self) -> Event:
        """Remove and return the oldest event (``IndexError`` if empty)."""
        return self._items.popleft()

    def drain(self) -> Iterator[Event]:
        while self._items:
            yield self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)
