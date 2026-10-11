"""Helpers for testing strategies and for early demos.

``naive_backtest`` is a stand-in for the Day 4 execution handler: it fills every
order at the *next bar's open* plus an optional proportional commission. It does
not model slippage, partial fills or limit/stop orders. The timing convention
(decide on bar *t*'s close, trade on bar *t+1*'s open) is the same one the real
engine uses, so results transfer once costs are added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.events import FillEvent, OrderEvent, SignalEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.portfolio import Portfolio
from quantlab.strategy.base import Strategy


def frame_from_closes(
    closes: Sequence[float] | np.ndarray,
    *,
    start: str = "2020-01-01",
    freq: str = "B",
    volume: float = 1_000.0,
    range_pct: float = 0.001,
) -> pd.DataFrame:
    """Build a valid OHLCV frame from a close-price path.

    ``open[t]`` equals ``close[t-1]`` (the first open equals the first close);
    high and low bracket each bar by ``range_pct``. Handy for unit tests where
    the exact price path matters.
    """
    path = np.asarray(closes, dtype="float64")
    if path.ndim != 1 or len(path) < 2:
        raise ValueError("closes must be a 1-D sequence with at least two values")
    opens = np.concatenate([[path[0]], path[:-1]])
    high = np.maximum(opens, path) * (1.0 + range_pct)
    low = np.minimum(opens, path) * (1.0 - range_pct)
    index = pd.date_range(start=start, periods=len(path), freq=freq, name="date")
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": path, "volume": float(volume)},
        index=index,
    )


def collect_signals(
    strategy: Strategy,
    market: Mapping[str, pd.DataFrame],
    *,
    validate: bool = True,
) -> list[SignalEvent]:
    """Run ``strategy`` over ``market`` and return every signal it emits."""
    feed = HistoricalDataFeed(market, validate=validate)
    strategy.on_start(feed)
    signals: list[SignalEvent] = []
    for event in feed:
        signals.extend(strategy.on_bar(event, feed))
    return signals


def naive_backtest(
    market: Mapping[str, pd.DataFrame],
    strategy: Strategy,
    portfolio: Portfolio | None = None,
    *,
    commission_bps: float = 0.0,
) -> Portfolio:
    """Backtest with next-open fills (see module docstring for limitations).

    Orders for an instrument with no bar on the next timestamp stay pending
    until it trades again.
    """
    if commission_bps < 0:
        raise ValueError("commission_bps must be non-negative")
    portfolio = portfolio if portfolio is not None else Portfolio(100_000.0)

    feed = HistoricalDataFeed(market)
    strategy.on_start(feed)

    pending: list[OrderEvent] = []
    for event in feed:
        carried: list[OrderEvent] = []
        for order in pending:
            bar = event.bars.get(order.symbol)
            if bar is None:
                carried.append(order)
                continue
            commission = abs(order.quantity) * bar.open * commission_bps / 1e4
            portfolio.on_fill(
                FillEvent(
                    timestamp=event.timestamp,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=bar.open,
                    commission=commission,
                    order_id=order.order_id,
                    strategy_id=order.strategy_id,
                )
            )
        pending = carried

        portfolio.on_market(event)

        for signal in strategy.on_bar(event, feed):
            order = portfolio.on_signal(signal)
            if order is not None:
                pending.append(order)
    return portfolio
