"""Feed -> portfolio wired together with hand-written signals and fills.

The execution handler arrives on Day 4; here fills are applied by hand at the
next bar's open, which is exactly the convention the real handler will follow.
"""

import pytest

from quantlab.events import FillEvent, SignalEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.portfolio import Portfolio

COMMISSION = 1.0


def test_scripted_round_trip_matches_independent_arithmetic(market):
    frame = market["AAA"]
    feed = HistoricalDataFeed({"AAA": frame})
    pf = Portfolio(100_000.0)

    pending = []
    entry_i, exit_i = 5, 20
    for i, event in enumerate(feed):
        for order in pending:  # fills happen at the open of the bar after the signal
            price = event.bars[order.symbol].open
            pf.on_fill(
                FillEvent(
                    event.timestamp, order.symbol, order.quantity, price,
                    commission=COMMISSION, order_id=order.order_id,
                )
            )
        pending = []

        pf.on_market(event)

        weight = {entry_i: 0.5, exit_i: 0.0}.get(i)
        if weight is not None:
            order = pf.on_signal(SignalEvent(event.timestamp, "AAA", weight))
            assert order is not None
            pending.append(order)

    close = frame["close"].to_numpy()
    qty = float(int(0.5 * 100_000.0 / close[entry_i]))
    cash_after_entry = 100_000.0 - qty * close[entry_i] - COMMISSION
    final_cash = cash_after_entry + qty * close[exit_i] - COMMISSION

    curve = pf.equity_curve()
    assert len(curve) == len(frame)
    assert curve.iloc[entry_i] == pytest.approx(100_000.0)  # nothing filled yet
    for t in range(entry_i + 1, exit_i + 1):
        assert curve.iloc[t] == pytest.approx(cash_after_entry + qty * close[t])
    assert curve.iloc[exit_i + 1 :].to_numpy() == pytest.approx(final_cash)

    assert pf.position_quantity("AAA") == 0.0
    assert pf.open_orders == {}
    assert pf.realized_pnl == pytest.approx(qty * (close[exit_i] - close[entry_i]))
    assert pf.total_commission == 2 * COMMISSION
    assert pf.cash == pytest.approx(final_cash)
    assert len(pf.trades_frame()) == 2
