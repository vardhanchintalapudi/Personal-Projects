"""Day 2 demo: replay a market through the feed and trade it with scripted signals.

The execution handler arrives on Day 4, so fills are applied by hand at the next
bar's open with a flat commission. This is the same timing the real handler uses.

    python scripts/demo_day2.py
"""

from __future__ import annotations

from quantlab.data import generate_ohlcv_panel
from quantlab.events import FillEvent, SignalEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.portfolio import Portfolio

COMMISSION = 1.0


def main() -> None:
    market = generate_ohlcv_panel(["AAA", "BBB"], periods=300, correlation=0.3, seed=42)
    feed = HistoricalDataFeed(market)
    portfolio = Portfolio(100_000.0, max_gross_leverage=1.0)

    # bar index -> {symbol: target weight}
    script = {
        10: {"AAA": 0.5, "BBB": -0.3},
        120: {"AAA": 0.2},
        200: {"AAA": 0.0, "BBB": 0.0},
    }

    pending = []
    for i, event in enumerate(feed):
        for order in pending:
            fill_price = event.bars[order.symbol].open
            portfolio.on_fill(
                FillEvent(
                    event.timestamp,
                    order.symbol,
                    order.quantity,
                    fill_price,
                    commission=COMMISSION,
                    order_id=order.order_id,
                )
            )
        pending = []

        portfolio.on_market(event)

        for symbol, weight in script.get(i, {}).items():
            order = portfolio.on_signal(SignalEvent(event.timestamp, symbol, weight))
            if order is not None:
                pending.append(order)

        # The feed only ever reveals the past: assert it on every step.
        assert feed.history("AAA").index[-1] == event.timestamp

    print("Trades")
    print(portfolio.trades_frame().round({"price": 2, "realized_pnl": 2}).to_string(index=False))
    print("\nSummary")
    for key, value in portfolio.summary().items():
        print(f"  {key:<16}{value:>14,.2f}")

    curve = portfolio.equity_curve()
    print(f"\nEquity range: {curve.min():,.0f} to {curve.max():,.0f} over {len(curve)} bars")


if __name__ == "__main__":
    main()
