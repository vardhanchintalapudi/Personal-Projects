"""Day 4 demo: what do trading costs do to a strategy?

Runs the same momentum strategy under three execution models and prints the cost
breakdown, plus the effect of a participation limit on a thinly traded market.

    python scripts/demo_day4.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.engine import BacktestEngine
from quantlab.execution import (
    BpsCommission,
    ExecutionHandler,
    FixedBpsSlippage,
    SquareRootImpact,
)
from quantlab.portfolio import Portfolio
from quantlab.strategy import TimeSeriesMomentum

SYMBOLS = ["EQ_A", "EQ_B", "EQ_C", "EQ_D"]
CAPITAL = 5_000_000.0


def run(market, handler):
    strategy = TimeSeriesMomentum(SYMBOLS, 60, rebalance_every=5)
    return BacktestEngine(market, strategy, Portfolio(CAPITAL), handler).run()


def main() -> None:
    liquid = generate_ohlcv_panel(
        SYMBOLS, periods=1000, regimes=DEFAULT_REGIMES, base_volume=2_000_000, seed=7
    )
    thin = generate_ohlcv_panel(
        SYMBOLS, periods=1000, regimes=DEFAULT_REGIMES, base_volume=40_000, seed=7
    )

    scenarios = {
        "Free trading": (liquid, ExecutionHandler(max_participation=None)),
        "Commission only (2 bps)": (
            liquid,
            ExecutionHandler(BpsCommission(2), max_participation=None),
        ),
        "Commission + slippage + impact": (
            liquid,
            ExecutionHandler(
                BpsCommission(2),
                FixedBpsSlippage(2),
                SquareRootImpact(coefficient=0.5),
                max_participation=None,
            ),
        ),
        "Thin market, 5% participation cap": (
            thin,
            ExecutionHandler(
                BpsCommission(2),
                FixedBpsSlippage(2),
                SquareRootImpact(coefficient=0.5),
                max_participation=0.05,
            ),
        ),
    }

    rows = []
    for name, (market, handler) in scenarios.items():
        result = run(market, handler)
        s = result.summary()
        rows.append(
            {
                "scenario": name,
                "total_return": s["total_return"],
                "commission": s["total_commission"],
                "slippage": s["total_slippage"],
                "impact": s["total_impact"],
                "fills": s["n_fills"],
                "unfilled_at_end": len(result.open_orders_at_end),
            }
        )

    pd.options.display.float_format = "{:,.3f}".format
    print(pd.DataFrame(rows).set_index("scenario").to_string())
    print("\nSame signals, same prices: the gaps between rows are purely execution.")


if __name__ == "__main__":
    main()
