"""Day 5 demo: full performance and risk report for several strategies vs a benchmark.

    python scripts/demo_day5.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.analytics import drawdown_table, equal_weight_returns
from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.engine import BacktestEngine
from quantlab.execution import ExecutionHandler
from quantlab.portfolio import Portfolio
from quantlab.strategy import (
    SMACrossover,
    TimeSeriesMomentum,
    VolTargetAllocation,
    ZScoreMeanReversion,
)

SYMBOLS = ["EQ_A", "EQ_B", "BOND", "GOLD"]


def main() -> None:
    market = generate_ohlcv_panel(
        SYMBOLS,
        periods=1500,
        annual_drift=[0.10, 0.10, 0.03, 0.04],
        annual_vol=[0.25, 0.22, 0.07, 0.15],
        correlation=0.3,
        regimes=DEFAULT_REGIMES,
        base_volume=3_000_000,
        seed=42,
    )
    benchmark = equal_weight_returns(market)

    strategies = {
        "SMA 20/100": SMACrossover(SYMBOLS, 20, 100),
        "Momentum 126": TimeSeriesMomentum(SYMBOLS, 126, rebalance_every=21),
        "Mean reversion": ZScoreMeanReversion(SYMBOLS, 20, 2.0, 0.5),
        "Vol target 8%": VolTargetAllocation(SYMBOLS, target_vol=0.08, lookback=60),
    }

    columns, results = {}, {}
    for name, strategy in strategies.items():
        result = BacktestEngine(
            market, strategy, Portfolio(1_000_000.0), ExecutionHandler.realistic()
        ).run()
        results[name] = result
        columns[name] = result.performance(benchmark=benchmark)

    table = pd.DataFrame(columns)
    shown = [
        "total_return",
        "cagr",
        "annual_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "var_95",
        "cvar_95",
        "beta",
        "alpha",
        "information_ratio",
    ]
    pd.options.display.float_format = "{:,.3f}".format
    print("Performance (synthetic data, realistic costs)\n")
    print(table.loc[shown].to_string())

    best = max(results, key=lambda n: columns[n]["sharpe"])
    print(f"\nWorst drawdowns of the best Sharpe strategy ({best}):")
    print(drawdown_table(results[best].returns, top=3).to_string(index=False))

    print("\nTrade statistics:")
    print(pd.DataFrame({n: r.trade_stats() for n, r in results.items()}).round(3).to_string())


if __name__ == "__main__":
    main()
