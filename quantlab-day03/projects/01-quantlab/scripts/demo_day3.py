"""Day 3 demo: run every built-in strategy on the same synthetic market.

Uses ``naive_backtest`` (next-open fills, flat commission in bps) because the real
execution handler arrives on Day 4 and the full analytics suite on Day 5, so the
statistics below are deliberately minimal.

    python scripts/demo_day3.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.portfolio import Portfolio
from quantlab.strategy import (
    SMACrossover,
    TimeSeriesMomentum,
    VolTargetAllocation,
    ZScoreMeanReversion,
)
from quantlab.testing import naive_backtest
from quantlab.utils import annualize_vol, simple_returns

SYMBOLS = ["EQ_A", "EQ_B", "BOND", "GOLD"]
CAPITAL = 1_000_000.0
COMMISSION_BPS = 5.0


def build_strategies():
    return {
        "SMA 20/100 (long only)": SMACrossover(SYMBOLS, 20, 100),
        "Momentum 126d (long/short)": TimeSeriesMomentum(SYMBOLS, 126, 0, rebalance_every=21),
        "Mean reversion z20 (L/S)": ZScoreMeanReversion(SYMBOLS, 20, 2.0, 0.5),
        "Vol target 8%": VolTargetAllocation(SYMBOLS, target_vol=0.08, lookback=60),
    }


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def main() -> None:
    market = generate_ohlcv_panel(
        SYMBOLS,
        periods=1500,
        annual_drift=[0.10, 0.10, 0.03, 0.04],
        annual_vol=[0.25, 0.22, 0.07, 0.15],
        correlation=0.3,
        regimes=DEFAULT_REGIMES,
        seed=42,
    )

    rows = []
    for name, strategy in build_strategies().items():
        pf = naive_backtest(market, strategy, Portfolio(CAPITAL), commission_bps=COMMISSION_BPS)
        curve = pf.equity_curve()
        rows.append(
            {
                "strategy": name,
                "total_return": curve.iloc[-1] / CAPITAL - 1.0,
                "ann_vol": annualize_vol(simple_returns(curve), 252),
                "max_drawdown": max_drawdown(curve),
                "trades": len(pf.trades_frame()),
                "commission": pf.total_commission,
            }
        )

    table = pd.DataFrame(rows).set_index("strategy")
    pd.options.display.float_format = "{:,.3f}".format
    print(f"Synthetic market, {len(SYMBOLS)} assets, {len(market[SYMBOLS[0]])} bars, "
          f"{COMMISSION_BPS:.0f} bps commission\n")
    print(table.to_string())
    print("\nNote: synthetic data, simplistic fills. Proper costs arrive on Day 4, "
          "full risk analytics on Day 5.")


if __name__ == "__main__":
    main()
