"""Simple benchmarks built from the same market data a strategy trades."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantlab.data.loaders import to_close_panel


def equal_weight_returns(
    market: Mapping[str, pd.DataFrame], *, rebalance: bool = True
) -> pd.Series:
    """Per-bar returns of an equal-weight portfolio of every instrument.

    ``rebalance=True`` resets to equal weights every bar (the average of the
    instruments' returns). ``rebalance=False`` buys equal amounts on day one
    and never trades again (buy and hold).

    Missing bars are carried forward; the first bar's return is 0 so the series
    lines up with a strategy that starts flat.
    """
    closes = to_close_panel(market).ffill()
    if rebalance:
        returns = closes.pct_change().mean(axis=1)
    else:
        index_level = (closes / closes.bfill().iloc[0]).mean(axis=1)
        returns = index_level.pct_change()
    returns.iloc[0] = 0.0
    return returns.rename("benchmark")
