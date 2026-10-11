"""Truncation invariance: a strategy that never peeks produces the same signals
on a truncated history as on the full history, up to the truncation point.

If any strategy used information from after the current bar (for example by
reading the feed's full timeline or normalising by whole-sample statistics),
shortening the data would change its earlier decisions and this test would fail.
"""

import pytest

from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.strategy import (
    SMACrossover,
    TimeSeriesMomentum,
    VolTargetAllocation,
    ZScoreMeanReversion,
)
from quantlab.testing import collect_signals

SYMBOLS = ["AAA", "BBB", "CCC"]

FACTORIES = {
    "sma": lambda: SMACrossover(SYMBOLS, 5, 20, allow_short=True),
    "tsmom": lambda: TimeSeriesMomentum(SYMBOLS, 30, 5, rebalance_every=7),
    "mean_reversion": lambda: ZScoreMeanReversion(SYMBOLS, 15, 1.5, 0.3),
    "vol_target": lambda: VolTargetAllocation(
        SYMBOLS, target_vol=0.08, lookback=30, rebalance_every=3, rebalance_band=0.0
    ),
}


@pytest.fixture(scope="module")
def full_market():
    return generate_ohlcv_panel(
        SYMBOLS, periods=450, regimes=DEFAULT_REGIMES, correlation=0.4, seed=21
    )


def _key(signal):
    return (signal.timestamp, signal.symbol, signal.target_weight, signal.reason)


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("cut", [120, 237, 400])
def test_signals_up_to_cut_do_not_depend_on_later_data(name, cut, full_market):
    full = collect_signals(FACTORIES[name](), full_market)
    assert full, "scenario should produce signals"

    truncated_market = {s: df.iloc[:cut] for s, df in full_market.items()}
    truncated = collect_signals(FACTORIES[name](), truncated_market)

    cutoff = truncated_market["AAA"].index[-1]
    expected = [_key(s) for s in full if s.timestamp <= cutoff]
    assert [_key(s) for s in truncated] == expected
