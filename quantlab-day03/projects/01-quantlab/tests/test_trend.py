import numpy as np
import pandas as pd
import pytest
from helpers import weight_path

from quantlab.strategy import SMACrossover, TimeSeriesMomentum
from quantlab.testing import collect_signals, frame_from_closes


def _market(**series):
    return {name: frame_from_closes(closes) for name, closes in series.items()}


# ----------------------------------------------------------------------- SMA


def test_sma_goes_long_in_uptrend_exactly_once():
    market = _market(A=np.linspace(100, 140, 60))
    signals = collect_signals(SMACrossover(["A"], fast=3, slow=5, weight=1.0), market)
    assert [s.target_weight for s in signals] == [1.0]
    assert signals[0].timestamp == market["A"].index[4]  # first bar with a full slow window


def test_sma_stays_flat_in_downtrend_unless_shorting():
    market = _market(A=np.linspace(140, 100, 60))
    assert collect_signals(SMACrossover(["A"], 3, 5, weight=1.0), market) == []
    shorts = collect_signals(SMACrossover(["A"], 3, 5, weight=1.0, allow_short=True), market)
    assert [s.target_weight for s in shorts] == [-1.0]


def test_sma_enters_and_exits_around_a_turn():
    path = np.concatenate([np.linspace(100, 120, 20), np.linspace(120, 100, 20)])
    market = _market(A=path)
    signals = collect_signals(SMACrossover(["A"], 2, 5, weight=1.0), market)
    assert [s.target_weight for s in signals] == [1.0, 0.0]
    exit_pos = market["A"].index.get_loc(signals[1].timestamp)
    assert exit_pos > 20  # exits only after the peak


def test_sma_default_weight_splits_equally():
    market = _market(A=np.linspace(100, 140, 40), B=np.linspace(50, 80, 40))
    signals = collect_signals(SMACrossover(["A", "B"], 3, 5), market)
    assert {s.symbol: s.target_weight for s in signals} == {"A": 0.5, "B": 0.5}


@pytest.mark.parametrize("allow_short", [False, True])
def test_sma_matches_pandas_reference(market, allow_short):
    symbols = ["AAA", "BBB", "CCC"]
    fast, slow = 8, 30
    strat = SMACrossover(symbols, fast, slow, allow_short=allow_short)
    signals = collect_signals(strat, market)
    for symbol in symbols:
        close = market[symbol]["close"]
        f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
        expected = pd.Series(0.0, index=close.index)
        expected[f > s] = 1 / 3
        if allow_short:
            expected[f < s] = -1 / 3
        expected[s.isna()] = 0.0
        pd.testing.assert_series_equal(
            weight_path(signals, symbol, close.index), expected, check_names=False
        )


@pytest.mark.parametrize(
    "kwargs", [{"fast": 0}, {"fast": 5, "slow": 5}, {"fast": 10, "slow": 5}, {"weight": 0.0}]
)
def test_sma_validation(kwargs):
    with pytest.raises(ValueError):
        SMACrossover(["A"], **kwargs)


# --------------------------------------------------------------------- TSMOM


def test_tsmom_follows_sign_of_trailing_return():
    up = _market(A=np.linspace(100, 160, 80))
    down = _market(A=np.linspace(160, 100, 80))
    kwargs = {"lookback": 20, "rebalance_every": 5, "weight": 1.0}
    assert collect_signals(TimeSeriesMomentum(["A"], **kwargs), up)[0].target_weight == 1.0
    assert collect_signals(TimeSeriesMomentum(["A"], **kwargs), down)[0].target_weight == -1.0
    assert collect_signals(TimeSeriesMomentum(["A"], allow_short=False, **kwargs), down) == []


def test_tsmom_threshold_keeps_weak_trends_flat():
    drifting = _market(A=100 * np.exp(np.linspace(0, 0.01, 60)))  # +1% over the run
    kwargs = {"lookback": 20, "rebalance_every": 5, "weight": 1.0}
    assert collect_signals(TimeSeriesMomentum(["A"], threshold=0.05, **kwargs), drifting) == []
    assert collect_signals(TimeSeriesMomentum(["A"], threshold=0.0, **kwargs), drifting)


def test_tsmom_skip_ignores_recent_reversal():
    path = np.concatenate([np.linspace(100, 140, 41), [138, 130, 120, 110, 100]])
    market = _market(A=path)

    def final_weight(skip):
        strat = TimeSeriesMomentum(["A"], 20, skip, rebalance_every=1, weight=1.0)
        collect_signals(strat, market)
        return strat.current_weights["A"]

    assert final_weight(skip=0) == -1.0  # last 20 bars fell
    assert final_weight(skip=5) == 1.0  # same window shifted back 5 bars is still rising


def test_tsmom_rebalances_only_on_schedule():
    path = np.concatenate([np.linspace(100, 160, 60), np.linspace(160, 90, 60)])
    market = _market(A=path)
    lookback, every = 20, 10
    signals = collect_signals(
        TimeSeriesMomentum(["A"], lookback, rebalance_every=every, weight=1.0), market
    )
    positions = [market["A"].index.get_loc(s.timestamp) for s in signals]
    assert positions and all((p - lookback) % every == 0 for p in positions)
    assert [s.target_weight for s in signals] == [1.0, -1.0]


def test_tsmom_matches_pandas_reference(market):
    symbols = ["AAA", "BBB", "CCC"]
    lookback, skip, every = 40, 5, 10
    strat = TimeSeriesMomentum(symbols, lookback, skip, rebalance_every=every)
    signals = collect_signals(strat, market)
    for symbol in symbols:
        close = market[symbol]["close"]
        trailing = close.shift(skip) / close.shift(lookback + skip) - 1.0
        eval_positions = range(lookback + skip, len(close), every)
        expected = pd.Series(np.nan, index=close.index)
        for pos in eval_positions:
            expected.iloc[pos] = np.sign(trailing.iloc[pos]) / 3
        expected = expected.ffill().fillna(0.0)
        pd.testing.assert_series_equal(
            weight_path(signals, symbol, close.index), expected, check_names=False
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"lookback": 0}, {"skip": -1}, {"rebalance_every": 0}, {"threshold": -0.1}, {"weight": -1}],
)
def test_tsmom_validation(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesMomentum(["A"], **kwargs)
