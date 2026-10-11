import numpy as np
import pandas as pd
import pytest
from helpers import weight_path

from quantlab.data import generate_ohlcv_panel
from quantlab.strategy import VolTargetAllocation
from quantlab.testing import collect_signals


def _returns(vols_daily, n=5_000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, vols_daily, size=(n, len(vols_daily)))


def test_inverse_vol_weights_and_hits_target_when_uncapped():
    returns = _returns([0.01, 0.03])
    strat = VolTargetAllocation(["A", "B"], target_vol=0.10, max_leverage=10.0)
    w = strat.allocate(returns)

    cov = np.cov(returns, rowvar=False) * 252
    assert np.sqrt(w @ cov @ w) == pytest.approx(0.10, abs=1e-9)
    assert w[0] > w[1]  # calmer asset gets more capital
    assert w[0] / w[1] == pytest.approx(3.0, rel=0.05)


def test_leverage_cap_binds_when_target_is_high():
    strat = VolTargetAllocation(["A", "B"], target_vol=0.50, max_leverage=0.6)
    w = strat.allocate(_returns([0.01, 0.01]))
    assert w.sum() == pytest.approx(0.6)


def test_low_target_leaves_cash():
    strat = VolTargetAllocation(["A", "B"], target_vol=0.02, max_leverage=1.0)
    assert strat.allocate(_returns([0.01, 0.02])).sum() < 0.5


def test_single_asset_scales_by_its_own_vol():
    returns = _returns([0.02], n=3_000)
    strat = VolTargetAllocation(["A"], target_vol=0.10, max_leverage=5.0)
    w = strat.allocate(returns)
    assert w[0] == pytest.approx(0.10 / (returns.std(ddof=1) * np.sqrt(252)))


def test_unusable_estimates_return_none():
    strat = VolTargetAllocation(["A", "B"])
    zero_vol = np.column_stack([np.zeros(100), np.random.default_rng(0).normal(0, 0.01, 100)])
    assert strat.allocate(zero_vol) is None
    assert strat.allocate(np.zeros((1, 2))) is None
    assert strat.allocate(np.zeros(10)) is None


def test_first_signals_arrive_when_history_is_full_and_calm_assets_get_more():
    panel = generate_ohlcv_panel(
        ["LOW", "HIGH"], periods=300, annual_vol=[0.10, 0.30], correlation=0.2, seed=3
    )
    lookback, every = 40, 10
    strat = VolTargetAllocation(
        ["LOW", "HIGH"], target_vol=0.08, lookback=lookback, rebalance_every=every
    )
    signals = collect_signals(strat, panel)

    index = panel["LOW"].index
    first_ts = signals[0].timestamp
    assert index.get_loc(first_ts) == lookback
    first = {s.symbol: s.target_weight for s in signals if s.timestamp == first_ts}
    assert first["LOW"] > first["HIGH"] > 0
    for s in signals:
        assert (index.get_loc(s.timestamp) - lookback) % every == 0


def test_estimated_vol_matches_target_on_first_rebalance():
    panel = generate_ohlcv_panel(["A", "B", "C"], periods=200, annual_vol=0.2, seed=11)
    lookback = 50
    strat = VolTargetAllocation(["A", "B", "C"], target_vol=0.07, lookback=lookback, max_leverage=5)
    signals = collect_signals(strat, panel)
    t0 = signals[0].timestamp
    first = {s.symbol: s.target_weight for s in signals if s.timestamp == t0}
    weights = np.array([first[k] for k in "ABC"])
    closes = pd.concat({k: panel[k]["close"] for k in "ABC"}, axis=1)
    window = closes.loc[:t0].iloc[-(lookback + 1) :]
    cov = window.pct_change().dropna().cov().to_numpy() * 252
    assert np.sqrt(weights @ cov @ weights) == pytest.approx(0.07, abs=1e-9)


def test_weights_never_exceed_cap_and_are_always_a_complete_target(market):
    symbols = ["AAA", "BBB", "CCC"]
    strat = VolTargetAllocation(
        symbols, target_vol=0.6, lookback=30, rebalance_every=3, max_leverage=0.9
    )
    signals = collect_signals(strat, market)
    index = market["AAA"].index
    total = sum(weight_path(signals, s, index) for s in symbols)
    assert total.max() <= 0.9 + 1e-9
    assert total.max() == pytest.approx(0.9, abs=1e-6)  # cap actually binds here


def test_rebalance_band_suppresses_small_adjustments(market):
    symbols = ["AAA", "BBB", "CCC"]
    tight = collect_signals(
        VolTargetAllocation(symbols, lookback=30, rebalance_every=2, rebalance_band=0.0), market
    )
    loose = collect_signals(
        VolTargetAllocation(symbols, lookback=30, rebalance_every=2, rebalance_band=0.05), market
    )
    assert len(loose) < len(tight)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_vol": 0},
        {"max_leverage": -1},
        {"periods_per_year": 0},
        {"lookback": 4},
        {"rebalance_every": 0},
        {"rebalance_band": -0.1},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        VolTargetAllocation(["A", "B"], **kwargs)
