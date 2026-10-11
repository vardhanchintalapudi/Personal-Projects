import numpy as np
import pytest

from quantlab.data import (
    DEFAULT_REGIMES,
    generate_ohlcv_panel,
    simulate_regime_states,
    validate_ohlcv,
)


def test_shape_columns_and_validity(market):
    assert set(market) == {"AAA", "BBB", "CCC"}
    for df in market.values():
        assert len(df) == 500
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert validate_ohlcv(df).ok


def test_reproducible_with_seed():
    a = generate_ohlcv_panel(["X", "Y"], periods=50, seed=1)
    b = generate_ohlcv_panel(["X", "Y"], periods=50, seed=1)
    c = generate_ohlcv_panel(["X", "Y"], periods=50, seed=2)
    assert a["X"].equals(b["X"])
    assert not a["X"].equals(c["X"])


def test_continuous_prices_no_gaps(aaa):
    np.testing.assert_allclose(aaa["open"].iloc[1:].to_numpy(), aaa["close"].iloc[:-1].to_numpy())


def test_initial_price_matches_s0():
    df = generate_ohlcv_panel(["X"], periods=10, s0=250.0, seed=3)["X"]
    assert df["open"].iloc[0] == pytest.approx(250.0)


def test_volatility_and_correlation_roughly_match_targets():
    panel = generate_ohlcv_panel(
        ["A", "B"], periods=4000, annual_vol=0.25, correlation=0.6, annual_drift=0.0, seed=11
    )
    rets = np.log(
        np.column_stack([panel["A"]["close"], panel["B"]["close"]])
    )
    rets = np.diff(rets, axis=0)
    ann_vol = rets.std(axis=0, ddof=1) * np.sqrt(252)
    corr = np.corrcoef(rets.T)[0, 1]
    assert ann_vol == pytest.approx([0.25, 0.25], abs=0.03)
    assert corr == pytest.approx(0.6, abs=0.05)


def test_regimes_raise_volatility_in_bear_state():
    panel = generate_ohlcv_panel(
        ["A"], periods=6000, regimes=DEFAULT_REGIMES, annual_drift=0.05, seed=5
    )
    close = panel["A"]["close"].to_numpy()
    rets = np.diff(np.log(close))
    # Re-simulate the same state path (rng stream starts with regime draw).
    rng = np.random.default_rng(5)
    from quantlab.data import DEFAULT_TRANSITION

    states = simulate_regime_states(6000, DEFAULT_TRANSITION, rng)[1:]
    bull_vol = rets[states == 0].std()
    bear_vol = rets[states == 1].std()
    assert bear_vol > 1.5 * bull_vol


def test_regime_chain_stationary_mix():
    from quantlab.data import DEFAULT_TRANSITION

    states = simulate_regime_states(50_000, DEFAULT_TRANSITION, np.random.default_rng(0))
    assert (states == 0).mean() == pytest.approx(0.75, abs=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbols": []},
        {"symbols": ["A", "A"]},
        {"symbols": ["A"], "periods": 1},
        {"symbols": ["A"], "annual_vol": -0.1},
        {"symbols": ["A"], "s0": 0.0},
        {"symbols": ["A", "B"], "correlation": 1.2},
        {"symbols": ["A", "B"], "annual_vol": [0.1, 0.2, 0.3]},
        {"symbols": ["A", "B"], "correlation": np.array([[1.0, 0.9], [0.1, 1.0]])},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        generate_ohlcv_panel(**kwargs)


def test_transition_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_regime_states(10, np.array([[0.5, 0.4], [0.5, 0.5]]), rng)
    with pytest.raises(ValueError, match="square"):
        simulate_regime_states(10, np.ones((2, 3)) / 3, rng)
