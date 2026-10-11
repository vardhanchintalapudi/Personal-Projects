import numpy as np
import pandas as pd
import pytest
from helpers import weight_path

from quantlab.strategy import ZScoreMeanReversion
from quantlab.testing import collect_signals, frame_from_closes


def _wiggle(n, level=100.0, amplitude=0.1):
    return level + amplitude * (-1.0) ** np.arange(n)


def test_long_entry_on_sharp_dip_and_exit_on_recovery():
    path = np.concatenate([_wiggle(30), [95.0], [100.0] * 5])
    market = {"A": frame_from_closes(path)}
    signals = collect_signals(ZScoreMeanReversion(["A"], 20, 2.0, 0.5, weight=1.0), market)
    assert [s.target_weight for s in signals] == [1.0, 0.0]
    assert signals[0].timestamp == market["A"].index[30]  # the dip bar itself
    assert signals[0].reason.startswith("z-score -")
    assert signals[1].timestamp > signals[0].timestamp


def test_short_entry_on_spike_only_when_allowed():
    path = np.concatenate([_wiggle(30), [105.0], [100.0] * 5])
    market = {"A": frame_from_closes(path)}
    shorts = collect_signals(ZScoreMeanReversion(["A"], 20, 2.0, 0.5, weight=1.0), market)
    assert [s.target_weight for s in shorts] == [-1.0, 0.0]
    no_shorts = ZScoreMeanReversion(["A"], 20, 2.0, 0.5, weight=1.0, allow_short=False)
    assert collect_signals(no_shorts, market) == []


def test_constant_prices_have_undefined_zscore_and_no_signals():
    market = {"A": frame_from_closes([100.0] * 60)}
    assert collect_signals(ZScoreMeanReversion(["A"]), market) == []


def _reference_weights(close, lookback, entry, exit_, allow_short, weight):
    state, out = 0, []
    for t in range(len(close)):
        if t + 1 >= lookback:
            window = close.iloc[t + 1 - lookback : t + 1]
            std = window.std(ddof=1)
            if std > 1e-12:
                z = (window.iloc[-1] - window.mean()) / std
                if state == 0:
                    if z <= -entry:
                        state = 1
                    elif z >= entry and allow_short:
                        state = -1
                elif state == 1 and z >= -exit_:
                    state = 0
                elif state == -1 and z <= exit_:
                    state = 0
        out.append(state * weight)
    return pd.Series(out, index=close.index)


@pytest.mark.parametrize("allow_short", [True, False])
def test_matches_reference_state_machine_on_synthetic_markets(market, allow_short):
    symbols = ["AAA", "BBB", "CCC"]
    strat = ZScoreMeanReversion(symbols, 15, 1.5, 0.3, allow_short=allow_short)
    signals = collect_signals(strat, market)
    assert len(signals) > 10  # the scenario actually exercises transitions
    for symbol in symbols:
        close = market[symbol]["close"]
        expected = _reference_weights(close, 15, 1.5, 0.3, allow_short, 1 / 3)
        pd.testing.assert_series_equal(
            weight_path(signals, symbol, close.index), expected, check_names=False
        )


def test_hysteresis_keeps_position_between_thresholds():
    # Dip to z << -2, then drift back to roughly z in (-2, -0.5): still long, no new signals.
    path = np.concatenate([_wiggle(30), [94.0], [96.0], [97.0], [97.5]])
    market = {"A": frame_from_closes(path)}
    strat = ZScoreMeanReversion(["A"], 30, 2.0, 0.1, weight=1.0)
    signals = collect_signals(strat, market)
    assert [s.target_weight for s in signals] == [1.0]  # entered, never exited


@pytest.mark.parametrize(
    "kwargs",
    [{"lookback": 2}, {"entry_z": 0.5, "exit_z": 0.5}, {"exit_z": -0.1}, {"weight": 0}],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        ZScoreMeanReversion(["A"], **kwargs)
