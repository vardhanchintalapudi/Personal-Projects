import numpy as np
import pandas as pd
import pytest
from helpers import Scripted

from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel, validate_ohlcv
from quantlab.events import SignalEvent
from quantlab.portfolio import Portfolio
from quantlab.strategy import (
    SMACrossover,
    Strategy,
    TimeSeriesMomentum,
    VolTargetAllocation,
    ZScoreMeanReversion,
    available_strategies,
    create_strategy,
    get_strategy,
    register_strategy,
)
from quantlab.testing import collect_signals, frame_from_closes, naive_backtest

SYMBOLS = ["AAA", "BBB", "CCC"]


# ------------------------------------------------------------------ helpers


def test_frame_from_closes_is_valid_and_continuous():
    frame = frame_from_closes([100, 101, 99, 102])
    assert validate_ohlcv(frame).ok
    assert frame["open"].tolist() == [100, 100, 101, 99]
    with pytest.raises(ValueError):
        frame_from_closes([100.0])


def test_collect_signals_returns_signal_events(market):
    signals = collect_signals(SMACrossover(SYMBOLS, 5, 20), market)
    assert signals and all(isinstance(s, SignalEvent) for s in signals)
    assert {s.strategy_id for s in signals} == {"SMACrossover"}


# ---------------------------------------------------------- naive backtester


def test_naive_backtest_fills_at_next_open_with_commission():
    closes = [100, 101, 102, 103, 104, 105, 106, 107]
    market = {"A": frame_from_closes(closes)}
    strat = Scripted(["A"], {3: {"A": 1.0}})  # decide on bar 3 (index 2, close 102)
    pf = naive_backtest(market, strat, Portfolio(10_000.0), commission_bps=10.0)

    trades = pf.trades_frame()
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["timestamp"] == market["A"].index[3]  # next bar
    assert row["price"] == 102.0  # next open == previous close for this frame
    assert row["quantity"] == float(int(10_000 / 102.0))
    assert row["commission"] == pytest.approx(row["quantity"] * 102.0 * 10 / 1e4)
    assert pf.total_commission == pytest.approx(row["commission"])


def test_orders_wait_for_the_next_bar_of_an_instrument_that_skipped():
    a = frame_from_closes(np.linspace(100, 110, 10))
    b = a.drop(a.index[5])  # B has no bar at index 5
    strat = Scripted(["B"], {5: {"B": 0.5}})  # decide on bar 5 -> index 4
    pf = naive_backtest({"A": a, "B": b}, strat, Portfolio(10_000.0))
    trade = pf.trades_frame().iloc[0]
    assert trade["timestamp"] == a.index[6]  # filled at the next B bar, not skipped


def test_naive_backtest_rejects_negative_commission(market):
    with pytest.raises(ValueError):
        naive_backtest(market, Scripted(["AAA"], {}), commission_bps=-1)


# ------------------------------------------------- strategies through portfolio


def test_trend_following_makes_money_in_a_clean_uptrend():
    market = generate_ohlcv_panel(["A"], periods=300, annual_drift=0.6, annual_vol=0.05, seed=1)
    pf = naive_backtest(market, SMACrossover(["A"], 10, 40, weight=1.0), Portfolio(100_000.0))
    assert pf.equity > 100_000.0 * 1.2
    assert pf.position_quantity("A") > 0
    assert pf.history_frame()["cash"].min() >= -1e-6  # long-only, fully funded


def test_momentum_shorts_profit_in_a_clean_downtrend():
    market = generate_ohlcv_panel(["A"], periods=300, annual_drift=-0.6, annual_vol=0.05, seed=2)
    strat = TimeSeriesMomentum(["A"], 40, rebalance_every=10, weight=1.0)
    pf = naive_backtest(market, strat, Portfolio(100_000.0))
    assert pf.position_quantity("A") < 0
    assert pf.equity > 100_000.0 * 1.1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SMACrossover(SYMBOLS, 5, 20, allow_short=True),
        lambda: TimeSeriesMomentum(SYMBOLS, 30, 5, rebalance_every=7),
        lambda: ZScoreMeanReversion(SYMBOLS, 15, 1.5, 0.3),
        lambda: VolTargetAllocation(SYMBOLS, target_vol=0.5, lookback=30, rebalance_every=3),
    ],
    ids=["sma", "tsmom", "mean_reversion", "vol_target"],
)
def test_portfolio_targets_match_strategy_weights_so_nothing_is_clipped(factory):
    market = generate_ohlcv_panel(
        SYMBOLS, periods=450, regimes=DEFAULT_REGIMES, correlation=0.4, seed=21
    )
    strat = factory()
    pf = naive_backtest(market, strat, Portfolio(250_000.0))

    wanted = strat.current_weights
    assert wanted, "strategy should have traded"
    assert pf.target_weights == pytest.approx(wanted, abs=1e-9)
    assert len(pf.trades_frame()) > 0
    assert np.isfinite(pf.equity_curve()).all()


def test_backtests_are_deterministic():
    def run():
        market = generate_ohlcv_panel(SYMBOLS, periods=300, seed=5)
        strat = TimeSeriesMomentum(SYMBOLS, 30, rebalance_every=5)
        return naive_backtest(market, strat, Portfolio(100_000.0)).equity_curve()

    pd.testing.assert_series_equal(run(), run())


# ------------------------------------------------------------------ registry


def test_registry_lists_and_creates_builtin_strategies():
    assert set(available_strategies()) >= {"sma_crossover", "tsmom", "mean_reversion", "vol_target"}
    strat = create_strategy("TSMOM", ["A", "B"], lookback=60)
    assert isinstance(strat, TimeSeriesMomentum) and strat.lookback == 60
    assert get_strategy(" sma_crossover ") is SMACrossover
    assert strat.params["lookback"] == 60


def test_registry_errors():
    with pytest.raises(KeyError, match="available"):
        get_strategy("does_not_exist")

    with pytest.raises(ValueError, match="already registered"):

        @register_strategy("sma_crossover")
        class Impostor(Strategy):
            def compute_targets(self, event, feed):
                return {}

    with pytest.raises(ValueError):
        register_strategy("   ")
