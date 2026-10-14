import numpy as np
import pandas as pd
import pytest
from helpers import Scripted, ohlc_frame

from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.engine import BacktestEngine, BacktestResult
from quantlab.execution import BpsCommission, ExecutionHandler, FixedBpsSlippage, SquareRootImpact
from quantlab.portfolio import Portfolio
from quantlab.strategy import SMACrossover, TimeSeriesMomentum, VolTargetAllocation
from quantlab.testing import frame_from_closes, naive_backtest

SYMBOLS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def regime_market():
    return generate_ohlcv_panel(
        SYMBOLS, periods=500, regimes=DEFAULT_REGIMES, correlation=0.4, seed=21
    )


def zero_cost_handler():
    return ExecutionHandler(max_participation=None)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SMACrossover(SYMBOLS, 5, 20, allow_short=True),
        lambda: TimeSeriesMomentum(SYMBOLS, 30, 5, rebalance_every=7),
        lambda: VolTargetAllocation(SYMBOLS, target_vol=0.2, lookback=30, rebalance_every=3),
    ],
    ids=["sma", "tsmom", "vol_target"],
)
def test_zero_cost_engine_reproduces_the_naive_backtest(factory, regime_market):
    """The queue-driven engine and the Day 3 stand-in must agree to the cent."""
    naive = naive_backtest(regime_market, factory(), Portfolio(100_000.0))
    result = BacktestEngine(
        regime_market, factory(), Portfolio(100_000.0), zero_cost_handler()
    ).run()
    pd.testing.assert_series_equal(result.equity, naive.equity_curve(), check_names=False)
    assert len(result.trades) == len(naive.trades_frame()) > 0


def test_cost_accounting_identity():
    """With identical share counts, zero-cost minus costly equity equals the recorded costs."""
    closes = np.linspace(100, 130, 40)
    market = {"A": frame_from_closes(closes, volume=50_000.0)}
    script = {5: {"A": 0.8}, 25: {"A": 0.0}}  # enter, later exit

    free = BacktestEngine(
        market, Scripted(["A"], script), Portfolio(100_000.0), zero_cost_handler()
    )
    free_result = free.run()

    costly_handler = ExecutionHandler(
        BpsCommission(5),
        FixedBpsSlippage(8),
        SquareRootImpact(coefficient=0.5),
        max_participation=None,
    )
    costly = BacktestEngine(market, Scripted(["A"], script), Portfolio(100_000.0), costly_handler)
    costly_result = costly.run()

    assert costly_result.total_cost > 0
    assert set(costly_result.trades["quantity"]) == set(free_result.trades["quantity"])
    gap = free_result.equity.iloc[-1] - costly_result.equity.iloc[-1]
    assert gap == pytest.approx(costly_result.total_cost, rel=1e-9)
    summary = costly_result.summary()
    assert summary["total_cost"] == pytest.approx(
        summary["total_commission"] + summary["total_slippage"] + summary["total_impact"]
    )
    assert min(summary["total_slippage"], summary["total_commission"]) > 0
    assert summary["total_impact"] > 0


def test_result_tables_and_properties(regime_market):
    result = BacktestEngine(
        regime_market, SMACrossover(SYMBOLS, 5, 20), execution=ExecutionHandler.realistic()
    ).run()
    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "SMACrossover"
    assert result.params["fast"] == 5
    assert list(result.signals.columns) == [
        "timestamp",
        "symbol",
        "target_weight",
        "strategy_id",
        "reason",
    ]
    assert len(result.signals) >= len(result.orders) >= len(result.trades) > 0
    assert len(result.history) == len(regime_market["AAA"])
    assert result.returns.iloc[0] == pytest.approx(0.0)  # nothing traded on bar one
    compounded = result.initial_cash * (1 + result.returns).prod()
    assert result.equity.iloc[-1] == pytest.approx(compounded)
    assert (result.costs["total_cost"] >= 0).all()
    assert result.summary()["n_fills"] == len(result.trades)


def test_equity_on_a_fill_bar_already_includes_the_fill():
    market = {"A": frame_from_closes([100, 100, 100, 110, 110, 110])}
    script = {2: {"A": 1.0}}  # signal on bar 2 (index 1) -> fills at open of index 2
    result = BacktestEngine(
        market, Scripted(["A"], script), Portfolio(10_000.0), zero_cost_handler()
    ).run()
    trade = result.trades.iloc[0]
    assert trade["timestamp"] == market["A"].index[2]
    # bought 100 shares at 100 on index 2; the jump to 110 on index 3 is the only P&L
    assert result.equity.iloc[2] == pytest.approx(10_000.0)
    assert result.equity.iloc[3] == pytest.approx(11_000.0)


def test_partial_fills_expiry_and_pending_orders_are_reconciled():
    # volume 1000/bar, 10% cap -> 100 shares/bar; wants 500, order lives 2 bars
    rows = [(100.0, 100.5, 99.5, 100.0, 1_000.0)] * 12
    market = {"A": ohlc_frame(rows)}
    handler = ExecutionHandler(max_participation=0.10, max_order_age=2)
    portfolio = Portfolio(50_000.0)
    result = BacktestEngine(market, Scripted(["A"], {3: {"A": 1.0}}), portfolio, handler).run()

    assert result.n_expired == 1
    assert portfolio.position_quantity("A") == 200.0  # two partial fills of 100
    assert portfolio.open_orders == {}  # expiry released the pending remainder
    assert result.open_orders_at_end == {}
    assert result.trades["quantity"].tolist() == [100.0, 100.0]


def test_orders_still_working_at_the_end_are_reported():
    rows = [(100.0, 100.5, 99.5, 100.0, 1_000.0)] * 6
    market = {"A": ohlc_frame(rows)}
    result = BacktestEngine(
        market,
        Scripted(["A"], {4: {"A": 1.0}}),
        Portfolio(1_000_000.0),
        ExecutionHandler(max_participation=0.10),
    ).run()
    assert result.open_orders_at_end  # 100/bar cannot absorb a ~10,000 share order in 2 bars


def test_engine_runs_only_once(regime_market):
    engine = BacktestEngine(regime_market, SMACrossover(SYMBOLS, 5, 20))
    engine.run()
    with pytest.raises(RuntimeError, match="already run"):
        engine.run()


def test_realistic_costs_reduce_performance_relative_to_free_trading(regime_market):
    def run(handler):
        strat = TimeSeriesMomentum(SYMBOLS, 20, rebalance_every=5)
        return BacktestEngine(regime_market, strat, Portfolio(1_000_000.0), handler).run()

    free = run(zero_cost_handler())
    realistic = run(ExecutionHandler.realistic(max_participation=None))
    assert realistic.total_cost > 0
    assert realistic.equity.iloc[-1] < free.equity.iloc[-1]


def test_engine_is_deterministic(regime_market):
    def run():
        return BacktestEngine(
            regime_market,
            SMACrossover(SYMBOLS, 5, 20, allow_short=True),
            execution=ExecutionHandler.realistic(),
        ).run()

    a, b = run(), run()
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.costs, b.costs)
