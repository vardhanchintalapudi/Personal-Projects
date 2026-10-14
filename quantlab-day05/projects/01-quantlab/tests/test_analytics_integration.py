import numpy as np
import pandas as pd
import pytest

from quantlab.analytics import (
    drawdown_table,
    equal_weight_returns,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
)
from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel
from quantlab.engine import BacktestEngine
from quantlab.execution import ExecutionHandler
from quantlab.portfolio import Portfolio
from quantlab.strategy import SMACrossover, TimeSeriesMomentum

SYMBOLS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def market():
    return generate_ohlcv_panel(
        SYMBOLS, periods=600, regimes=DEFAULT_REGIMES, correlation=0.4, seed=33
    )


@pytest.fixture(scope="module")
def result(market):
    strategy = TimeSeriesMomentum(SYMBOLS, 40, rebalance_every=10)
    return BacktestEngine(
        market, strategy, Portfolio(500_000.0), ExecutionHandler.realistic()
    ).run()


def test_performance_agrees_with_engine_summary_and_direct_calls(result):
    perf = result.performance()
    assert perf["total_return"] == pytest.approx(result.summary()["total_return"])
    assert perf["sharpe"] == pytest.approx(sharpe_ratio(result.returns))
    assert perf["max_drawdown"] == pytest.approx(max_drawdown(result.returns))
    assert perf["n_bars"] == len(result.history)
    direct = performance_summary(result.returns)
    pd.testing.assert_series_equal(perf, direct)


def test_max_drawdown_of_returns_matches_equity_curve(result):
    equity = pd.concat([pd.Series([result.initial_cash]), result.equity.reset_index(drop=True)])
    from_equity = float((equity / equity.cummax() - 1).min())
    assert max_drawdown(result.returns) == pytest.approx(from_equity)
    table = drawdown_table(result.returns, top=3)
    assert table["depth"].iloc[0] == pytest.approx(max_drawdown(result.returns))


def test_benchmark_relative_stats_on_a_real_run(market, result):
    benchmark = equal_weight_returns(market)
    assert len(benchmark) == len(result.returns)
    perf = result.performance(benchmark=benchmark)
    for key in ("beta", "alpha", "tracking_error", "information_ratio", "up_capture"):
        assert np.isfinite(perf[key])
    # a long/short trend strategy should not be a clone of the long-only benchmark
    assert abs(perf["beta"]) < 1.5


def test_trade_turnover_and_exposure_helpers(result):
    stats = result.trade_stats()
    assert stats["n_closing_fills"] > 0
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert stats["total_commission"] == pytest.approx(result.costs["commission"].sum())

    turnover = result.turnover()
    assert turnover["traded_notional"] > 0 and turnover["annual_turnover"] > 0

    exposure = result.exposure()
    assert 0.0 < exposure["pct_time_invested"] <= 1.0
    # Weights are set at signal time, so mark-to-market drift between rebalances (a rallying
    # short, a falling long) can lift gross leverage a little above the 1.0 cap.
    assert exposure["avg_gross_leverage"] <= 1.05
    assert exposure["max_gross_leverage"] <= 1.3


def test_buy_and_hold_like_strategy_tracks_the_buy_and_hold_benchmark(market):
    strategy = SMACrossover(SYMBOLS, 2, 3)  # almost always long, equal weight
    result = BacktestEngine(market, strategy, Portfolio(1_000_000.0)).run()
    benchmark = equal_weight_returns(market, rebalance=False)
    perf = result.performance(benchmark=benchmark)
    assert perf["correlation"] > 0.5
