import numpy as np
import pandas as pd
import pytest

from quantlab.analytics import (
    equal_weight_returns,
    exposure_statistics,
    trade_statistics,
    turnover_statistics,
)
from quantlab.testing import frame_from_closes


def make_trades(pnls, commission=1.0, qty=10.0, price=100.0):
    return pd.DataFrame(
        {
            "quantity": [qty] * len(pnls),
            "price": [price] * len(pnls),
            "commission": [commission] * len(pnls),
            "realized_pnl": pnls,
        }
    )


def test_trade_statistics_hand_calculation():
    stats = trade_statistics(make_trades([0.0, 100.0, 0.0, -50.0, 200.0]))
    assert stats["n_closing_fills"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_win"] == pytest.approx(150.0)
    assert stats["avg_loss"] == pytest.approx(-50.0)
    assert stats["payoff_ratio"] == pytest.approx(3.0)
    assert stats["profit_factor"] == pytest.approx(300.0 / 50.0)
    assert stats["expectancy"] == pytest.approx(250.0 / 3)
    assert stats["largest_win"] == 200.0 and stats["largest_loss"] == -50.0
    assert stats["gross_realized_pnl"] == 250.0
    assert stats["total_commission"] == 5.0


def test_trade_statistics_edge_cases():
    only_wins = trade_statistics(make_trades([10.0, 20.0]))
    assert np.isinf(only_wins["profit_factor"]) and np.isnan(only_wins["payoff_ratio"])
    only_losses = trade_statistics(make_trades([-10.0]))
    assert only_losses["win_rate"] == 0.0 and np.isnan(only_losses["avg_win"])
    no_closes = trade_statistics(make_trades([0.0, 0.0]))
    assert no_closes["n_closing_fills"] == 0 and no_closes["gross_realized_pnl"] == 0.0
    assert no_closes["total_commission"] == 2.0
    empty = trade_statistics(pd.DataFrame(columns=["realized_pnl", "commission"]))
    assert empty["n_closing_fills"] == 0 and np.isnan(empty["win_rate"])


def test_turnover_statistics():
    history = pd.DataFrame({"equity": [100_000.0] * 252})
    trades = make_trades([0.0, 0.0], qty=500.0, price=100.0)  # 2 fills of 50,000
    stats = turnover_statistics(history, trades, 252)
    assert stats["traded_notional"] == 100_000.0
    assert stats["annual_turnover"] == pytest.approx(0.5)  # one year, one-way = 50k / 100k
    assert stats["avg_trade_size"] == 50_000.0
    none = turnover_statistics(history, make_trades([]), 252)
    assert none["traded_notional"] == 0.0 and none["annual_turnover"] == 0.0
    assert np.isnan(turnover_statistics(history.iloc[:0], trades)["annual_turnover"])


def test_exposure_statistics():
    history = pd.DataFrame(
        {
            "equity": [100.0, 100.0, 100.0, 100.0],
            "gross_exposure": [0.0, 50.0, 150.0, 100.0],
            "net_exposure": [0.0, 50.0, -50.0, 100.0],
        }
    )
    stats = exposure_statistics(history)
    assert stats["avg_gross_leverage"] == pytest.approx(0.75)
    assert stats["max_gross_leverage"] == pytest.approx(1.5)
    assert stats["avg_net_exposure"] == pytest.approx(0.25)
    assert stats["pct_time_invested"] == pytest.approx(0.75)
    assert np.isnan(exposure_statistics(history.iloc[:0])["avg_gross_leverage"])


def test_equal_weight_benchmark_rebalanced_and_buy_and_hold():
    market = {
        "A": frame_from_closes([100.0, 110.0, 121.0]),
        "B": frame_from_closes([100.0, 100.0, 100.0]),
    }
    daily = equal_weight_returns(market, rebalance=True)
    assert daily.tolist() == pytest.approx([0.0, 0.05, 0.05])
    hold = equal_weight_returns(market, rebalance=False)
    assert hold.tolist() == pytest.approx([0.0, 0.05, 1.105 / 1.05 - 1])
    assert daily.name == "benchmark"
