import pandas as pd
import pytest

from quantlab.events import Bar, FillEvent, MarketEvent, SignalEvent
from quantlab.portfolio import Portfolio, Position

TS = pd.Timestamp("2024-01-02")


def market_event(prices: dict[str, float], ts: pd.Timestamp = TS) -> MarketEvent:
    return MarketEvent(ts, {s: Bar(p, p, p, p, 1_000) for s, p in prices.items()})


def fill(symbol, qty, price, commission=0.0, order_id=None, ts=TS):
    return FillEvent(ts, symbol, qty, price, commission=commission, order_id=order_id)


# --------------------------------------------------------------------- Position


def test_position_open_add_and_average_cost():
    pos = Position("A")
    assert pos.apply_fill(10, 100.0) == 0.0
    assert pos.apply_fill(10, 110.0) == 0.0
    assert pos.quantity == 20
    assert pos.avg_price == pytest.approx(105.0)
    assert pos.unrealized_pnl(115.0) == pytest.approx(200.0)


def test_position_partial_and_full_close_realise_pnl():
    pos = Position("A")
    pos.apply_fill(10, 100.0)
    assert pos.apply_fill(-4, 110.0) == pytest.approx(40.0)
    assert pos.quantity == 6 and pos.avg_price == 100.0  # basis unchanged
    assert pos.apply_fill(-6, 90.0) == pytest.approx(-60.0)
    assert pos.is_flat and pos.avg_price == 0.0
    assert pos.realized_pnl == pytest.approx(-20.0)
    assert pos.unrealized_pnl(123.0) == 0.0


def test_position_flip_long_to_short_and_back():
    pos = Position("A")
    pos.apply_fill(10, 100.0)
    realized = pos.apply_fill(-15, 120.0)  # close 10 at +20, open 5 short at 120
    assert realized == pytest.approx(200.0)
    assert pos.quantity == -5 and pos.avg_price == 120.0
    realized = pos.apply_fill(8, 100.0)  # close 5 short at +20, open 3 long at 100
    assert realized == pytest.approx(100.0)
    assert pos.quantity == 3 and pos.avg_price == 100.0


def test_short_position_profits_when_price_falls():
    pos = Position("A")
    pos.apply_fill(-10, 50.0)
    assert pos.unrealized_pnl(40.0) == pytest.approx(100.0)
    assert pos.apply_fill(10, 40.0) == pytest.approx(100.0)


def test_negligible_fill_is_ignored():
    pos = Position("A")
    assert pos.apply_fill(1e-12, 10.0) == 0.0 and pos.is_flat


# -------------------------------------------------------------------- Portfolio


def test_cash_and_equity_accounting_for_long_trade():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0}))
    pf.on_fill(fill("A", 10, 100.0, commission=5.0))
    assert pf.cash == pytest.approx(10_000 - 1_000 - 5)
    assert pf.equity == pytest.approx(10_000 - 5)  # only commission lost at same price
    pf.on_market(market_event({"A": 110.0}, TS + pd.Timedelta(days=1)))
    assert pf.equity == pytest.approx(10_000 - 5 + 100)
    assert pf.unrealized_pnl == pytest.approx(100.0)
    assert pf.total_commission == 5.0


def test_short_sale_adds_cash_and_tracks_liability():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 50.0}))
    pf.on_fill(fill("A", -20, 50.0))
    assert pf.cash == pytest.approx(11_000.0)
    assert pf.market_value == pytest.approx(-1_000.0)
    assert pf.equity == pytest.approx(10_000.0)
    pf.on_market(market_event({"A": 40.0}, TS + pd.Timedelta(days=1)))
    assert pf.equity == pytest.approx(10_200.0)
    assert pf.gross_exposure == pytest.approx(800.0)
    assert pf.net_exposure == pytest.approx(-800.0)


def test_signal_sizing_rounds_toward_zero():
    pf = Portfolio(100_000.0)
    pf.on_market(market_event({"A": 33.33}))
    order = pf.on_signal(SignalEvent(TS, "A", 1.0, strategy_id="s"))
    assert order.quantity == 3000.0  # trunc(100000 / 33.33) = 3000.3 -> 3000
    assert order.order_id == 1 and order.strategy_id == "s"

    short = Portfolio(100_000.0)
    short.on_market(market_event({"A": 33.33}))
    assert short.on_signal(SignalEvent(TS, "A", -1.0)).quantity == -3000.0


def test_fractional_sizing_when_allowed():
    pf = Portfolio(1_000.0, allow_fractional=True)
    pf.on_market(market_event({"A": 300.0}))
    assert pf.on_signal(SignalEvent(TS, "A", 1.0)).quantity == pytest.approx(1000 / 300)


def test_signal_without_price_raises():
    with pytest.raises(KeyError, match="no market price"):
        Portfolio().on_signal(SignalEvent(TS, "A", 0.5))


def test_position_weight_and_gross_leverage_caps():
    pf = Portfolio(100_000.0, max_position_weight=0.5, max_gross_leverage=0.8)
    pf.on_market(market_event({"A": 100.0, "B": 100.0, "C": 100.0}))
    a = pf.on_signal(SignalEvent(TS, "A", 1.0))  # clipped to 0.5
    b = pf.on_signal(SignalEvent(TS, "B", 0.5))  # only 0.3 capacity left
    c = pf.on_signal(SignalEvent(TS, "C", -0.5))  # no capacity left
    assert a.quantity == 500 and b.quantity == 300 and c is None
    assert pf.target_weights == pytest.approx({"A": 0.5, "B": 0.3, "C": 0.0})
    assert sum(abs(w) for w in pf.target_weights.values()) <= 0.8 + 1e-12


def test_pending_orders_prevent_duplicate_orders():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0}))
    first = pf.on_signal(SignalEvent(TS, "A", 0.5))
    assert first.quantity == 50 and pf.pending_quantity("A") == 50
    assert pf.on_signal(SignalEvent(TS, "A", 0.5)) is None  # already working
    top_up = pf.on_signal(SignalEvent(TS, "A", 0.8))
    assert top_up.quantity == 30
    assert pf.open_orders == {1: 50, 2: 30}


def test_fills_clear_working_orders_including_partials():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0}))
    order = pf.on_signal(SignalEvent(TS, "A", 1.0))
    pf.on_fill(fill("A", 40, 100.0, order_id=order.order_id))
    assert pf.open_orders == {order.order_id: 60}
    pf.on_fill(fill("A", 60, 100.0, order_id=order.order_id))
    assert pf.open_orders == {} and pf.position_quantity("A") == 100


def test_cancel_order_frees_capacity():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0}))
    order = pf.on_signal(SignalEvent(TS, "A", 0.5))
    assert pf.cancel_order(order.order_id) is True
    assert pf.cancel_order(order.order_id) is False
    assert pf.on_signal(SignalEvent(TS, "A", 0.5)).quantity == 50


def test_exit_signal_closes_whole_position():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0}))
    pf.on_fill(fill("A", 50, 100.0))
    order = pf.on_signal(SignalEvent(TS, "A", 0.0))
    assert order.quantity == -50


def test_min_trade_fraction_skips_small_rebalances_but_not_exits():
    pf = Portfolio(100_000.0, min_trade_fraction=0.05)
    pf.on_market(market_event({"A": 100.0}))
    pf.on_fill(fill("A", 500, 100.0))  # 50% weight
    assert pf.on_signal(SignalEvent(TS, "A", 0.52)) is None  # 2% change < 5%
    assert pf.on_signal(SignalEvent(TS, "A", 0.60)).quantity == 100  # 10% change
    pf2 = Portfolio(100_000.0, min_trade_fraction=0.5)
    pf2.on_market(market_event({"A": 100.0}))
    pf2.on_fill(fill("A", 10, 100.0))  # 1% weight, exit is a tiny trade
    assert pf2.on_signal(SignalEvent(TS, "A", 0.0)).quantity == -10


def test_non_positive_equity_blocks_new_orders():
    pf = Portfolio(1_000.0)
    pf.on_market(market_event({"A": 100.0}))
    pf.on_fill(fill("A", -20, 100.0))  # short 2000 notional
    pf.on_market(market_event({"A": 200.0}, TS + pd.Timedelta(days=1)))  # equity < 0
    assert pf.equity < 0
    assert pf.on_signal(SignalEvent(TS, "A", 0.5)) is None


def test_weights_and_summary():
    pf = Portfolio(10_000.0)
    pf.on_market(market_event({"A": 100.0, "B": 50.0}))
    pf.on_fill(fill("A", 50, 100.0, commission=1.0))
    pf.on_fill(fill("B", -40, 50.0, commission=1.0))
    w = pf.weights()
    assert w["A"] == pytest.approx(5_000 / pf.equity)
    assert w["B"] == pytest.approx(-2_000 / pf.equity)
    assert pf.positions.keys() == {"A", "B"}
    summary = pf.summary()
    assert summary["total_commission"] == 2.0 and summary["n_trades"] == 2.0
    assert summary["final_equity"] == pytest.approx(10_000 - 2.0)


def test_history_and_trade_frames():
    pf = Portfolio(10_000.0)
    assert pf.history_frame().empty and pf.equity_curve().empty
    pf.on_market(market_event({"A": 100.0}))
    pf.on_fill(fill("A", 10, 100.0, order_id=7))
    pf.on_fill(fill("A", -10, 110.0, ts=TS + pd.Timedelta(days=1)))
    pf.on_market(market_event({"A": 110.0}, TS + pd.Timedelta(days=1)))

    hist = pf.history_frame()
    assert list(hist.columns) == [
        "cash",
        "market_value",
        "equity",
        "gross_exposure",
        "net_exposure",
    ]
    assert hist.index.name == "date" and len(hist) == 2
    assert hist["equity"].iloc[-1] == pytest.approx(10_100.0)

    trades = pf.trades_frame()
    assert trades["realized_pnl"].tolist() == [0.0, pytest.approx(100.0)]
    assert trades["order_id"].iloc[0] == 7
    assert pf.realized_pnl == pytest.approx(100.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_cash": 0},
        {"max_position_weight": 0},
        {"max_gross_leverage": -1},
        {"min_trade_fraction": -0.1},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        Portfolio(**kwargs)
