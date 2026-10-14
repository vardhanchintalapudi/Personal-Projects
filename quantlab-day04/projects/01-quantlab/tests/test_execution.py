import math

import pandas as pd
import pytest
from helpers import ohlc_frame

from quantlab.data import generate_ohlcv_panel
from quantlab.events import Bar, OrderEvent, OrderType
from quantlab.execution import (
    BarRangeSlippage,
    BpsCommission,
    ExecutionHandler,
    FixedBpsSlippage,
    NoImpact,
    NoSlippage,
    PerShareCommission,
    SquareRootImpact,
    ZeroCommission,
)
from quantlab.feed import HistoricalDataFeed
from quantlab.testing import frame_from_closes


def run_bar(handler, feed):
    """Advance the feed one bar and let the handler act on it."""
    event = feed.step()
    return handler.on_market(event, feed)


def order(qty, oid=1, symbol="A", **kwargs):
    ts = kwargs.pop("timestamp", pd.Timestamp("2021-01-01"))
    return OrderEvent(ts, symbol, qty, oid, **kwargs)


# ------------------------------------------------------------------ cost models


def test_commission_models():
    assert ZeroCommission()(100, 50.0) == 0.0
    assert BpsCommission(10)(-100, 50.0) == pytest.approx(5.0)  # 5000 notional * 10bp
    per_share = PerShareCommission(per_share=0.01, minimum=1.0, max_fraction_of_notional=0.01)
    assert per_share(500, 20.0) == pytest.approx(5.0)
    assert per_share(10, 20.0) == pytest.approx(1.0)  # minimum ticket
    assert per_share(10, 1.0) == pytest.approx(0.10)  # capped at 1% of $10 notional
    for bad in (lambda: BpsCommission(-1), lambda: PerShareCommission(per_share=-1)):
        with pytest.raises(ValueError):
            bad()


def test_slippage_models():
    bar = Bar(100.0, 104.0, 98.0, 102.0, 1_000)
    assert NoSlippage().fraction(order(1), bar) == 0.0
    assert FixedBpsSlippage(5).fraction(order(1), bar) == pytest.approx(5e-4)
    assert BarRangeSlippage(0.1).fraction(order(1), bar) == pytest.approx(0.1 * 6 / 100)
    with pytest.raises(ValueError):
        FixedBpsSlippage(-1)


def _feed_with_history(volume_last=1_000.0, n=30):
    frame = generate_ohlcv_panel(["A"], periods=n, seed=4, base_volume=1_000)["A"].copy()
    frame.iloc[-1, frame.columns.get_loc("volume")] = volume_last
    feed = HistoricalDataFeed({"A": frame})
    for _ in range(n):
        feed.step()
    return feed


def test_sqrt_impact_scales_with_root_of_size_and_ignores_current_bar():
    model = SquareRootImpact(coefficient=1.0, lookback=20)
    feed = _feed_with_history()
    small, big = model.fraction("A", 100, feed), model.fraction("A", 400, feed)
    assert small > 0
    assert big == pytest.approx(2.0 * small)  # sqrt(4) == 2

    # The current (last published) bar's volume must not influence the estimate.
    same_history_other_volume = _feed_with_history(volume_last=9_000_000.0)
    assert model.fraction("A", 100, same_history_other_volume) == pytest.approx(small)


def test_sqrt_impact_is_zero_without_history_and_capped():
    model = SquareRootImpact(coefficient=1.0, min_history=5, max_fraction=0.05)
    feed = HistoricalDataFeed({"A": frame_from_closes([100, 101, 102, 103, 104, 105])})
    feed.step()
    feed.step()
    assert model.fraction("A", 100, feed) == 0.0
    assert NoImpact().fraction("A", 100, feed) == 0.0

    full = _feed_with_history()
    assert model.fraction("A", 1e12, full) == 0.05
    for bad in (
        lambda: SquareRootImpact(coefficient=-1),
        lambda: SquareRootImpact(lookback=1),
        lambda: SquareRootImpact(max_fraction=1.0),
    ):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------ order types


def test_market_orders_fill_at_open_with_adverse_slippage_and_commission():
    frame = ohlc_frame([(100, 101, 99, 100, 10_000)] * 3)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(BpsCommission(10), FixedBpsSlippage(10), max_participation=None)
    handler.submit(order(100, 1))
    handler.submit(order(-50, 2))
    report = run_bar(handler, feed)

    buy, sell = report.fills
    assert buy.price == pytest.approx(100.0 * 1.001)
    assert sell.price == pytest.approx(100.0 * 0.999)
    assert buy.commission == pytest.approx(100 * buy.price * 10 / 1e4)
    assert buy.order_id == 1 and sell.quantity == -50
    assert handler.working_orders == {}


def test_orders_are_not_filled_on_the_bar_they_were_submitted_on():
    frame = ohlc_frame([(100, 101, 99, 100, 10_000)] * 3)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(max_participation=None)
    run_bar(handler, feed)  # bar 0 happens with no working orders
    handler.submit(order(10))
    assert handler.working_orders == {1: 10}
    assert len(run_bar(handler, feed).fills) == 1  # filled on the next bar


def test_participation_cap_forces_partial_fills_over_several_bars():
    frame = ohlc_frame([(100, 101, 99, 100, 1_000)] * 12)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(max_participation=0.10)
    handler.submit(order(1_000))
    filled = []
    for _ in range(12):
        filled.extend(f.quantity for f in run_bar(handler, feed).fills)
    assert filled == [100.0] * 10
    assert handler.working_orders == {}


def test_participation_capacity_is_shared_between_orders_in_a_symbol():
    frame = ohlc_frame([(100, 101, 99, 100, 1_000)] * 3)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(max_participation=0.10)
    handler.submit(order(80, 1))
    handler.submit(order(80, 2))
    fills = run_bar(handler, feed).fills
    assert [f.quantity for f in fills] == [80.0, 20.0]  # second gets what is left of 100
    assert handler.working_orders == {2: 60.0}


def test_whole_share_orders_fill_in_whole_shares_fractional_orders_do_not():
    frame = ohlc_frame([(100, 101, 99, 100, 1_005)] * 2)  # 10% of volume = 100.5 shares
    whole = ExecutionHandler(max_participation=0.10)
    whole.submit(order(500))
    assert run_bar(whole, HistoricalDataFeed({"A": frame})).fills[0].quantity == 100.0

    fractional = ExecutionHandler(max_participation=0.10)
    fractional.submit(order(500.5))
    qty = run_bar(fractional, HistoricalDataFeed({"A": frame})).fills[0].quantity
    assert qty == pytest.approx(100.5)


def test_buy_limit_orders():
    frame = ohlc_frame(
        [
            (100, 101, 99, 100, 5_000),  # bar 0: warm-up
            (98, 99, 97, 98, 5_000),  # bar 1: gaps below the limit -> fills at the better open
            (101, 102, 99.5, 101, 5_000),  # bar 2: dips to the limit -> fills exactly at it
            (101, 103, 100, 102, 5_000),  # bar 3: never trades at/below the limit -> no fill
        ]
    )
    cases = [(1, 99.0, 98.0), (2, 99.5, 99.5), (3, 99.0, None)]  # (bar, limit, expected fill)
    for bar_index, limit, expected in cases:
        feed = HistoricalDataFeed({"A": frame})
        for _ in range(bar_index):
            feed.step()
        handler = ExecutionHandler(slippage=FixedBpsSlippage(50), max_participation=None)
        handler.submit(order(10, order_type=OrderType.LIMIT, limit_price=limit))
        fills = run_bar(handler, feed).fills
        if expected is None:
            assert fills == ()
            assert handler.working_orders == {1: 10}
        else:  # exact prices: limit orders pay no slippage
            assert [f.price for f in fills] == [pytest.approx(expected)]


def test_sell_limit_orders_mirror_buy_limits():
    frame = ohlc_frame(
        [(100, 101, 99, 100, 5_000), (100, 103, 99, 102, 5_000), (106, 107, 105, 106, 5_000)]
    )
    feed = HistoricalDataFeed({"A": frame})
    feed.step()
    handler = ExecutionHandler(max_participation=None)
    handler.submit(order(-10, 1, order_type=OrderType.LIMIT, limit_price=102.0))
    assert run_bar(handler, feed).fills[0].price == pytest.approx(102.0)  # high 103 >= 102

    handler.submit(order(-10, 2, order_type=OrderType.LIMIT, limit_price=104.0))
    # the bar gapped up, so the open is a better price than the limit
    assert run_bar(handler, feed).fills[0].price == pytest.approx(106.0)


def test_stop_orders_trigger_on_the_bar_range_and_pay_slippage():
    frame = ohlc_frame(
        [
            (100, 101, 99, 100, 5_000),
            (100, 104, 99, 103, 5_000),  # buy stop 102 triggers intrabar
            (108, 109, 107, 108, 5_000),  # buy stop 102 would gap through to the open
            (100, 101, 95, 96, 5_000),  # sell stop 97 triggers intrabar
        ]
    )

    def fill_price(bar_index, qty, stop):
        feed = HistoricalDataFeed({"A": frame})
        for _ in range(bar_index):
            feed.step()
        handler = ExecutionHandler(slippage=FixedBpsSlippage(10), max_participation=None)
        handler.submit(order(qty, order_type=OrderType.STOP, stop_price=stop))
        fills = run_bar(handler, feed).fills
        return fills[0].price if fills else None

    assert fill_price(1, 10, 102.0) == pytest.approx(102.0 * 1.001)
    assert fill_price(2, 10, 102.0) == pytest.approx(108.0 * 1.001)
    assert fill_price(3, -10, 97.0) == pytest.approx(97.0 * 0.999)
    assert fill_price(1, 10, 110.0) is None  # never reached


def test_orders_expire_and_partial_remainders_are_reported():
    frame = ohlc_frame([(100, 101, 99, 100, 1_000)] * 5)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(max_participation=0.10, max_order_age=2)
    handler.submit(order(1_000, 7))
    first = run_bar(handler, feed)
    second = run_bar(handler, feed)
    assert [f.quantity for f in first.fills + second.fills] == [100.0, 100.0]
    assert [o.order_id for o in second.expired] == [7]
    assert handler.working_orders == {}


def test_unfilled_limit_expires_but_idle_symbols_do_not_age():
    a = ohlc_frame([(100, 101, 99, 100, 5_000)] * 4)
    b = a.drop(a.index[1])  # B does not trade on bar 1
    feed = HistoricalDataFeed({"A": a, "B": b})
    handler = ExecutionHandler(max_participation=None, max_order_age=2)
    handler.submit(order(10, 1, symbol="B", order_type=OrderType.LIMIT, limit_price=50.0))
    assert run_bar(handler, feed).expired == ()  # bar 0: age 1
    assert run_bar(handler, feed).expired == ()  # bar 1: B absent, no ageing
    expired = run_bar(handler, feed).expired  # bar 2: age 2 -> expires
    assert [o.order_id for o in expired] == [1]


def test_cancel_and_duplicate_ids():
    handler = ExecutionHandler()
    handler.submit(order(5, 3))
    with pytest.raises(ValueError, match="already working"):
        handler.submit(order(5, 3))
    assert handler.cancel(3) is True and handler.cancel(3) is False


def test_costs_frame_breaks_down_every_fill():
    frame = ohlc_frame([(100, 101, 99, 100, 10_000)] * 3)
    feed = HistoricalDataFeed({"A": frame})
    handler = ExecutionHandler(BpsCommission(10), FixedBpsSlippage(20), max_participation=None)
    assert handler.costs_frame().empty
    handler.submit(order(100))
    run_bar(handler, feed)
    costs = handler.costs_frame()
    row = costs.iloc[0]
    assert row["reference_price"] == 100.0
    assert row["slippage_cost"] == pytest.approx(100 * 100.0 * 20e-4)
    assert row["commission"] == pytest.approx(100 * 100.2 * 10e-4)
    assert row["total_cost"] == pytest.approx(row["slippage_cost"] + row["commission"])
    assert row["participation"] == pytest.approx(100 / 10_000)


@pytest.mark.parametrize(
    "kwargs", [{"max_participation": 0.0}, {"max_participation": 1.5}, {"max_order_age": 0}]
)
def test_handler_validation(kwargs):
    with pytest.raises(ValueError):
        ExecutionHandler(**kwargs)


def test_realistic_preset_has_nonzero_costs():
    handler = ExecutionHandler.realistic()
    assert handler.commission(100, 50.0) > 0
    assert handler.slippage.fraction(order(1), Bar(1, 1, 1, 1, 1)) > 0
    assert isinstance(handler.impact, SquareRootImpact)
    assert math.isclose(handler.max_participation, 0.10)
