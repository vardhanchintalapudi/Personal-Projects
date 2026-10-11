import numpy as np
import pytest

from quantlab.data import DataValidationError, generate_ohlcv_panel
from quantlab.events import MarketEvent
from quantlab.feed import HistoricalDataFeed


def test_replays_every_bar_in_order(market):
    feed = HistoricalDataFeed(market)
    assert len(feed) == 500
    events = list(feed)
    assert len(events) == 500
    assert all(isinstance(e, MarketEvent) for e in events)
    stamps = [e.timestamp for e in events]
    assert stamps == sorted(stamps)
    assert not feed.has_next()

    first = events[0].bars["AAA"]
    row = market["AAA"].iloc[0]
    assert (first.open, first.close, first.volume) == (row["open"], row["close"], row["volume"])


def test_exhausted_feed_raises_on_manual_step(market):
    feed = HistoricalDataFeed({"AAA": market["AAA"].iloc[:2]})
    feed.step()
    feed.step()
    with pytest.raises(IndexError):
        feed.step()


def test_no_lookahead_at_any_step(market):
    feed = HistoricalDataFeed(market)
    for i, event in enumerate(feed, start=1):
        for symbol in feed.symbols:
            hist = feed.history(symbol)
            assert len(hist) == i == feed.bars_seen(symbol)
            assert hist.index[-1] == event.timestamp == feed.current_time
            assert hist.iloc[-1] == event.bars[symbol].close
            assert (hist.index <= event.timestamp).all()


def test_missing_bars_are_absent_and_prices_carry_forward():
    panel = generate_ohlcv_panel(["A", "B"], periods=10, seed=1)
    b = panel["B"].drop(panel["B"].index[[3, 4]])
    feed = HistoricalDataFeed({"A": panel["A"], "B": b})
    assert len(feed) == 10

    events = list(feed)
    assert "B" not in events[3].bars and "B" not in events[4].bars
    assert "B" in events[5].bars
    assert feed.bars_seen("B") == 8 and feed.bars_seen("A") == 10

    feed.reset()
    for _ in range(5):  # publish timestamps 0..4
        feed.step()
    assert feed.last_close("B") == panel["B"]["close"].iloc[2]
    assert feed.last_prices()["B"] == panel["B"]["close"].iloc[2]
    assert feed.current_time == panel["A"].index[4]


def test_history_windows_and_fields(market):
    feed = HistoricalDataFeed(market)
    assert feed.history("AAA").empty
    assert feed.last_close("AAA") is None
    for _ in range(30):
        feed.step()

    assert len(feed.history("AAA", 10)) == 10
    assert len(feed.history("AAA", 1000)) == 30
    np.testing.assert_array_equal(
        feed.history("AAA", 5, "high").to_numpy(), market["AAA"]["high"].iloc[25:30].to_numpy()
    )
    assert feed.history_frame("AAA", 3).shape == (3, 5)
    assert feed.window("AAA", 30).shape == (30,)
    assert feed.window("AAA", 31) is None


def test_history_panel_alignment():
    panel = generate_ohlcv_panel(["A", "B"], periods=10, seed=2)
    b = panel["B"].drop(panel["B"].index[[1, 2]])
    feed = HistoricalDataFeed({"A": panel["A"], "B": b})
    for _ in range(6):
        feed.step()

    inner = feed.history_panel(["A", "B"])
    outer = feed.history_panel(["A", "B"], how="outer")
    assert list(inner.columns) == ["A", "B"]
    assert len(inner) == 4 and len(outer) == 6
    assert outer["B"].isna().sum() == 2
    assert not inner.isna().any().any()
    assert (outer.index <= feed.current_time).all()


def test_reset_replays_identically(market):
    feed = HistoricalDataFeed(market)
    first = [e.timestamp for e in feed][:10]
    feed.reset()
    assert feed.current_time is None
    assert [e.timestamp for e in feed][:10] == first


@pytest.mark.parametrize("bad", ["symbol", "field", "n"])
def test_query_validation(market, bad):
    feed = HistoricalDataFeed(market)
    feed.step()
    with pytest.raises((KeyError, ValueError)):
        if bad == "symbol":
            feed.history("NOPE")
        elif bad == "field":
            feed.history("AAA", 5, "vwap")
        else:
            feed.history("AAA", 0)


def test_construction_checks(market):
    with pytest.raises(ValueError, match="empty"):
        HistoricalDataFeed({})

    broken = market["AAA"].copy()
    broken.iloc[0, broken.columns.get_loc("close")] = -1.0
    with pytest.raises(DataValidationError):
        HistoricalDataFeed({"AAA": broken})
    HistoricalDataFeed({"AAA": broken}, validate=False)

    aware = market["BBB"].copy()
    aware.index = aware.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone"):
        HistoricalDataFeed({"AAA": market["AAA"], "BBB": aware})
