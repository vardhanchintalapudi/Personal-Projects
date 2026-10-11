import pytest
from helpers import Scripted

from quantlab.feed import HistoricalDataFeed
from quantlab.strategy import Strategy, Target
from quantlab.testing import collect_signals


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        Strategy(["AAA"])  # type: ignore[abstract]


def test_initial_flat_target_emits_nothing(market):
    strat = Scripted(["AAA"], {1: {"AAA": 0.0}})
    assert collect_signals(strat, market) == []


def test_only_changes_are_emitted(market):
    script = {1: {"AAA": 0.5}, 2: {"AAA": 0.5}, 3: {"AAA": 0.5 + 1e-9}, 4: {"AAA": 0.7}}
    signals = collect_signals(Scripted(["AAA"], script), market)
    assert [s.target_weight for s in signals] == [0.5, 0.7]
    index = market["AAA"].index
    assert [s.timestamp for s in signals] == [index[0], index[3]]


def test_weight_tolerance_suppresses_small_moves(market):
    script = {1: {"AAA": 0.5}, 2: {"AAA": 0.52}, 3: {"AAA": 0.6}}
    signals = collect_signals(Scripted(["AAA"], script, weight_tolerance=0.05), market)
    assert [s.target_weight for s in signals] == [0.5, 0.6]


def test_reason_float_targets_and_strategy_id(market):
    script = {1: {"AAA": Target(0.4, "because"), "BBB": 0.3}}
    named = collect_signals(Scripted(["AAA", "BBB"], script, strategy_id="my-strat"), market)
    by_symbol = {s.symbol: s for s in named}
    assert by_symbol["AAA"].reason == "because" and by_symbol["BBB"].reason == ""
    assert {s.strategy_id for s in named} == {"my-strat"}

    default = collect_signals(Scripted(["AAA"], {1: {"AAA": 0.1}}), market)
    assert default[0].strategy_id == "Scripted"


def test_exposure_reducing_signals_come_first(market):
    script = {
        1: {"AAA": 0.6, "BBB": 0.4},
        2: {"BBB": 0.6, "AAA": 0.4},  # BBB increases, AAA reduces, but BBB is listed first
    }
    signals = collect_signals(Scripted(["AAA", "BBB"], script), market)
    second_bar = [s for s in signals if s.timestamp == market["AAA"].index[1]]
    assert [s.symbol for s in second_bar] == ["AAA", "BBB"]


def test_on_start_resets_state_so_runs_are_repeatable(market):
    strat = Scripted(["AAA"], {1: {"AAA": 0.5}, 5: {"AAA": 0.0}})
    first = collect_signals(strat, market)
    second = collect_signals(strat, market)
    assert first == second and len(first) == 2


def test_current_weights_and_active_symbols(market):
    strat = Scripted(["AAA", "BBB"], {1: {"AAA": 0.25}})
    feed = HistoricalDataFeed(market)
    strat.on_start(feed)
    event = feed.step()
    strat.on_bar(event, feed)
    assert strat.current_weights == {"AAA": 0.25}
    assert strat.active_symbols(event) == ("AAA", "BBB")


@pytest.mark.parametrize(
    "symbols, kwargs",
    [
        ([], {}),
        (["A", "A"], {}),
        (["A"], {"weight_tolerance": -1.0}),
    ],
)
def test_constructor_validation(symbols, kwargs):
    with pytest.raises(ValueError):
        Scripted(symbols, {}, **kwargs)


def test_on_start_rejects_unknown_symbols(market):
    with pytest.raises(KeyError, match="NOPE"):
        Scripted(["AAA", "NOPE"], {}).on_start(HistoricalDataFeed(market))


def test_equal_weight_and_repr():
    assert Strategy.equal_weight(["A", "B", "C", "D"], None) == 0.25
    assert Strategy.equal_weight(["A"], 0.4) == 0.4
    with pytest.raises(ValueError):
        Strategy.equal_weight(["A"], 0.0)
    strat = Scripted(["A"], {}, params={"x": 1})
    assert repr(strat) == "Scripted(symbols=['A'], x=1)"
