import pytest

from quantlab.data import generate_ohlcv_panel


@pytest.fixture(scope="session")
def market():
    return generate_ohlcv_panel(["AAA", "BBB", "CCC"], periods=500, seed=7)


@pytest.fixture()
def aaa(market):
    return market["AAA"].copy()
