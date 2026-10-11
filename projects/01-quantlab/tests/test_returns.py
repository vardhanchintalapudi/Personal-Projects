import numpy as np
import pandas as pd
import pytest

from quantlab.utils import (
    annualization_factor,
    annualize_return,
    annualize_vol,
    cumulative_returns,
    log_returns,
    simple_returns,
)


@pytest.fixture()
def prices():
    return pd.Series([100.0, 110.0, 99.0, 108.9], index=pd.date_range("2024-01-01", periods=4))


def test_simple_and_log_returns(prices):
    simple = simple_returns(prices)
    assert simple.tolist() == pytest.approx([0.10, -0.10, 0.10])
    np.testing.assert_allclose(np.exp(log_returns(prices)) - 1, simple)


def test_cumulative_matches_price_ratio(prices):
    cum = cumulative_returns(simple_returns(prices))
    assert cum.iloc[-1] == pytest.approx(prices.iloc[-1] / prices.iloc[0] - 1)


def test_annualization_factor_aliases():
    assert annualization_factor("B") == 252
    assert annualization_factor("W") == 52
    assert annualization_factor("5min") == round(252 * 390 / 5)
    assert annualization_factor(periods_per_year=12) == 12
    with pytest.raises(ValueError):
        annualization_factor("bogus")
    with pytest.raises(ValueError):
        annualization_factor()
    with pytest.raises(ValueError):
        annualization_factor(periods_per_year=0)


def test_annualize_return_and_vol():
    r = pd.Series([0.01] * 252)
    assert annualize_return(r, 252) == pytest.approx(1.01**252 - 1)
    assert annualize_vol(r, 252) == pytest.approx(0.0)
    noisy = pd.Series(np.random.default_rng(0).normal(0, 0.01, 10_000))
    assert annualize_vol(noisy, 252) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_degenerate_inputs():
    assert np.isnan(annualize_return(pd.Series(dtype=float), 252))
    assert np.isnan(annualize_vol(pd.Series([0.1]), 252))
    assert annualize_return(pd.Series([-1.5]), 252) == -1.0
