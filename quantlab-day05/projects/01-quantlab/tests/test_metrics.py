import numpy as np
import pandas as pd
import pytest

from quantlab.analytics import (
    annual_volatility,
    benchmark_stats,
    cagr,
    calmar_ratio,
    conditional_var,
    drawdown_series,
    drawdown_table,
    max_drawdown,
    monthly_returns,
    performance_summary,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
    value_at_risk,
    win_rate,
)

DAYS = pd.bdate_range("2023-01-02", periods=400)


def series(values, start="2023-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


# -------------------------------------------------------------------- headline


def test_total_return_and_cagr():
    r = series([0.10, -0.10])
    assert total_return(r) == pytest.approx(1.1 * 0.9 - 1)
    flat_year = series([0.0] * 252)
    assert cagr(flat_year, 252) == pytest.approx(0.0)
    steady = series([0.001] * 252)
    assert cagr(steady, 252) == pytest.approx(1.001**252 - 1)
    assert annual_volatility(series([0.01, -0.01, 0.01, -0.01]), 252) == pytest.approx(
        0.01 * np.sqrt(252) * np.sqrt(4 / 3)
    )


def test_sharpe_matches_hand_calculation():
    r = series([0.01, 0.02, 0.03])  # mean 0.02, sample std 0.01
    assert sharpe_ratio(r, 252) == pytest.approx(2.0 * np.sqrt(252))
    # an annual risk-free rate of 2.52% is 0.0001 per bar
    assert sharpe_ratio(r, 252, risk_free=0.0252) == pytest.approx(1.99 * np.sqrt(252))


def test_sharpe_is_nan_when_undefined():
    assert np.isnan(sharpe_ratio(series([0.01] * 10)))  # zero volatility
    assert np.isnan(sharpe_ratio(series([0.01])))
    assert np.isnan(sharpe_ratio(pd.Series(dtype=float)))


def test_sortino_matches_hand_calculation():
    r = series([0.02, -0.01, 0.03, -0.02])
    downside = np.sqrt((0.01**2 + 0.02**2) / 4)
    assert sortino_ratio(r, 252) == pytest.approx(0.005 / downside * np.sqrt(252))
    assert np.isnan(sortino_ratio(series([0.01, 0.02, 0.03])))  # no downside at all
    # a positive target raises the bar and lowers the ratio
    assert sortino_ratio(r, 252, target=0.5) < sortino_ratio(r, 252)


def test_nans_are_ignored():
    with_nan = series([0.01, np.nan, 0.02, 0.03])
    assert sharpe_ratio(with_nan) == pytest.approx(sharpe_ratio(series([0.01, 0.02, 0.03])))


# ------------------------------------------------------------------- drawdowns


def test_drawdown_series_and_max_drawdown():
    r = series([0.1, -0.2, 0.05, 0.1, -0.1])
    wealth = np.cumprod(1 + r.to_numpy())
    expected = wealth / np.maximum.accumulate(wealth) - 1
    np.testing.assert_allclose(drawdown_series(r).to_numpy(), expected)
    assert max_drawdown(r) == pytest.approx(-0.2)
    assert max_drawdown(series([0.01, 0.02])) == 0.0


def test_first_bar_loss_counts_as_drawdown_from_starting_wealth():
    assert max_drawdown(series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_calmar():
    r = series([0.001] * 100 + [-0.1] + [0.001] * 150)
    assert calmar_ratio(r, 252) == pytest.approx(cagr(r, 252) / abs(max_drawdown(r)))
    assert np.isnan(calmar_ratio(series([0.01, 0.01])))  # never in drawdown


def test_drawdown_table_finds_recovered_and_open_episodes():
    r = series([0.1, -0.2, 0.25, 0.1, -0.05, -0.05])
    table = drawdown_table(r)
    assert list(table.columns) == [
        "peak",
        "trough",
        "recovery",
        "depth",
        "bars_to_trough",
        "bars_underwater",
    ]
    assert len(table) == 2
    first = table.iloc[0]  # deepest: -20%, recovered
    assert first["depth"] == pytest.approx(-0.2)
    assert first["peak"] == r.index[0] and first["trough"] == r.index[1]
    assert first["recovery"] == r.index[2]
    assert first["bars_to_trough"] == 1 and first["bars_underwater"] == 2
    second = table.iloc[1]  # still underwater at the end
    assert second["depth"] == pytest.approx(0.95 * 0.95 - 1)
    assert pd.isna(second["recovery"])
    assert second["trough"] == r.index[5]
    assert second["bars_underwater"] == 2  # peak at bar 3, data ends at bar 5


def test_drawdown_table_top_n_and_empty():
    r = series([0.1, -0.1, 0.2, -0.05, 0.3, -0.02, 0.4])
    assert len(drawdown_table(r, top=2)) == 2
    assert drawdown_table(r, top=2)["depth"].is_monotonic_increasing
    assert drawdown_table(pd.Series(dtype=float)).empty
    assert drawdown_table(series([0.01, 0.02])).empty


# ------------------------------------------------------------------- tail risk


def test_historical_var_and_cvar_on_a_uniform_grid():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert value_at_risk(r, 0.95) == pytest.approx(0.045)
    # tail = the six worst points -0.050 ... -0.045, mean -0.0475
    assert conditional_var(r, 0.95) == pytest.approx(0.0475)
    assert conditional_var(r, 0.95) >= value_at_risk(r, 0.95)


def test_gaussian_var_formula_and_cvar():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 50_000))
    z95 = 1.6448536269514722
    expected_var = -(r.mean() - z95 * r.std(ddof=1))
    assert value_at_risk(r, 0.95, "gaussian") == pytest.approx(expected_var, rel=1e-6)
    assert value_at_risk(r, 0.95, "historical") == pytest.approx(expected_var, rel=0.05)
    assert conditional_var(r, 0.95, "gaussian") == pytest.approx(
        conditional_var(r, 0.95, "historical"), rel=0.05
    )


def test_cornish_fisher_penalises_fat_left_tails():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, 20_000)
    crashes = rng.random(20_000) < 0.01
    r[crashes] -= 0.08
    r = pd.Series(r)
    assert r.skew() < 0 and r.kurt() > 0
    assert value_at_risk(r, 0.99, "cornish_fisher") > value_at_risk(r, 0.99, "gaussian")


def test_var_validation_and_degenerate_inputs():
    r = series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError):
        value_at_risk(r, 0.3)
    with pytest.raises(ValueError):
        value_at_risk(r, 0.95, "magic")
    with pytest.raises(ValueError):
        conditional_var(r, 1.0)
    with pytest.raises(ValueError):
        conditional_var(r, 0.95, "cornish_fisher")
    assert np.isnan(value_at_risk(series([0.01])))
    assert np.isnan(conditional_var(series([0.01])))


def test_tail_ratio_and_win_rate():
    sym = pd.Series(np.linspace(-1, 1, 1001))
    assert tail_ratio(sym) == pytest.approx(1.0)
    skewed = pd.Series(np.concatenate([np.linspace(-0.01, 0.0, 500), np.linspace(0.0, 0.05, 501)]))
    assert tail_ratio(skewed) > 1.0
    assert win_rate(series([0.01, -0.01, 0.0, 0.02, 0.0])) == pytest.approx(2 / 3)
    assert np.isnan(win_rate(series([0.0, 0.0])))


# ------------------------------------------------------------ rolling/calendar


def test_rolling_metrics_match_manual_windows():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0004, 0.01, 300), index=DAYS[:300])
    rs = rolling_sharpe(r, window=60)
    assert rs.iloc[:59].isna().all() and rs.iloc[59:].notna().all()
    window = r.iloc[100:160]
    assert rs.iloc[159] == pytest.approx(window.mean() / window.std(ddof=1) * np.sqrt(252))
    rv = rolling_volatility(r, window=60)
    assert rv.iloc[159] == pytest.approx(window.std(ddof=1) * np.sqrt(252))


def test_rolling_sharpe_is_nan_for_constant_windows():
    flat = pd.Series([0.001] * 30, index=DAYS[:30])
    assert rolling_sharpe(flat, window=10).isna().all()


def test_monthly_returns_table():
    idx = pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02", "2024-01-02"])
    r = pd.Series([0.01, 0.02, -0.03, 0.01, 0.05], index=idx)
    table = monthly_returns(r)
    assert list(table.columns) == list(range(1, 13))
    assert list(table.index) == [2023, 2024]
    assert table.loc[2023, 1] == pytest.approx(1.01 * 1.02 - 1)
    assert table.loc[2023, 2] == pytest.approx(0.97 * 1.01 - 1)
    assert table.loc[2024, 1] == pytest.approx(0.05)
    assert pd.isna(table.loc[2023, 3])
    assert monthly_returns(pd.Series(dtype=float)).empty


# -------------------------------------------------------- benchmark-relative


def test_benchmark_stats_for_a_levered_clone_with_alpha():
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0.0003, 0.01, 500), index=pd.bdate_range("2023-01-02", periods=500))
    r = 0.0002 + 1.5 * b
    stats = benchmark_stats(r, b, 252)
    assert stats["beta"] == pytest.approx(1.5)
    assert stats["alpha"] == pytest.approx(0.0002 * 252)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["r_squared"] == pytest.approx(1.0)
    assert stats["up_capture"] > 1.5 and stats["down_capture"] < 1.5  # alpha lifts both
    assert stats["information_ratio"] > 0 and stats["tracking_error"] > 0


def test_benchmark_stats_align_on_common_dates_and_handle_tiny_inputs():
    r = series([0.01, 0.02, -0.01, 0.0, 0.03])
    b = series([0.01, 0.0, -0.02, 0.01, 0.02, 0.5], start="2023-01-02")
    assert np.isfinite(benchmark_stats(r, b)["beta"])
    nan_stats = benchmark_stats(series([0.01, 0.02]), series([0.01, 0.02]))
    assert all(np.isnan(v) for v in nan_stats.values())
    flat_benchmark = benchmark_stats(r, series([0.0] * 5))
    assert np.isnan(flat_benchmark["beta"]) and np.isnan(flat_benchmark["alpha"])


# ---------------------------------------------------------------- summary


def test_performance_summary_contents():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0005, 0.01, 400), index=DAYS)
    summary = performance_summary(r)
    for key in [
        "total_return", "cagr", "annual_vol", "sharpe", "sortino", "max_drawdown", "calmar",
        "var_95", "cvar_95", "skew", "excess_kurtosis", "win_rate", "tail_ratio", "n_bars",
    ]:
        assert key in summary.index and np.isfinite(summary[key])
    assert summary["n_bars"] == 400
    assert "beta" not in summary.index
    assert summary["total_return"] == pytest.approx(total_return(r))

    with_bench = performance_summary(r, benchmark=r * 0.5)
    assert with_bench["beta"] == pytest.approx(2.0)


def test_performance_summary_survives_degenerate_series():
    for r in (pd.Series(dtype=float), series([0.0] * 5), series([0.01])):
        summary = performance_summary(r)
        assert "sharpe" in summary.index  # no exception, values may be NaN
