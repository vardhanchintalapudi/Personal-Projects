"""Performance and risk metrics computed from a series of periodic simple returns.

Every function is pure: returns in, numbers out. Conventions:

* ``returns`` is a ``pd.Series`` of simple returns per bar; NaNs are dropped.
* ``periods_per_year`` annualises (252 for daily equity bars).
* Risk measures (``max_drawdown`` aside) report *losses as positive numbers*, so
  a 95 % VaR of ``0.021`` means "lose 2.1 % or more on the worst 5 % of bars".
* Anything undefined (an empty series, zero volatility) returns ``nan`` rather
  than raising, so summaries over many strategies never crash on one bad run.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from quantlab.utils.returns import annualize_return, annualize_vol

_NORMAL = NormalDist()
_DD_TOL = 1e-12


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).astype("float64").dropna()


# ------------------------------------------------------------------ headline


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else float("nan")


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    return annualize_return(_clean(returns), periods_per_year)


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return annualize_vol(_clean(returns), periods_per_year)


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0
) -> float:
    """Annualised Sharpe ratio: mean excess return over its standard deviation.

    ``risk_free`` is an *annual* rate and is converted to a per-bar rate.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free / periods_per_year
    std = float(excess.std(ddof=1))
    if std < 1e-15:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = 252, target: float = 0.0
) -> float:
    """Annualised Sortino ratio against an annual ``target`` return.

    The denominator is the target downside deviation over *all* bars,
    ``sqrt(mean(min(r - target, 0) ** 2))``, so bars above target count as zero
    downside rather than being excluded.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    gap = r - target / periods_per_year
    downside = float(np.sqrt(np.mean(np.minimum(gap, 0.0) ** 2)))
    if downside < 1e-15:
        return float("nan")
    return float(gap.mean() / downside * np.sqrt(periods_per_year))


# ----------------------------------------------------------------- drawdowns


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak of wealth (starting wealth 1.0), as a negative fraction."""
    r = _clean(returns)
    wealth = (1.0 + r).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return (wealth / peak - 1.0).rename("drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline as a negative fraction (``0.0`` if never below peak)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or abs(mdd) < _DD_TOL:
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest drawdown episodes.

    Columns: ``peak`` (last bar at a high before the episode), ``trough``,
    ``recovery`` (first bar back at the old high, ``NaT`` if still underwater),
    ``depth`` (negative fraction), ``bars_to_trough`` and ``bars_underwater``
    (peak to recovery, or to the end of the data if unrecovered).
    """
    dd = drawdown_series(returns)
    columns = ["peak", "trough", "recovery", "depth", "bars_to_trough", "bars_underwater"]
    if dd.empty:
        return pd.DataFrame(columns=columns)

    values = dd.to_numpy()
    index = dd.index
    under = values < -_DD_TOL

    episodes = []
    i, n = 0, len(values)
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i
        while i < n and under[i]:
            i += 1
        end = i  # first bar back at the peak, or n if never recovered
        trough = start + int(np.argmin(values[start:end]))
        peak = start - 1 if start > 0 else 0
        episodes.append(
            {
                "peak": index[peak],
                "trough": index[trough],
                "recovery": index[end] if end < n else pd.NaT,
                "depth": float(values[trough]),
                "bars_to_trough": trough - peak,
                "bars_underwater": (end if end < n else n - 1) - peak,
            }
        )

    table = pd.DataFrame(episodes, columns=columns)
    return table.sort_values("depth").head(top).reset_index(drop=True)


# ------------------------------------------------------------- tail risk


def _tail_quantile(r: pd.Series, level: float) -> float:
    return float(np.quantile(r.to_numpy(), 1.0 - level))


def value_at_risk(
    returns: pd.Series, level: float = 0.95, method: str = "historical"
) -> float:
    """Value at risk as a positive loss fraction per bar.

    ``method``: ``"historical"`` (empirical quantile), ``"gaussian"`` (normal fit)
    or ``"cornish_fisher"`` (normal fit adjusted for skewness and excess
    kurtosis, better for fat-tailed returns).
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")

    if method == "historical":
        return -_tail_quantile(r, level)

    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    z = _NORMAL.inv_cdf(1.0 - level)
    if method == "gaussian":
        return -(mu + z * sigma)
    if method == "cornish_fisher":
        s, k = float(r.skew()), float(r.kurt())
        z_cf = (
            z
            + (z**2 - 1.0) * s / 6.0
            + (z**3 - 3.0 * z) * k / 24.0
            - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
        )
        return -(mu + z_cf * sigma)
    raise ValueError("method must be 'historical', 'gaussian' or 'cornish_fisher'")


def conditional_var(
    returns: pd.Series, level: float = 0.95, method: str = "historical"
) -> float:
    """Expected shortfall: the average loss on the bars at or beyond the VaR threshold."""
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1)")
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")

    if method == "historical":
        threshold = _tail_quantile(r, level)
        return -float(r[r <= threshold].mean())
    if method == "gaussian":
        mu, sigma = float(r.mean()), float(r.std(ddof=1))
        z = _NORMAL.inv_cdf(1.0 - level)
        return -mu + sigma * _NORMAL.pdf(z) / (1.0 - level)
    raise ValueError("method must be 'historical' or 'gaussian'")


def tail_ratio(returns: pd.Series, q: float = 0.05) -> float:
    """Size of the right tail relative to the left: ``|P(1-q)| / |P(q)|``. Above 1 is good."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    left = abs(np.quantile(r.to_numpy(), q))
    right = abs(np.quantile(r.to_numpy(), 1.0 - q))
    return float(right / left) if left > 1e-15 else float("nan")


def win_rate(returns: pd.Series) -> float:
    """Share of bars with a gain, among bars with any profit or loss."""
    r = _clean(returns)
    active = r[r != 0.0]
    return float((active > 0).mean()) if len(active) else float("nan")


# -------------------------------------------------------- rolling and calendar


def rolling_sharpe(
    returns: pd.Series, window: int = 126, periods_per_year: int = 252, risk_free: float = 0.0
) -> pd.Series:
    """Rolling annualised Sharpe ratio (``NaN`` until ``window`` bars exist)."""
    r = pd.Series(returns).astype("float64")
    excess = r - risk_free / periods_per_year
    roll = excess.rolling(window)
    std = roll.std(ddof=1).where(lambda s: s > 1e-15)
    return (roll.mean() / std * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_volatility(
    returns: pd.Series, window: int = 63, periods_per_year: int = 252
) -> pd.Series:
    r = pd.Series(returns).astype("float64")
    return (r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)).rename("rolling_vol")


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded return per calendar month: rows are years, columns are months 1-12."""
    r = _clean(returns)
    if r.empty:
        return pd.DataFrame(columns=range(1, 13))
    grouped = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = grouped.unstack(level=1)
    return table.reindex(columns=range(1, 13))


# --------------------------------------------------------- benchmark-relative


def benchmark_stats(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Beta, annualised alpha, tracking error and capture ratios against a benchmark.

    The two series are aligned on their common timestamps.
    """
    joined = pd.concat([_clean(returns).rename("r"), _clean(benchmark).rename("b")], axis=1)
    joined = joined.dropna()
    keys = [
        "beta",
        "alpha",
        "correlation",
        "r_squared",
        "tracking_error",
        "information_ratio",
        "up_capture",
        "down_capture",
    ]
    if len(joined) < 3:
        return dict.fromkeys(keys, float("nan"))

    r, b = joined["r"], joined["b"]
    var_b = float(b.var(ddof=1))
    beta = float(r.cov(b) / var_b) if var_b > 1e-18 else float("nan")
    rf = risk_free / periods_per_year
    alpha = (
        float(((r - rf).mean() - beta * (b - rf).mean()) * periods_per_year)
        if np.isfinite(beta)
        else float("nan")
    )
    corr = float(r.corr(b)) if var_b > 1e-18 and float(r.var(ddof=1)) > 1e-18 else float("nan")
    active = r - b
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    ir = float(active.mean() * periods_per_year / te) if te > 1e-15 else float("nan")

    up, down = b > 0, b < 0
    up_capture = float(r[up].mean() / b[up].mean()) if up.any() else float("nan")
    down_capture = float(r[down].mean() / b[down].mean()) if down.any() else float("nan")

    return {
        "beta": beta,
        "alpha": alpha,
        "correlation": corr,
        "r_squared": corr**2 if np.isfinite(corr) else float("nan"),
        "tracking_error": te,
        "information_ratio": ir,
        "up_capture": up_capture,
        "down_capture": down_capture,
    }


# --------------------------------------------------------------------- summary


def performance_summary(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.Series:
    """One-stop summary of a return series (plus benchmark-relative stats if given)."""
    r = _clean(returns)
    stats: dict[str, float] = {
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "annual_vol": annual_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, periods_per_year, risk_free),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "var_95": value_at_risk(r, 0.95),
        "cvar_95": conditional_var(r, 0.95),
        "skew": float(r.skew()) if len(r) > 2 else float("nan"),
        "excess_kurtosis": float(r.kurt()) if len(r) > 3 else float("nan"),
        "best_bar": float(r.max()) if len(r) else float("nan"),
        "worst_bar": float(r.min()) if len(r) else float("nan"),
        "win_rate": win_rate(r),
        "tail_ratio": tail_ratio(r),
        "n_bars": float(len(r)),
    }
    if benchmark is not None:
        stats.update(benchmark_stats(r, benchmark, periods_per_year, risk_free))
    return pd.Series(stats, name="performance")
