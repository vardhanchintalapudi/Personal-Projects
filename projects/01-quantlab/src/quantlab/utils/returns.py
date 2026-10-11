"""Return calculations and annualisation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import PERIODS_PER_YEAR

Series = pd.Series | pd.DataFrame


def simple_returns(prices: Series) -> Series:
    """Period-over-period simple returns ``P_t / P_{t-1} - 1`` (first row dropped)."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: Series) -> Series:
    """Period-over-period log returns ``ln(P_t / P_{t-1})`` (first row dropped)."""
    return np.log(prices).diff().iloc[1:]


def cumulative_returns(returns: Series) -> Series:
    """Compound simple returns into a cumulative-return curve starting near 0."""
    return (1.0 + returns).cumprod() - 1.0


def annualization_factor(freq: str | None = None, periods_per_year: int | None = None) -> int:
    """Return bars-per-year for a pandas frequency alias.

    An explicit ``periods_per_year`` always wins. Aliases with multipliers
    (e.g. ``"5min"``) are resolved by scaling the base alias.
    """
    if periods_per_year is not None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        return periods_per_year
    if freq is None:
        raise ValueError("Provide either freq or periods_per_year")

    base = freq.lstrip("0123456789")
    mult_str = freq[: len(freq) - len(base)]
    mult = int(mult_str) if mult_str else 1
    if base not in PERIODS_PER_YEAR:
        raise ValueError(f"Unknown frequency alias: {freq!r}")
    return max(1, round(PERIODS_PER_YEAR[base] / mult))


def annualize_return(returns: pd.Series, periods_per_year: int) -> float:
    """Geometric (CAGR-style) annualised return from periodic simple returns."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / n) - 1.0


def annualize_vol(returns: pd.Series, periods_per_year: int, ddof: int = 1) -> float:
    """Annualised standard deviation of periodic returns."""
    if len(returns) <= ddof:
        return float("nan")
    return float(returns.std(ddof=ddof) * np.sqrt(periods_per_year))
