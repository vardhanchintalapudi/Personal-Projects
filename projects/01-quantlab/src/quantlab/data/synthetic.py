"""Synthetic market data for deterministic tests, demos and research sandboxes.

Prices follow a correlated geometric Brownian motion simulated on a *fine*
intraday grid, so each bar's open/high/low/close come from a real path rather
than ad-hoc noise. An optional two-state Markov chain switches the market
between calm/bullish and turbulent/bearish regimes.

Prices are continuous: ``open[t] == close[t-1]`` (no overnight gaps).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_SETTINGS
from quantlab.utils.returns import annualization_factor


@dataclass(frozen=True)
class Regime:
    """A market regime expressed relative to each asset's base parameters.

    ``drift_shift`` is *added* to the asset's annual drift and ``vol_multiplier``
    scales its annual volatility while the regime is active.
    """

    name: str
    drift_shift: float
    vol_multiplier: float


#: Calm bull market vs. turbulent bear market. With the default transition
#: matrix the stationary mix is 75/25 and the average drift shift is zero.
DEFAULT_REGIMES: tuple[Regime, ...] = (
    Regime("bull", drift_shift=0.06, vol_multiplier=0.8),
    Regime("bear", drift_shift=-0.18, vol_multiplier=1.8),
)

DEFAULT_TRANSITION: np.ndarray = np.array([[0.98, 0.02], [0.06, 0.94]])


def simulate_regime_states(
    n_bars: int,
    transition: np.ndarray,
    rng: np.random.Generator,
    initial_state: int = 0,
) -> np.ndarray:
    """Sample a Markov-chain regime path of length ``n_bars``."""
    transition = np.asarray(transition, dtype=float)
    k = transition.shape[0]
    if transition.shape != (k, k):
        raise ValueError("transition must be a square matrix")
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("each row of transition must sum to 1")
    if not 0 <= initial_state < k:
        raise ValueError("initial_state out of range")

    cum = np.cumsum(transition, axis=1)
    draws = rng.random(n_bars)
    states = np.empty(n_bars, dtype=np.int64)
    state = initial_state
    for t in range(n_bars):
        states[t] = state
        state = int(np.searchsorted(cum[state], draws[t], side="right"))
        state = min(state, k - 1)
    return states


def _as_vector(value: float | Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have length {n}")
    return arr


def _correlation_matrix(correlation: float | np.ndarray, n: int) -> np.ndarray:
    if np.ndim(correlation) == 0:
        rho = float(correlation)
        if n > 1 and not -1.0 / (n - 1) < rho < 1.0:
            raise ValueError(f"constant correlation must be in (-1/(n-1), 1); got {rho}")
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        return corr
    corr = np.asarray(correlation, dtype=float)
    if corr.shape != (n, n):
        raise ValueError(f"correlation matrix must be {n}x{n}")
    if not np.allclose(corr, corr.T):
        raise ValueError("correlation matrix must be symmetric")
    return corr


def generate_ohlcv_panel(
    symbols: Sequence[str],
    *,
    start: str | pd.Timestamp = "2018-01-01",
    periods: int = 1000,
    freq: str = "B",
    annual_drift: float | Sequence[float] = 0.08,
    annual_vol: float | Sequence[float] = 0.20,
    correlation: float | np.ndarray = 0.3,
    s0: float | Sequence[float] = 100.0,
    base_volume: float | Sequence[float] = 1_000_000,
    intraday_steps: int = 8,
    regimes: Sequence[Regime] | None = None,
    transition: np.ndarray | None = None,
    periods_per_year: int | None = None,
    seed: int | None = DEFAULT_SETTINGS.seed,
) -> dict[str, pd.DataFrame]:
    """Generate a dict of ``symbol -> OHLCV DataFrame`` from correlated GBM.

    Parameters
    ----------
    symbols:
        Instrument names; one frame is produced per symbol.
    start, periods, freq:
        Define the timestamp index via ``pd.date_range``.
    annual_drift, annual_vol:
        Per-asset (or scalar) annualised drift and volatility of log-prices.
    correlation:
        Scalar constant pairwise correlation or a full ``n x n`` matrix.
    s0:
        Initial price(s).
    base_volume:
        Typical bar volume; actual volume is log-normal and rises with
        absolute return so that high-volatility bars trade more.
    intraday_steps:
        Fine-grid steps per bar used to build realistic high/low values.
    regimes, transition:
        Enable regime switching. ``transition`` is a row-stochastic matrix with one
        row per regime. When omitted, :data:`DEFAULT_TRANSITION` (two states) is
        used, so supply your own matrix if you pass more or fewer than two regimes.
    periods_per_year:
        Override the bars-per-year inferred from ``freq``.
    seed:
        Seed for reproducibility.
    """
    symbols = list(symbols)
    n = len(symbols)
    if n == 0:
        raise ValueError("at least one symbol is required")
    if len(set(symbols)) != n:
        raise ValueError("symbols must be unique")
    if periods < 2:
        raise ValueError("periods must be >= 2")
    if intraday_steps < 1:
        raise ValueError("intraday_steps must be >= 1")

    mu = _as_vector(annual_drift, n, "annual_drift")
    sigma = _as_vector(annual_vol, n, "annual_vol")
    if np.any(sigma <= 0):
        raise ValueError("annual_vol must be positive")
    start_prices = _as_vector(s0, n, "s0")
    if np.any(start_prices <= 0):
        raise ValueError("s0 must be positive")
    volumes = _as_vector(base_volume, n, "base_volume")

    corr = _correlation_matrix(correlation, n)
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix is not positive definite") from exc

    ppy = annualization_factor(freq, periods_per_year)
    m = intraday_steps
    n_fine = periods * m
    dt = 1.0 / (ppy * m)
    rng = np.random.default_rng(seed)

    # Regime path -> per-fine-step drift shifts and vol multipliers.
    if regimes:
        trans = DEFAULT_TRANSITION if transition is None else np.asarray(transition)
        if trans.shape != (len(regimes), len(regimes)):
            raise ValueError("transition shape must match the number of regimes")
        states = simulate_regime_states(periods, trans, rng)
        drift_shift = np.array([r.drift_shift for r in regimes])[states]
        vol_mult = np.array([r.vol_multiplier for r in regimes])[states]
        drift_shift = np.repeat(drift_shift, m)
        vol_mult = np.repeat(vol_mult, m)
    else:
        drift_shift = np.zeros(n_fine)
        vol_mult = np.ones(n_fine)

    mu_t = mu[None, :] + drift_shift[:, None]  # (n_fine, n)
    sigma_t = sigma[None, :] * vol_mult[:, None]  # (n_fine, n)

    z = rng.standard_normal((n_fine, n)) @ chol.T
    log_steps = (mu_t - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * z
    log_path = np.vstack([np.zeros((1, n)), np.cumsum(log_steps, axis=0)])
    prices = start_prices[None, :] * np.exp(log_path)  # (n_fine + 1, n)

    open_ = prices[:-1:m]
    close = prices[m::m]
    body = prices[:-1].reshape(periods, m, n)
    high = np.maximum(body.max(axis=1), close)
    low = np.minimum(body.min(axis=1), close)

    bar_ret = np.abs(np.log(close / open_))
    bar_sigma = sigma[None, :] * np.sqrt(1.0 / ppy)
    activity = 1.0 + 3.0 * bar_ret / bar_sigma
    noise = np.exp(0.25 * rng.standard_normal((periods, n)))
    volume = np.maximum(1, np.rint(volumes[None, :] * activity * noise)).astype("int64")

    index = pd.date_range(start=start, periods=periods, freq=freq, name="date")
    panel: dict[str, pd.DataFrame] = {}
    for j, sym in enumerate(symbols):
        panel[sym] = pd.DataFrame(
            {
                "open": open_[:, j],
                "high": high[:, j],
                "low": low[:, j],
                "close": close[:, j],
                "volume": volume[:, j],
            },
            index=index,
        )
    return panel
