"""Portfolio-level allocation strategies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from quantlab.events import MarketEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.strategy.base import Strategy, Target
from quantlab.strategy.registry import register_strategy


@register_strategy("vol_target")
class VolTargetAllocation(Strategy):
    """Long-only inverse-volatility portfolio scaled to a target volatility.

    Every ``rebalance_every`` bars:

    1. estimate the annualised covariance matrix ``S`` from the last
       ``lookback`` simple returns;
    2. take inverse-volatility weights ``w_i ~ 1 / sigma_i`` normalised to sum
       to one, so calmer instruments get more capital;
    3. compute the portfolio volatility ``sqrt(w' S w)`` and scale the weights
       by ``target_vol / portfolio_vol``, capped at ``max_leverage`` of gross
       exposure.

    When the target is above what the instruments can deliver unlevered
    (``scale > 1``), exposure is limited by ``max_leverage``; when it is below,
    the portfolio holds the remainder in cash. The estimate is backward
    looking, so realised volatility will drift from the target when regimes
    change, which is the point of re-estimating regularly.

    Parameters
    ----------
    target_vol:
        Annualised volatility target, e.g. ``0.10`` for 10 %.
    lookback:
        Number of returns used for the covariance estimate (``>= 5``).
    rebalance_every:
        Bars between re-estimations.
    max_leverage:
        Cap on the sum of weights (gross exposure as a fraction of equity).
    periods_per_year:
        Annualisation factor for the bar size.
    rebalance_band:
        Skip a rebalance unless some weight moved by more than this (default
        0.02, two percentage points). It is applied to the whole weight vector,
        so the held weights are always one complete, internally consistent
        target rather than a blend of old and new ones.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        target_vol: float = 0.10,
        lookback: int = 60,
        rebalance_every: int = 5,
        max_leverage: float = 1.0,
        periods_per_year: int = 252,
        rebalance_band: float = 0.02,
        **kwargs,
    ) -> None:
        if target_vol <= 0 or max_leverage <= 0 or periods_per_year <= 0:
            raise ValueError("target_vol, max_leverage and periods_per_year must be positive")
        if rebalance_band < 0:
            raise ValueError("rebalance_band must be non-negative")
        if lookback < 5 or rebalance_every < 1:
            raise ValueError("require lookback >= 5 and rebalance_every >= 1")
        self.target_vol = target_vol
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.max_leverage = max_leverage
        self.periods_per_year = periods_per_year
        self.rebalance_band = rebalance_band
        self.min_history = lookback + 1
        self._next_eval = 0
        super().__init__(
            symbols,
            params={
                "target_vol": target_vol,
                "lookback": lookback,
                "rebalance_every": rebalance_every,
                "max_leverage": max_leverage,
                "periods_per_year": periods_per_year,
                "rebalance_band": rebalance_band,
            },
            **kwargs,
        )

    def reset_state(self) -> None:
        self._next_eval = 0

    def compute_targets(
        self, event: MarketEvent, feed: HistoricalDataFeed
    ) -> dict[str, Target]:
        if self._bars < self._next_eval:
            return {}

        panel = feed.history_panel(self.symbols, self.lookback + 1, "close", how="inner")
        if len(panel) < self.lookback + 1:
            return {}

        weights = self.allocate(panel.pct_change().dropna().to_numpy())
        if weights is None:
            return {}

        self._next_eval = self._bars + self.rebalance_every

        held = np.array([self._last_weight.get(s, 0.0) for s in self.symbols])
        if np.max(np.abs(weights - held)) <= self.rebalance_band:
            return {}

        scale = float(weights.sum())
        return {
            symbol: Target(float(w), f"vol-target scale {scale:.2f}")
            for symbol, w in zip(self.symbols, weights, strict=True)
        }

    def allocate(self, returns: np.ndarray) -> np.ndarray | None:
        """Weights for a ``(n_obs, n_assets)`` matrix of simple returns.

        Returns ``None`` when the estimate is unusable (an instrument with zero
        volatility, or a degenerate portfolio).
        """
        if returns.ndim != 2 or returns.shape[0] < 2:
            return None
        cov = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * self.periods_per_year
        vols = np.sqrt(np.diag(cov))
        if np.any(vols <= 1e-12):
            return None

        inverse_vol = 1.0 / vols
        base = inverse_vol / inverse_vol.sum()
        portfolio_vol = float(np.sqrt(base @ cov @ base))
        if portfolio_vol <= 1e-12:
            return None

        scale = min(self.target_vol / portfolio_vol, self.max_leverage)
        return base * scale
