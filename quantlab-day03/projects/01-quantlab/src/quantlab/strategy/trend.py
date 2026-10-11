"""Trend-following strategies: moving-average crossover and time-series momentum."""

from __future__ import annotations

from collections.abc import Sequence

from quantlab.events import MarketEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.strategy.base import Strategy, Target
from quantlab.strategy.registry import register_strategy


@register_strategy("sma_crossover")
class SMACrossover(Strategy):
    """Hold each instrument while its fast moving average is above the slow one.

    Parameters
    ----------
    fast, slow:
        Window lengths in bars, ``1 <= fast < slow``.
    weight:
        Weight per instrument when active. Defaults to ``1 / len(symbols)``.
    allow_short:
        Go short (``-weight``) while the fast average is below the slow one;
        otherwise stay flat.

    Signals are evaluated on the close of each bar, so the earliest possible
    first signal is on bar number ``slow``.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        fast: int = 20,
        slow: int = 50,
        *,
        weight: float | None = None,
        allow_short: bool = False,
        **kwargs,
    ) -> None:
        if fast < 1 or slow <= fast:
            raise ValueError("require 1 <= fast < slow")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.weight = self.equal_weight(symbols, weight)
        self.min_history = slow
        super().__init__(
            symbols,
            params={"fast": fast, "slow": slow, "weight": self.weight, "allow_short": allow_short},
            **kwargs,
        )

    def compute_targets(
        self, event: MarketEvent, feed: HistoricalDataFeed
    ) -> dict[str, Target]:
        targets: dict[str, Target] = {}
        for symbol in self.active_symbols(event):
            closes = feed.window(symbol, self.slow)
            if closes is None:
                continue
            fast_ma = closes[-self.fast :].mean()
            slow_ma = closes.mean()
            if fast_ma > slow_ma:
                weight = self.weight
            elif fast_ma < slow_ma:
                weight = -self.weight if self.allow_short else 0.0
            else:
                continue  # exact tie: keep the previous position
            reason = f"SMA{self.fast} {fast_ma:.4f} vs SMA{self.slow} {slow_ma:.4f}"
            targets[symbol] = Target(weight, reason)
        return targets


@register_strategy("tsmom")
class TimeSeriesMomentum(Strategy):
    """Trade the sign of each instrument's own trailing return.

    The return is measured over ``lookback`` bars ending ``skip`` bars ago (the
    classic "12-1" construction uses ``lookback=252-21, skip=21`` to avoid
    short-term reversal). Positions are only reconsidered every
    ``rebalance_every`` bars, starting on the first bar with enough history.

    Parameters
    ----------
    lookback, skip:
        Measurement window and the number of most recent bars to ignore.
    rebalance_every:
        Bars between decisions.
    threshold:
        Minimum absolute trailing return to take a position; weaker trends
        are flat.
    weight, allow_short:
        As for :class:`SMACrossover`; shorting is on by default here.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        lookback: int = 126,
        skip: int = 0,
        *,
        rebalance_every: int = 21,
        threshold: float = 0.0,
        weight: float | None = None,
        allow_short: bool = True,
        **kwargs,
    ) -> None:
        if lookback < 1 or skip < 0 or rebalance_every < 1 or threshold < 0:
            raise ValueError("invalid lookback / skip / rebalance_every / threshold")
        self.lookback = lookback
        self.skip = skip
        self.rebalance_every = rebalance_every
        self.threshold = threshold
        self.allow_short = allow_short
        self.weight = self.equal_weight(symbols, weight)
        self.min_history = lookback + skip + 1
        self._next_eval = 0
        super().__init__(
            symbols,
            params={
                "lookback": lookback,
                "skip": skip,
                "rebalance_every": rebalance_every,
                "threshold": threshold,
                "weight": self.weight,
                "allow_short": allow_short,
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

        targets: dict[str, Target] = {}
        for symbol in self.active_symbols(event):
            closes = feed.window(symbol, self.min_history)
            if closes is None:
                continue
            trailing = closes[-1 - self.skip] / closes[0] - 1.0
            if trailing > self.threshold:
                weight = self.weight
            elif trailing < -self.threshold:
                weight = -self.weight if self.allow_short else 0.0
            else:
                weight = 0.0
            targets[symbol] = Target(weight, f"{self.lookback}-bar return {trailing:+.2%}")

        if targets:
            self._next_eval = self._bars + self.rebalance_every
        return targets
