"""Mean-reversion strategy based on rolling z-scores."""

from __future__ import annotations

from collections.abc import Sequence

from quantlab.events import MarketEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.strategy.base import Strategy, Target
from quantlab.strategy.registry import register_strategy


@register_strategy("mean_reversion")
class ZScoreMeanReversion(Strategy):
    """Fade stretched prices and exit as they snap back.

    For each instrument the z-score of the latest close against its trailing
    ``lookback`` closes is tracked with a three-state machine::

        flat  --z <= -entry_z-->  long   --z >= -exit_z-->  flat
        flat  --z >=  entry_z-->  short  --z <=  exit_z-->  flat

    Because ``exit_z < entry_z`` the machine has hysteresis: once in a trade
    it stays put while the z-score sits between the two thresholds, which
    avoids flip-flopping around a single cut-off.

    Parameters
    ----------
    lookback:
        Window for the rolling mean and standard deviation (``>= 3``).
    entry_z, exit_z:
        Entry and exit thresholds, ``entry_z > exit_z >= 0``.
    weight, allow_short:
        As for the trend strategies; shorting is on by default.

    There is deliberately no stop-loss here: a stretched price that keeps
    stretching stays in the trade. Risk controls arrive with the execution
    layer and are part of why mean reversion needs realistic costs and
    walk-forward testing (Days 4 and 6).
    """

    def __init__(
        self,
        symbols: Sequence[str],
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        *,
        weight: float | None = None,
        allow_short: bool = True,
        **kwargs,
    ) -> None:
        if lookback < 3:
            raise ValueError("lookback must be at least 3")
        if exit_z < 0 or entry_z <= exit_z:
            raise ValueError("require entry_z > exit_z >= 0")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.allow_short = allow_short
        self.weight = self.equal_weight(symbols, weight)
        self.min_history = lookback
        self._state: dict[str, int] = {}
        super().__init__(
            symbols,
            params={
                "lookback": lookback,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "weight": self.weight,
                "allow_short": allow_short,
            },
            **kwargs,
        )

    def reset_state(self) -> None:
        self._state.clear()

    def compute_targets(
        self, event: MarketEvent, feed: HistoricalDataFeed
    ) -> dict[str, Target]:
        targets: dict[str, Target] = {}
        for symbol in self.active_symbols(event):
            closes = feed.window(symbol, self.lookback)
            if closes is None:
                continue
            std = closes.std(ddof=1)
            if std <= 1e-12:
                continue  # flat window: z-score undefined
            z = (closes[-1] - closes.mean()) / std

            previous = self._state.get(symbol, 0)
            state = previous
            if previous == 0:
                if z <= -self.entry_z:
                    state = 1
                elif z >= self.entry_z and self.allow_short:
                    state = -1
            elif previous == 1 and z >= -self.exit_z:
                state = 0
            elif previous == -1 and z <= self.exit_z:
                state = 0

            if state != previous:
                self._state[symbol] = state
                targets[symbol] = Target(state * self.weight, f"z-score {z:+.2f}")
        return targets
