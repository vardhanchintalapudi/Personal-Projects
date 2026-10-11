"""Strategy interface.

A strategy answers one question each bar: *what exposure do I want?* It never
sizes orders, sees cash or touches the portfolio. Subclasses implement
:meth:`Strategy.compute_targets` and return desired weights; the base class

* skips targets that equal what was last emitted (no signal spam),
* orders signals so exposure-reducing ones go first (so a leverage cap in the
  portfolio never blocks a rebalance that frees capacity), and
* stamps every signal with the bar's timestamp and the strategy id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantlab.events import MarketEvent, SignalEvent
from quantlab.feed import HistoricalDataFeed


@dataclass(frozen=True, slots=True)
class Target:
    """A desired weight plus a short human-readable reason (kept in the signal)."""

    weight: float
    reason: str = ""


class Strategy(ABC):
    """Base class for all strategies.

    Parameters
    ----------
    symbols:
        Instruments the strategy trades. Must exist in the feed it is run on.
    strategy_id:
        Label attached to signals, orders and fills. Defaults to the class name.
    weight_tolerance:
        A new target is only emitted when it differs from the last emitted
        weight by more than this. Use it to avoid churning on tiny changes.
    params:
        Constructor parameters, kept for logging and reproducible reports.
    """

    #: Bars of history needed before the first signal is possible.
    min_history: int = 1

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        strategy_id: str | None = None,
        weight_tolerance: float = 1e-6,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        symbols = tuple(symbols)
        if not symbols:
            raise ValueError("a strategy needs at least one symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        if weight_tolerance < 0:
            raise ValueError("weight_tolerance must be non-negative")

        self.symbols = symbols
        self.strategy_id = strategy_id or type(self).__name__
        self.weight_tolerance = weight_tolerance
        self.params: dict[str, Any] = dict(params or {})
        self._last_weight: dict[str, float] = {}
        self._bars = 0

    # ------------------------------------------------------------- lifecycle

    def on_start(self, feed: HistoricalDataFeed) -> None:
        """Called once before the first bar. Validates symbols and clears state."""
        missing = [s for s in self.symbols if s not in feed.symbols]
        if missing:
            raise KeyError(f"symbols not in feed: {missing}")
        self._last_weight.clear()
        self._bars = 0
        self.reset_state()

    def reset_state(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Subclass hook: clear any indicator or position state."""

    def on_bar(self, event: MarketEvent, feed: HistoricalDataFeed) -> list[SignalEvent]:
        """Process one :class:`MarketEvent` and return the signals it triggers."""
        self._bars += 1
        raw = self.compute_targets(event, feed)

        changes: list[tuple[str, float, str, float]] = []
        for symbol, target in raw.items():
            if isinstance(target, Target):
                weight, reason = target.weight, target.reason
            else:
                weight, reason = float(target), ""
            last = self._last_weight.get(symbol, 0.0)
            if abs(weight - last) > self.weight_tolerance:
                changes.append((symbol, weight, reason, abs(weight) - abs(last)))

        # Exposure-reducing changes first: they free capacity for the increases.
        changes.sort(key=lambda c: c[3])

        signals: list[SignalEvent] = []
        for symbol, weight, reason, _ in changes:
            self._last_weight[symbol] = weight
            signals.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    target_weight=weight,
                    strategy_id=self.strategy_id,
                    reason=reason,
                )
            )
        return signals

    @abstractmethod
    def compute_targets(
        self, event: MarketEvent, feed: HistoricalDataFeed
    ) -> Mapping[str, Target | float]:
        """Return ``{symbol: desired weight}`` for any symbols to (re)decide this bar.

        Symbols that are omitted keep their previous target.
        """

    # --------------------------------------------------------------- helpers

    @property
    def current_weights(self) -> dict[str, float]:
        """The last weight emitted for each symbol."""
        return dict(self._last_weight)

    def active_symbols(self, event: MarketEvent) -> tuple[str, ...]:
        """Strategy symbols that actually have a bar in this event."""
        return tuple(s for s in self.symbols if s in event.bars)

    @staticmethod
    def equal_weight(symbols: Sequence[str], weight: float | None) -> float:
        """Per-symbol weight: the given one, or ``1 / n`` when omitted."""
        if weight is None:
            return 1.0 / len(symbols)
        if weight <= 0:
            raise ValueError("weight must be positive")
        return float(weight)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}(symbols={list(self.symbols)!r}{', ' if args else ''}{args})"
