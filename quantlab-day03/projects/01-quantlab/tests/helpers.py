"""Shared test utilities."""

from __future__ import annotations

import pandas as pd

from quantlab.strategy import Strategy


def weight_path(signals, symbol: str, index: pd.Index) -> pd.Series:
    """Reconstruct the weight a strategy held on every bar from its signals."""
    emitted = pd.Series(
        {s.timestamp: s.target_weight for s in signals if s.symbol == symbol}, dtype="float64"
    )
    return emitted.reindex(index).ffill().fillna(0.0)


class Scripted(Strategy):
    """Emits whatever ``script[bar_number]`` says (bar numbers start at 1)."""

    def __init__(self, symbols, script, **kwargs):
        self.script = script
        super().__init__(symbols, **kwargs)

    def compute_targets(self, event, feed):
        return self.script.get(self._bars, {})
