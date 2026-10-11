"""Trading strategies. Importing this package registers all built-in strategies."""

from quantlab.strategy.allocation import VolTargetAllocation
from quantlab.strategy.base import Strategy, Target
from quantlab.strategy.mean_reversion import ZScoreMeanReversion
from quantlab.strategy.registry import (
    available_strategies,
    create_strategy,
    get_strategy,
    register_strategy,
)
from quantlab.strategy.trend import SMACrossover, TimeSeriesMomentum

__all__ = [
    "SMACrossover",
    "Strategy",
    "Target",
    "TimeSeriesMomentum",
    "VolTargetAllocation",
    "ZScoreMeanReversion",
    "available_strategies",
    "create_strategy",
    "get_strategy",
    "register_strategy",
]
