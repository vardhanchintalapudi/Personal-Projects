"""Name-based strategy registry.

Lets configs and the Day 7 CLI refer to strategies by string
(``"sma_crossover"``) without importing their classes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from quantlab.strategy.base import Strategy

S = TypeVar("S", bound=type[Strategy])

_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(name: str) -> Callable[[S], S]:
    """Class decorator that registers a strategy under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("strategy name must be non-empty")

    def decorator(cls: S) -> S:
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"strategy name {key!r} already registered by {existing.__name__}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def get_strategy(name: str) -> type[Strategy]:
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        options = ", ".join(available_strategies()) or "none registered"
        raise KeyError(f"unknown strategy {name!r}; available: {options}") from None


def create_strategy(name: str, symbols: Sequence[str], **params: Any) -> Strategy:
    """Instantiate a registered strategy: ``create_strategy("tsmom", ["A", "B"], lookback=60)``."""
    return get_strategy(name)(symbols, **params)
