"""Execution simulation: costs, order types and partial fills.

An :class:`ExecutionHandler` holds working orders and, on every new bar, decides
what trades. Costs are layered so each can be studied on its own:

* **commission** - what the broker charges (:class:`CommissionModel`),
* **slippage** - a spread-like adverse price move on market/stop orders (:class:`SlippageModel`),
* **market impact** - an adverse move that grows with order size relative to
  typical volume (:class:`ImpactModel`, square-root law by default).

Order types
-----------
``MARKET``  fills at the bar's open.
``LIMIT``   buy fills if the bar trades at or below the limit (at the open if it
            gapped through, else at the limit); sells mirror that. No slippage.
``STOP``    a stop-market order: triggers when the bar trades through the stop
            (at the open if it gapped through, else at the stop) and then pays
            slippage and impact like a market order.

Orders are evaluated on bars *after* the one on which they were submitted, so a
decision made on a close is never filled at that same close.

Intrabar order of events is unknowable from OHLC data, so each bar is treated
independently: a stop or limit is checked against the whole bar's range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from quantlab.events import Bar, FillEvent, MarketEvent, OrderEvent, OrderType
from quantlab.feed import HistoricalDataFeed

_EPS = 1e-9

# --------------------------------------------------------------------- commission


class CommissionModel(Protocol):
    def __call__(self, quantity: float, price: float) -> float:
        """Commission in cash for trading ``quantity`` (signed) at ``price``."""


@dataclass(frozen=True)
class ZeroCommission:
    def __call__(self, quantity: float, price: float) -> float:
        return 0.0


@dataclass(frozen=True)
class BpsCommission:
    """Proportional commission, ``bps`` basis points of traded notional."""

    bps: float = 1.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("bps must be non-negative")

    def __call__(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.bps / 1e4


@dataclass(frozen=True)
class PerShareCommission:
    """Per-share commission with a minimum ticket and a cap as a share of notional.

    Defaults resemble a US retail-pro schedule: $0.005/share, $1 minimum,
    never more than 1 % of the trade's value.
    """

    per_share: float = 0.005
    minimum: float = 1.0
    max_fraction_of_notional: float = 0.01

    def __post_init__(self) -> None:
        if min(self.per_share, self.minimum, self.max_fraction_of_notional) < 0:
            raise ValueError("commission parameters must be non-negative")

    def __call__(self, quantity: float, price: float) -> float:
        notional = abs(quantity) * price
        fee = max(abs(quantity) * self.per_share, self.minimum)
        return min(fee, notional * self.max_fraction_of_notional)


# ---------------------------------------------------------------------- slippage


class SlippageModel(Protocol):
    def fraction(self, order: OrderEvent, bar: Bar) -> float:
        """Adverse price move as a fraction of the reference price (``>= 0``)."""


@dataclass(frozen=True)
class NoSlippage:
    def fraction(self, order: OrderEvent, bar: Bar) -> float:
        return 0.0


@dataclass(frozen=True)
class FixedBpsSlippage:
    """A constant adverse move of ``bps`` basis points (think: half the spread)."""

    bps: float = 1.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("bps must be non-negative")

    def fraction(self, order: OrderEvent, bar: Bar) -> float:
        return self.bps / 1e4


@dataclass(frozen=True)
class BarRangeSlippage:
    """Slippage proportional to the bar's own range: wild bars cost more."""

    fraction_of_range: float = 0.05

    def __post_init__(self) -> None:
        if self.fraction_of_range < 0:
            raise ValueError("fraction_of_range must be non-negative")

    def fraction(self, order: OrderEvent, bar: Bar) -> float:
        if bar.open <= 0:
            return 0.0
        return self.fraction_of_range * max(bar.high - bar.low, 0.0) / bar.open


# ------------------------------------------------------------------------ impact


class ImpactModel(Protocol):
    def fraction(self, symbol: str, quantity: float, feed: HistoricalDataFeed) -> float:
        """Adverse price move as a fraction of the reference price (``>= 0``)."""


@dataclass(frozen=True)
class NoImpact:
    def fraction(self, symbol: str, quantity: float, feed: HistoricalDataFeed) -> float:
        return 0.0


@dataclass(frozen=True)
class SquareRootImpact:
    """The square-root market-impact law.

    ``impact = coefficient * sigma * sqrt(|quantity| / ADV)``

    where ``sigma`` is the per-bar volatility of returns and ``ADV`` the average
    bar volume, both measured over the ``lookback`` bars *before* the bar being
    filled (so the model cannot see the current bar's volume or range). With no
    usable history the impact is zero.

    ``max_fraction`` caps the result so an absurdly large order cannot produce a
    nonsensical price.
    """

    coefficient: float = 0.5
    lookback: int = 20
    min_history: int = 5
    max_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.coefficient < 0 or self.lookback < 2 or self.min_history < 3:
            raise ValueError("invalid impact parameters")
        if not 0 < self.max_fraction < 1:
            raise ValueError("max_fraction must be in (0, 1)")

    def fraction(self, symbol: str, quantity: float, feed: HistoricalDataFeed) -> float:
        frame = feed.history_frame(symbol, self.lookback + 1).iloc[:-1]  # drop the current bar
        if len(frame) < self.min_history:
            return 0.0
        returns = frame["close"].pct_change().dropna()
        adv = float(frame["volume"].mean())
        if adv <= 0 or len(returns) < 2:
            return 0.0
        sigma = float(returns.std(ddof=1))
        if not np.isfinite(sigma):
            return 0.0
        value = self.coefficient * sigma * math.sqrt(abs(quantity) / adv)
        return min(value, self.max_fraction)


# --------------------------------------------------------------------- handler


@dataclass(frozen=True)
class ExecutionReport:
    """What happened on one bar."""

    fills: tuple[FillEvent, ...] = ()
    expired: tuple[OrderEvent, ...] = ()


@dataclass(slots=True)
class _Working:
    order: OrderEvent
    remaining: float
    age: int = 0


class ExecutionHandler:
    """Simulated broker.

    Parameters
    ----------
    commission, slippage, impact:
        Cost models. Defaults are all zero so results are easy to reason about;
        use :meth:`realistic` for sensible non-zero defaults.
    max_participation:
        Largest share of a bar's volume that this account may trade in one
        symbol per bar (all working orders in the symbol share it). Orders larger
        than that fill partially over several bars. ``None`` disables the limit.
    max_order_age:
        Cancel an order after this many bars on which its symbol traded (the
        unfilled remainder is reported as expired). ``None`` keeps orders until
        filled or cancelled.
    """

    def __init__(
        self,
        commission: CommissionModel | None = None,
        slippage: SlippageModel | None = None,
        impact: ImpactModel | None = None,
        *,
        max_participation: float | None = 0.10,
        max_order_age: int | None = None,
    ) -> None:
        if max_participation is not None and not 0 < max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1] or None")
        if max_order_age is not None and max_order_age < 1:
            raise ValueError("max_order_age must be >= 1 or None")
        self.commission: CommissionModel = commission or ZeroCommission()
        self.slippage: SlippageModel = slippage or NoSlippage()
        self.impact: ImpactModel = impact or NoImpact()
        self.max_participation = max_participation
        self.max_order_age = max_order_age
        self._working: dict[int, _Working] = {}
        self._log: list[dict[str, object]] = []

    @classmethod
    def realistic(cls, **overrides) -> ExecutionHandler:
        """Per-share commission, 1 bp slippage, square-root impact, 10 % participation."""
        settings = {
            "commission": PerShareCommission(),
            "slippage": FixedBpsSlippage(1.0),
            "impact": SquareRootImpact(),
        }
        settings.update(overrides)
        return cls(**settings)

    # ------------------------------------------------------------ order flow

    def submit(self, order: OrderEvent) -> None:
        if order.order_id in self._working:
            raise ValueError(f"order {order.order_id} is already working")
        self._working[order.order_id] = _Working(order, order.quantity)

    def cancel(self, order_id: int) -> bool:
        return self._working.pop(order_id, None) is not None

    @property
    def working_orders(self) -> dict[int, float]:
        """``{order_id: remaining signed quantity}``."""
        return {oid: w.remaining for oid, w in self._working.items()}

    def on_market(self, event: MarketEvent, feed: HistoricalDataFeed) -> ExecutionReport:
        """Try to fill every working order against the new bar."""
        fills: list[FillEvent] = []
        expired: list[OrderEvent] = []
        used: dict[str, float] = {}

        for order_id in list(self._working):
            work = self._working[order_id]
            bar = event.bars.get(work.order.symbol)
            if bar is None:
                continue  # symbol did not trade: nothing happens, no ageing

            work.age += 1
            fill = self._try_fill(work, bar, event.timestamp, feed, used)
            if fill is not None:
                fills.append(fill)
                work.remaining -= fill.quantity
                if abs(work.remaining) < _EPS:
                    del self._working[order_id]
                    continue

            if self.max_order_age is not None and work.age >= self.max_order_age:
                expired.append(work.order)
                del self._working[order_id]

        return ExecutionReport(tuple(fills), tuple(expired))

    # ------------------------------------------------------------ analytics

    def costs_frame(self) -> pd.DataFrame:
        """One row per fill with the reference price and each cost component in cash."""
        cols = [
            "timestamp",
            "symbol",
            "order_id",
            "quantity",
            "reference_price",
            "fill_price",
            "participation",
            "commission",
            "slippage_cost",
            "impact_cost",
            "total_cost",
        ]
        frame = pd.DataFrame(self._log, columns=cols[:-1])
        frame["total_cost"] = frame["commission"] + frame["slippage_cost"] + frame["impact_cost"]
        return frame

    # -------------------------------------------------------------- internals

    @staticmethod
    def _reference_price(order: OrderEvent, bar: Bar, side: float) -> float | None:
        """Price before slippage/impact, or ``None`` if the order does not trade this bar."""
        if order.order_type is OrderType.MARKET:
            return bar.open
        if order.order_type is OrderType.LIMIT:
            limit = float(order.limit_price)  # validated non-None for limit orders
            if side > 0:
                if bar.open <= limit:
                    return bar.open
                return limit if bar.low <= limit else None
            if bar.open >= limit:
                return bar.open
            return limit if bar.high >= limit else None
        stop = float(order.stop_price)  # STOP
        if side > 0:
            return max(bar.open, stop) if bar.high >= stop else None
        return min(bar.open, stop) if bar.low <= stop else None

    def _try_fill(
        self,
        work: _Working,
        bar: Bar,
        timestamp: pd.Timestamp,
        feed: HistoricalDataFeed,
        used: dict[str, float],
    ) -> FillEvent | None:
        order = work.order
        side = 1.0 if work.remaining > 0 else -1.0
        reference = self._reference_price(order, bar, side)
        if reference is None:
            return None

        quantity_abs = abs(work.remaining)
        if self.max_participation is not None:
            available = max(0.0, bar.volume * self.max_participation - used.get(order.symbol, 0.0))
            quantity_abs = min(quantity_abs, available)
        if float(order.quantity).is_integer():  # whole-share orders fill in whole shares
            quantity_abs = math.floor(quantity_abs + _EPS)
        if quantity_abs < _EPS:
            return None
        quantity = side * quantity_abs

        slip = impact = 0.0
        if order.order_type is not OrderType.LIMIT:
            slip = self.slippage.fraction(order, bar)
            impact = self.impact.fraction(order.symbol, quantity, feed)
        adverse = min(slip + impact, 0.9)
        price = reference * (1.0 + side * adverse)
        commission = self.commission(quantity, price)

        used[order.symbol] = used.get(order.symbol, 0.0) + quantity_abs
        self._log.append(
            {
                "timestamp": timestamp,
                "symbol": order.symbol,
                "order_id": order.order_id,
                "quantity": quantity,
                "reference_price": reference,
                "fill_price": price,
                "participation": quantity_abs / bar.volume if bar.volume > 0 else math.nan,
                "commission": commission,
                "slippage_cost": quantity_abs * reference * slip,
                "impact_cost": quantity_abs * reference * impact,
            }
        )
        return FillEvent(
            timestamp=timestamp,
            symbol=order.symbol,
            quantity=quantity,
            price=price,
            commission=commission,
            order_id=order.order_id,
            strategy_id=order.strategy_id,
        )
