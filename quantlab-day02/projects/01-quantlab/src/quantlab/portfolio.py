"""Position accounting, order sizing and equity tracking.

The portfolio is the only component that knows about money. It

* turns strategy :class:`SignalEvent` target weights into :class:`OrderEvent` quantities,
* applies :class:`FillEvent` cash flows and updates positions with average-cost accounting,
* marks positions to market on every :class:`MarketEvent` and records the equity curve.

Timing convention: a signal generated on bar *t* is sized with bar *t*'s close and
(with the Day 4 execution handler) filled on bar *t+1*, so decisions never use the
price they are executed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.events import FillEvent, MarketEvent, OrderEvent, SignalEvent

EPS = 1e-9


@dataclass(slots=True)
class Position:
    """Average-cost position in one instrument. Negative quantity means short."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < EPS

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return 0.0 if self.is_flat else self.quantity * (price - self.avg_price)

    def apply_fill(self, quantity: float, price: float) -> float:
        """Apply a signed trade and return the gross P&L it realised.

        * Adding to a position re-averages the cost basis.
        * Reducing a position realises P&L on the closed part at the old basis.
        * Crossing through zero closes the old side and opens the remainder at ``price``.
        """
        if abs(quantity) < EPS:
            return 0.0

        old = self.quantity
        if self.is_flat or old * quantity > 0:
            new = old + quantity
            self.avg_price = (self.avg_price * abs(old) + price * abs(quantity)) / abs(new)
            self.quantity = new
            return 0.0

        closing = min(abs(quantity), abs(old))
        side = 1.0 if old > 0 else -1.0
        realized = closing * (price - self.avg_price) * side

        new = old + quantity
        if abs(new) < EPS:
            self.quantity = 0.0
            self.avg_price = 0.0
        elif new * old < 0:  # flipped through zero
            self.quantity = new
            self.avg_price = price
        else:  # partial close keeps its cost basis
            self.quantity = new

        self.realized_pnl += realized
        return realized


@dataclass(slots=True)
class _OpenOrder:
    symbol: str
    remaining: float


class Portfolio:
    """Cash, positions, risk caps and the recorded equity curve.

    Parameters
    ----------
    initial_cash:
        Starting capital.
    allow_fractional:
        If False (default) order sizes are rounded *toward zero* to whole shares,
        so the portfolio never over-shoots a target.
    max_position_weight:
        Cap on ``|target weight|`` for any single instrument.
    max_gross_leverage:
        Cap on the sum of ``|target weight|`` across instruments. Targets are
        granted first-come first-served: a new signal is clipped to whatever
        capacity the other standing targets leave.
    min_trade_fraction:
        Skip rebalancing trades smaller than this fraction of equity (reduces
        churn). Trades that fully close a position are always allowed.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        *,
        allow_fractional: bool = False,
        max_position_weight: float = 1.0,
        max_gross_leverage: float = 1.0,
        min_trade_fraction: float = 0.0,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if max_position_weight <= 0 or max_gross_leverage <= 0:
            raise ValueError("weight and leverage caps must be positive")
        if min_trade_fraction < 0:
            raise ValueError("min_trade_fraction must be non-negative")

        self.initial_cash = float(initial_cash)
        self.allow_fractional = allow_fractional
        self.max_position_weight = max_position_weight
        self.max_gross_leverage = max_gross_leverage
        self.min_trade_fraction = min_trade_fraction

        self._cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}
        self._targets: dict[str, float] = {}
        self._open: dict[int, _OpenOrder] = {}
        self._next_order_id = 1
        self._trades: list[dict[str, object]] = []
        self._records: list[tuple[pd.Timestamp, float, float, float, float, float]] = []
        self.total_commission = 0.0

    # ------------------------------------------------------------- inspection

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Non-flat positions (a copy of the mapping; Position objects are live)."""
        return {s: p for s, p in self._positions.items() if not p.is_flat}

    def position_quantity(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        return pos.quantity if pos else 0.0

    def pending_quantity(self, symbol: str) -> float:
        """Signed quantity of working orders that have not filled yet."""
        return sum(o.remaining for o in self._open.values() if o.symbol == symbol)

    def last_price(self, symbol: str) -> float:
        try:
            return self._prices[symbol]
        except KeyError:
            raise KeyError(f"no market price known for {symbol!r} yet") from None

    def _price_for(self, pos: Position) -> float:
        return self._prices.get(pos.symbol, pos.avg_price)

    @property
    def market_value(self) -> float:
        return sum(p.market_value(self._price_for(p)) for p in self._positions.values())

    @property
    def equity(self) -> float:
        return self._cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value(self._price_for(p))) for p in self._positions.values())

    @property
    def net_exposure(self) -> float:
        return self.market_value

    @property
    def realized_pnl(self) -> float:
        """Gross realised P&L (before commissions)."""
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl(self._price_for(p)) for p in self._positions.values())

    def weights(self) -> dict[str, float]:
        """Current position values as a fraction of equity."""
        eq = self.equity
        if eq <= 0:
            return {}
        return {
            s: p.market_value(self._price_for(p)) / eq
            for s, p in self._positions.items()
            if not p.is_flat
        }

    @property
    def target_weights(self) -> dict[str, float]:
        return dict(self._targets)

    @property
    def open_orders(self) -> dict[int, float]:
        """``{order_id: remaining signed quantity}`` for working orders."""
        return {oid: o.remaining for oid, o in self._open.items()}

    # ---------------------------------------------------------------- handlers

    def on_market(self, event: MarketEvent) -> None:
        """Mark to market at this bar's close and record an equity snapshot."""
        for symbol, bar in event.bars.items():
            self._prices[symbol] = bar.close
        self._records.append(
            (
                event.timestamp,
                self._cash,
                self.market_value,
                self.equity,
                self.gross_exposure,
                self.net_exposure,
            )
        )

    def on_signal(self, signal: SignalEvent) -> OrderEvent | None:
        """Convert a target weight into the order needed to reach it (or ``None``)."""
        symbol = signal.symbol
        price = self.last_price(symbol)

        cap = self.max_position_weight
        weight = float(np.clip(signal.target_weight, -cap, cap))
        others = sum(abs(w) for s, w in self._targets.items() if s != symbol)
        capacity = max(0.0, self.max_gross_leverage - others)
        weight = float(np.clip(weight, -capacity, capacity))
        if abs(weight) < 1e-12:
            weight = 0.0  # normalise -0.0
        self._targets[symbol] = weight

        equity = self.equity
        if equity <= 0:
            return None

        target_qty = self._round_quantity(weight * equity / price)
        effective = self.position_quantity(symbol) + self.pending_quantity(symbol)
        delta = target_qty - effective
        if abs(delta) < EPS:
            return None
        if target_qty != 0 and abs(delta) * price / equity < self.min_trade_fraction:
            return None

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            quantity=delta,
            order_id=self._next_order_id,
            strategy_id=signal.strategy_id,
        )
        self._next_order_id += 1
        self._open[order.order_id] = _OpenOrder(symbol, delta)
        return order

    def on_fill(self, fill: FillEvent) -> float:
        """Apply an execution and return the gross P&L it realised."""
        pos = self._positions.setdefault(fill.symbol, Position(fill.symbol))
        realized = pos.apply_fill(fill.quantity, fill.price)
        self._cash += fill.cash_flow
        self.total_commission += fill.commission
        self._prices.setdefault(fill.symbol, fill.price)

        if fill.order_id in self._open:
            working = self._open[fill.order_id]
            working.remaining -= fill.quantity
            if abs(working.remaining) < EPS or working.remaining * fill.quantity < 0:
                del self._open[fill.order_id]

        self._trades.append(
            {
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.commission,
                "realized_pnl": realized,
                "order_id": fill.order_id,
                "strategy_id": fill.strategy_id,
            }
        )
        return realized

    def cancel_order(self, order_id: int) -> bool:
        """Forget a working order (e.g. expired or rejected). Returns True if it existed."""
        return self._open.pop(order_id, None) is not None

    # ----------------------------------------------------------------- outputs

    def equity_curve(self) -> pd.Series:
        """Equity at each recorded market event."""
        return self.history_frame()["equity"]

    def history_frame(self) -> pd.DataFrame:
        cols = ["cash", "market_value", "equity", "gross_exposure", "net_exposure"]
        if not self._records:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
        frame = pd.DataFrame(self._records, columns=["date", *cols]).set_index("date")
        return frame

    def trades_frame(self) -> pd.DataFrame:
        cols = [
            "timestamp",
            "symbol",
            "quantity",
            "price",
            "commission",
            "realized_pnl",
            "order_id",
            "strategy_id",
        ]
        return pd.DataFrame(self._trades, columns=cols)

    def summary(self) -> dict[str, float]:
        equity = self.equity
        return {
            "initial_cash": self.initial_cash,
            "final_equity": equity,
            "total_return": equity / self.initial_cash - 1.0,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_commission": self.total_commission,
            "n_trades": float(len(self._trades)),
        }

    # ----------------------------------------------------------------- helpers

    def _round_quantity(self, quantity: float) -> float:
        if self.allow_fractional:
            return quantity
        return float(math.trunc(quantity))
