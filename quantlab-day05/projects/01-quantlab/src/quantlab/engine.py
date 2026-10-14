"""The backtest engine: one event loop that wires every component together.

Per bar, in this order::

    1. execution fills working orders against the new bar        -> FillEvents
    2. the queue is drained, so the portfolio sees:
         FillEvent    -> cash and positions update
         MarketEvent  -> mark to market, record equity, strategy runs   -> SignalEvents
         SignalEvent  -> portfolio sizes it                              -> OrderEvents
         OrderEvent   -> execution accepts it (works from the next bar)

Because fills are queued ahead of the market event, the equity recorded for a
bar already includes trades executed at that bar's open, and because new orders
only reach the execution handler after the strategy has run, nothing decided on
a bar can trade before the next one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.analytics import (
    exposure_statistics,
    performance_summary,
    trade_statistics,
    turnover_statistics,
)
from quantlab.events import (
    Event,
    EventQueue,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from quantlab.execution import ExecutionHandler
from quantlab.feed import HistoricalDataFeed
from quantlab.portfolio import Portfolio
from quantlab.strategy.base import Strategy


@dataclass
class BacktestResult:
    """Everything a backtest produced, as plain pandas objects."""

    strategy_name: str
    params: dict[str, Any]
    initial_cash: float
    history: pd.DataFrame  # cash, market_value, equity, gross_exposure, net_exposure per bar
    trades: pd.DataFrame  # one row per fill with realised P&L
    orders: pd.DataFrame  # every order the portfolio created
    signals: pd.DataFrame  # every signal the strategy emitted
    costs: pd.DataFrame  # per-fill commission / slippage / impact
    n_expired: int = 0
    open_orders_at_end: dict[int, float] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.history["equity"]

    @property
    def returns(self) -> pd.Series:
        """Per-bar simple returns of equity; the first is measured against initial cash."""
        equity = self.equity
        return (equity / equity.shift(1).fillna(self.initial_cash) - 1.0).rename("returns")

    @property
    def total_cost(self) -> float:
        return float(self.costs["total_cost"].sum()) if len(self.costs) else 0.0

    def performance(
        self,
        benchmark: pd.Series | None = None,
        *,
        periods_per_year: int = 252,
        risk_free: float = 0.0,
    ) -> pd.Series:
        """Return/risk statistics of the equity curve (see :func:`performance_summary`)."""
        return performance_summary(
            self.returns,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
            benchmark=benchmark,
        )

    def trade_stats(self) -> dict[str, float]:
        """Win rate, payoff ratio, profit factor and friends over closing fills."""
        return trade_statistics(self.trades)

    def turnover(self, periods_per_year: int = 252) -> dict[str, float]:
        return turnover_statistics(self.history, self.trades, periods_per_year)

    def exposure(self) -> dict[str, float]:
        return exposure_statistics(self.history)

    def summary(self) -> dict[str, float]:
        equity = self.equity
        final = float(equity.iloc[-1]) if len(equity) else self.initial_cash
        return {
            "initial_cash": self.initial_cash,
            "final_equity": final,
            "total_return": final / self.initial_cash - 1.0,
            "n_signals": float(len(self.signals)),
            "n_orders": float(len(self.orders)),
            "n_fills": float(len(self.trades)),
            "n_expired_orders": float(self.n_expired),
            "total_commission": float(self.costs["commission"].sum()) if len(self.costs) else 0.0,
            "total_slippage": float(self.costs["slippage_cost"].sum()) if len(self.costs) else 0.0,
            "total_impact": float(self.costs["impact_cost"].sum()) if len(self.costs) else 0.0,
            "total_cost": self.total_cost,
        }


class BacktestEngine:
    """Run one strategy over one market.

    Parameters
    ----------
    market:
        ``{symbol: OHLCV DataFrame}``.
    strategy:
        Any :class:`~quantlab.strategy.Strategy`.
    portfolio:
        Defaults to ``Portfolio(100_000)``.
    execution:
        Defaults to a zero-cost :class:`ExecutionHandler`; pass
        ``ExecutionHandler.realistic()`` for costs.

    An engine instance runs once; build a new one for each backtest.
    """

    def __init__(
        self,
        market: Mapping[str, pd.DataFrame],
        strategy: Strategy,
        portfolio: Portfolio | None = None,
        execution: ExecutionHandler | None = None,
    ) -> None:
        self.market = market
        self.strategy = strategy
        self.portfolio = portfolio if portfolio is not None else Portfolio(100_000.0)
        self.execution = execution if execution is not None else ExecutionHandler()
        self._has_run = False

    def run(self) -> BacktestResult:
        if self._has_run:
            raise RuntimeError("this engine has already run; create a new BacktestEngine")
        self._has_run = True

        feed = HistoricalDataFeed(self.market)
        self.strategy.on_start(feed)

        queue = EventQueue()
        signal_log: list[dict[str, object]] = []
        order_log: list[dict[str, object]] = []
        n_expired = 0

        def on_fill(event: FillEvent) -> None:
            self.portfolio.on_fill(event)

        def on_market(event: MarketEvent) -> None:
            self.portfolio.on_market(event)
            for signal in self.strategy.on_bar(event, feed):
                queue.put(signal)

        def on_signal(event: SignalEvent) -> None:
            signal_log.append(
                {
                    "timestamp": event.timestamp,
                    "symbol": event.symbol,
                    "target_weight": event.target_weight,
                    "strategy_id": event.strategy_id,
                    "reason": event.reason,
                }
            )
            order = self.portfolio.on_signal(event)
            if order is not None:
                queue.put(order)

        def on_order(event: OrderEvent) -> None:
            order_log.append(
                {
                    "timestamp": event.timestamp,
                    "order_id": event.order_id,
                    "symbol": event.symbol,
                    "quantity": event.quantity,
                    "order_type": event.order_type.value,
                    "strategy_id": event.strategy_id,
                }
            )
            self.execution.submit(event)

        handlers = {
            EventType.FILL: on_fill,
            EventType.MARKET: on_market,
            EventType.SIGNAL: on_signal,
            EventType.ORDER: on_order,
        }

        for market_event in feed:
            report = self.execution.on_market(market_event, feed)
            for fill in report.fills:
                queue.put(fill)
            for order in report.expired:
                self.portfolio.cancel_order(order.order_id)
                n_expired += 1

            queue.put(market_event)
            event: Event
            for event in queue.drain():
                handlers[event.type](event)  # type: ignore[arg-type]

        signal_cols = ["timestamp", "symbol", "target_weight", "strategy_id", "reason"]
        order_cols = ["timestamp", "order_id", "symbol", "quantity", "order_type", "strategy_id"]
        return BacktestResult(
            strategy_name=self.strategy.strategy_id,
            params=dict(self.strategy.params),
            initial_cash=self.portfolio.initial_cash,
            history=self.portfolio.history_frame(),
            trades=self.portfolio.trades_frame(),
            orders=pd.DataFrame(order_log, columns=order_cols),
            signals=pd.DataFrame(signal_log, columns=signal_cols),
            costs=self.execution.costs_frame(),
            n_expired=n_expired,
            open_orders_at_end=self.execution.working_orders,
        )
