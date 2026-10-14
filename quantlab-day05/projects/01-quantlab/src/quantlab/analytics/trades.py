"""Trade-level statistics, turnover and exposure from a backtest's tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Statistics over the fills that realised profit or loss.

    ``trades`` is ``BacktestResult.trades`` (or ``Portfolio.trades_frame()``).
    Opening fills realise nothing, so only *closing or reducing* fills are
    counted; a position built in three fills and closed in one is one winning
    or losing observation, not four. P&L is gross of commission (commissions
    are reported separately and in the cost breakdown).
    """
    keys = [
        "n_closing_fills",
        "win_rate",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "profit_factor",
        "expectancy",
        "largest_win",
        "largest_loss",
        "gross_realized_pnl",
        "total_commission",
    ]
    if trades is None or len(trades) == 0:
        return dict.fromkeys(keys, float("nan")) | {"n_closing_fills": 0.0}

    pnl = trades["realized_pnl"].astype("float64")
    closing = pnl[pnl != 0.0]
    commission = float(trades["commission"].sum())
    if closing.empty:
        return dict.fromkeys(keys, float("nan")) | {
            "n_closing_fills": 0.0,
            "gross_realized_pnl": 0.0,
            "total_commission": commission,
        }

    wins, losses = closing[closing > 0], closing[closing < 0]
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    gross_loss = float(-losses.sum())
    return {
        "n_closing_fills": float(len(closing)),
        "win_rate": float(len(wins) / len(closing)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": avg_win / abs(avg_loss) if len(losses) and len(wins) else float("nan"),
        "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy": float(closing.mean()),
        "largest_win": float(closing.max()),
        "largest_loss": float(closing.min()),
        "gross_realized_pnl": float(closing.sum()),
        "total_commission": commission,
    }


def turnover_statistics(
    history: pd.DataFrame, trades: pd.DataFrame, periods_per_year: int = 252
) -> dict[str, float]:
    """Traded notional relative to the size of the portfolio.

    ``annual_turnover`` is *one-way*: (bought + sold notional) / 2 / average
    equity / years, so a strategy that fully rotates its book once a year scores 1.0.
    """
    if len(history) == 0:
        nan = float("nan")
        return {"traded_notional": 0.0, "annual_turnover": nan, "avg_trade_size": nan}
    notional = (
        float((trades["quantity"].abs() * trades["price"]).sum()) if len(trades) else 0.0
    )
    avg_equity = float(history["equity"].mean())
    years = len(history) / periods_per_year
    annual = notional / 2.0 / avg_equity / years if avg_equity > 0 and years > 0 else float("nan")
    return {
        "traded_notional": notional,
        "annual_turnover": annual,
        "avg_trade_size": notional / len(trades) if len(trades) else float("nan"),
    }


def exposure_statistics(history: pd.DataFrame) -> dict[str, float]:
    """Leverage and time-in-market from the recorded equity history."""
    if len(history) == 0:
        return dict.fromkeys(
            ["avg_gross_leverage", "max_gross_leverage", "avg_net_exposure", "pct_time_invested"],
            float("nan"),
        )
    equity = history["equity"].replace(0.0, np.nan)
    gross = history["gross_exposure"] / equity
    net = history["net_exposure"] / equity
    return {
        "avg_gross_leverage": float(gross.mean()),
        "max_gross_leverage": float(gross.max()),
        "avg_net_exposure": float(net.mean()),
        "pct_time_invested": float((history["gross_exposure"] > 1e-9).mean()),
    }
