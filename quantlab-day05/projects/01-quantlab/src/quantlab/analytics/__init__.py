"""Performance, risk and trade analytics."""

from quantlab.analytics.benchmark import equal_weight_returns
from quantlab.analytics.metrics import (
    annual_volatility,
    benchmark_stats,
    cagr,
    calmar_ratio,
    conditional_var,
    drawdown_series,
    drawdown_table,
    max_drawdown,
    monthly_returns,
    performance_summary,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    total_return,
    value_at_risk,
    win_rate,
)
from quantlab.analytics.trades import exposure_statistics, trade_statistics, turnover_statistics

__all__ = [
    "annual_volatility",
    "benchmark_stats",
    "cagr",
    "calmar_ratio",
    "conditional_var",
    "drawdown_series",
    "drawdown_table",
    "equal_weight_returns",
    "exposure_statistics",
    "max_drawdown",
    "monthly_returns",
    "performance_summary",
    "rolling_sharpe",
    "rolling_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "tail_ratio",
    "total_return",
    "trade_statistics",
    "turnover_statistics",
    "value_at_risk",
    "win_rate",
]
