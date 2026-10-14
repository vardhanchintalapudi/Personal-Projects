# QuantLab

Event-driven backtesting engine and risk analytics suite for systematic trading research.

**Status:** Day 5 of 7 complete (analytics). Week of Oct 10 - Oct 16, 2026.

## Why event-driven?

Vectorised backtests are fast but make it easy to leak future information into past decisions. An
event-driven design replays history one bar at a time through the same code path a live system would
use (data feed -> strategy -> portfolio -> execution), which makes lookahead bias structurally hard
and execution costs first-class citizens.

See [docs/architecture.md](docs/architecture.md) for the full component diagram.

## Roadmap

| Day | Slice | Status |
|-----|-------|--------|
| 1 | Scaffold, schema validation, synthetic market generator, loaders, return utilities | Done |
| 2 | Event system, bar-replay data feed, portfolio and position accounting | Done |
| 3 | Strategy interface and reference strategies | Done |
| 4 | Execution: commissions, slippage, market impact, order types, engine | Done |
| 5 | Performance and risk analytics | Done |
| 6 | Walk-forward optimisation, Monte Carlo, deflated Sharpe | Todo |
| 7 | Tear sheet, CLI, docs, release | Todo |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/demo_day1.py
```

## What exists today (Days 1-5)

```python
from quantlab.data import DEFAULT_REGIMES, generate_ohlcv_panel, to_close_panel, validate_ohlcv
from quantlab.utils import annualization_factor, annualize_vol, simple_returns

market = generate_ohlcv_panel(
    ["AAA", "BBB", "CCC"],
    periods=1000,
    annual_vol=[0.25, 0.20, 0.15],
    correlation=0.4,
    regimes=DEFAULT_REGIMES,   # calm bull <-> turbulent bear Markov chain
    seed=42,
)

validate_ohlcv(market["AAA"])                      # raises DataValidationError on bad data
closes = to_close_panel(market)                    # wide date x symbol frame
rets = simple_returns(closes)
print(rets.apply(annualize_vol, periods_per_year=annualization_factor("B")))
```

| Module | Responsibility |
|--------|----------------|
| `quantlab.data.schema` | OHLCV contract: index, columns, NaN/inf, positive prices, bar consistency |
| `quantlab.data.synthetic` | Correlated GBM on an intraday grid, optional Markov regime switching |
| `quantlab.data.loaders` | CSV/Parquet I/O, header normalisation, timezones, resampling, panels |
| `quantlab.utils.returns` | Simple/log/cumulative returns, annualisation of return and volatility |
| `quantlab.config` | Frequency table and project-wide defaults |
| `quantlab.events` | Immutable `Market/Signal/Order/Fill` events and the FIFO `EventQueue` |
| `quantlab.feed` | `HistoricalDataFeed`: bar replay where the past is the only thing you can query |
| `quantlab.portfolio` | Average-cost positions, cash, sizing from target weights, leverage caps, equity curve |
| `quantlab.strategy` | `Strategy` base class, registry, and four reference strategies (below) |
| `quantlab.execution` | Commission, slippage and square-root impact models; market/limit/stop orders; participation caps and partial fills |
| `quantlab.engine` | `BacktestEngine`: the queue-driven loop, returns a `BacktestResult` with equity, trades, orders, signals and a per-fill cost breakdown |
| `quantlab.analytics` | Sharpe, Sortino, Calmar, drawdown tables, VaR/CVaR (historical, Gaussian, Cornish-Fisher), rolling metrics, monthly returns, beta/alpha/capture, trade, turnover and exposure statistics |
| `quantlab.testing` | `frame_from_closes`, `collect_signals`, and a stand-in `naive_backtest` (until Day 4) |

### Replaying a market and trading it (Day 2)

```python
from quantlab.events import FillEvent, SignalEvent
from quantlab.feed import HistoricalDataFeed
from quantlab.portfolio import Portfolio

feed = HistoricalDataFeed(market)
portfolio = Portfolio(100_000, max_gross_leverage=1.0)

for event in feed:                              # one MarketEvent per timestamp
    portfolio.on_market(event)                  # mark to market, record equity
    closes = feed.history("AAA", 50)            # past only, never the future
    order = portfolio.on_signal(SignalEvent(event.timestamp, "AAA", 0.5))
    # ...an execution handler turns `order` into a FillEvent (Day 4)
```

### Strategies (Day 3)

Strategies only decide *what exposure they want*; the portfolio sizes and the (Day 4) execution
handler trades. Every strategy is registered under a name for configs and the Day 7 CLI.

| Key | Class | Idea | Main parameters |
|-----|-------|------|-----------------|
| `sma_crossover` | `SMACrossover` | Long while fast SMA > slow SMA (optionally short below) | `fast`, `slow`, `allow_short` |
| `tsmom` | `TimeSeriesMomentum` | Trade the sign of each asset's own trailing return | `lookback`, `skip`, `rebalance_every`, `threshold` |
| `mean_reversion` | `ZScoreMeanReversion` | Fade stretched z-scores with entry/exit hysteresis | `lookback`, `entry_z`, `exit_z` |
| `vol_target` | `VolTargetAllocation` | Inverse-vol weights scaled to a target portfolio volatility | `target_vol`, `lookback`, `rebalance_every`, `max_leverage` |

```python
from quantlab.data import generate_ohlcv_panel
from quantlab.portfolio import Portfolio
from quantlab.strategy import create_strategy
from quantlab.testing import naive_backtest

market = generate_ohlcv_panel(["A", "B", "C"], periods=1000, seed=1)
strategy = create_strategy("tsmom", ["A", "B", "C"], lookback=126, rebalance_every=21)
portfolio = naive_backtest(market, strategy, Portfolio(100_000), commission_bps=5)
print(portfolio.summary())
```

Writing your own is one method:

```python
from quantlab.strategy import Strategy, Target, register_strategy

@register_strategy("buy_and_hold")
class BuyAndHold(Strategy):
    def compute_targets(self, event, feed):
        return {s: Target(1 / len(self.symbols), "hold") for s in self.active_symbols(event)}
```

`python scripts/demo_day3.py` runs all four on one synthetic market. Sample output (synthetic
random-walk data, 5 bps commission, so read it as a smoke test, not evidence of an edge):

| Strategy | Total return | Ann. vol | Max drawdown | Trades |
|----------|-------------:|---------:|-------------:|-------:|
| SMA 20/100 (long only) | 55.8% | 11.2% | -18.6% | 60 |
| Momentum 126d (long/short) | 63.7% | 12.8% | -31.9% | 40 |
| Mean reversion z20 (L/S) | -18.1% | 8.1% | -20.6% | 448 |
| Vol target 8% | 31.6% | 8.6% | -23.3% | 404 |

The mean-reversion loss is expected: the synthetic prices are a random walk with drift, so there
is nothing to revert to, and 448 trades pay a lot of commission. The vol-targeted portfolio's
realised volatility lands close to its 8% target, which is the behaviour it exists to produce.

### Realistic execution (Day 4)

```python
from quantlab.engine import BacktestEngine
from quantlab.execution import ExecutionHandler

result = BacktestEngine(market, strategy, execution=ExecutionHandler.realistic()).run()
print(result.summary())     # includes total commission / slippage / impact
print(result.costs.head())  # per fill: reference price, fill price, each cost in cash
```

`ExecutionHandler.realistic()` is per-share commission, 1 bp slippage, square-root impact and a 10 %
participation cap. `python scripts/demo_day4.py` shows how much each layer costs on the same signals.

### Analytics (Day 5)

```python
from quantlab.analytics import drawdown_table, equal_weight_returns

benchmark = equal_weight_returns(market)
perf = result.performance(benchmark=benchmark)   # pd.Series: sharpe, sortino, calmar, var_95, beta, alpha, ...
drawdown_table(result.returns, top=3)            # peak, trough, recovery, depth, bars underwater
result.trade_stats(); result.turnover(); result.exposure()
```

All metrics are pure functions of a return series, use only the standard library's `NormalDist` for
the normal-distribution maths (no SciPy), and return `nan` instead of raising when something is
undefined. Risk numbers are reported as positive losses; drawdowns as negative fractions.
`python scripts/demo_day5.py` prints the full comparison table for all four strategies.

## Design notes

- **Synthetic data on a fine grid.** High and low come from simulated intraday paths, so bar ranges
  are consistent with the volatility rather than bolted on.
- **Validation at the boundary.** Loaders and writers refuse malformed data, so downstream code can
  assume clean inputs.
- **Determinism.** Every generator takes a seed; tests assert reproducibility.
- **Lookahead is structurally impossible.** The feed slices history at an internal cursor; the
  tests check at every step that no returned timestamp is later than the current bar.
- **Signals are target weights.** Strategies say *what exposure they want*; only the portfolio
  knows equity, prices and risk caps, so it alone converts weights into share quantities.
- **Strategies cannot peek, and a test proves it.** `test_strategy_no_lookahead.py` re-runs every
  strategy on truncated histories and requires identical signals up to the cut.
- **Reference-implementation tests.** Each strategy is checked against an independent pandas
  (or plain-loop) implementation across multi-asset synthetic markets, not just hand-built cases.
- **Costs are accounted for exactly.** A test runs one scripted trade with and without costs and
  asserts that the equity gap equals the recorded commission + slippage + impact to machine precision.
- **The engine is checked against an independent implementation.** With zero costs it reproduces the
  Day 3 `naive_backtest` equity curve to the cent for three different strategies.
- **Impact cannot see the current bar.** The square-root model measures volatility and volume on the
  bars before the one being filled; a test changes the current bar's volume and checks nothing moves.
- **Every formula is pinned by a hand calculation.** Sharpe, Sortino, drawdown episodes, VaR/CVaR
  on a uniform grid, beta/alpha of a levered clone with known alpha, monthly compounding: each has a
  test whose expected value was worked out on paper, not by calling the function.
- **Layers agree with each other.** `result.performance()["total_return"]` must equal the engine's
  summary, and `max_drawdown(returns)` must equal the drawdown of the raw equity curve.
- **Pending orders count.** Sizing uses position plus working orders, so a slow fill can never
  trigger a duplicate order.

## Limitations (so far)

Synthetic prices are continuous (no overnight gaps) and returns are lognormal, so tails are lighter
than real markets. Each bar is treated independently by the execution simulator, so stops and limits
see the whole bar's range with no intrabar ordering. The mean-reversion strategy has no stop-loss.
