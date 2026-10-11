# QuantLab

Event-driven backtesting engine and risk analytics suite for systematic trading research.

**Status:** Day 2 of 7 complete (event system, feed, portfolio). Week of Oct 10 - Oct 16, 2026.

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
| 3 | Strategy interface and reference strategies | Todo |
| 4 | Execution: commissions, slippage, market impact, order types | Todo |
| 5 | Performance and risk analytics | Todo |
| 6 | Walk-forward optimisation, Monte Carlo, deflated Sharpe | Todo |
| 7 | Tear sheet, CLI, docs, release | Todo |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/demo_day1.py
```

## What exists today (Days 1-2)

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
- **Pending orders count.** Sizing uses position plus working orders, so a slow fill can never
  trigger a duplicate order.

## Limitations (so far)

Synthetic prices are continuous (no overnight gaps) and returns are lognormal, so tails are lighter
than real markets. Later days add execution frictions; heavier-tailed generators may follow.
