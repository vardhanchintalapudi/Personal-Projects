# QuantLab Devlog

## Day 1 - Data layer (Oct 10, 2026)

**Built**
- Project scaffold: `pyproject.toml`, ruff/pytest config, `src/` layout, CI discovery
- `quantlab.data.schema`: strict OHLCV validation with a structured report
- `quantlab.data.synthetic`: correlated multi-asset GBM on an intraday grid with optional regime switching
- `quantlab.data.loaders`: CSV/Parquet I/O, header normalisation, timezone handling, resampling, close panels
- `quantlab.utils.returns`: returns and annualisation helpers
- 40 tests, `scripts/demo_day1.py`

**Design decision**
Simulating on a fine intraday grid costs a little compute but gives bars whose high/low are
consistent with the underlying volatility, which matters for stop-loss and limit-order logic on Day 4.

**Tomorrow (Day 2)**
Event classes, the event queue, a bar-replay data feed that cannot look ahead, and portfolio/position
accounting.

**Notes / what I learned**
-

## Day 2 - Events, feed and portfolio (Oct 11, 2026)

**Built**
- `quantlab.events`: immutable `Bar`, `MarketEvent`, `SignalEvent` (target weight), `OrderEvent`,
  `FillEvent`, plus a FIFO `EventQueue` whose `drain()` also yields events added mid-iteration
- `quantlab.feed.HistoricalDataFeed`: union-timeline replay of many symbols, cursor-based history
  (`history`, `history_frame`, `history_panel`, `window`), forward-filled last prices, `reset()`
- `quantlab.portfolio`: average-cost `Position` (adds, partial closes, flips), `Portfolio` with
  sizing from target weights, per-name and gross-leverage caps, pending-order tracking, minimum
  trade size, equity/trade history frames
- 45 new tests including a feed-to-portfolio integration test checked against independent arithmetic
- `scripts/demo_day2.py`

**Design decision**
Signals carry a *target weight*, not a buy/sell instruction. That keeps every strategy (trend,
mean reversion, vol-targeted allocation) expressible with one event, and keeps all money-aware
logic (equity, price, rounding, caps) in the portfolio.

**Gotcha worth remembering**
Gross-leverage capping is first-come first-served across symbols: the order in which a strategy
emits signals matters when the cap binds. It is documented and tested; Day 3 strategies emit equal
weights so they are unaffected.

**Tomorrow (Day 3)**
`Strategy` base class, SMA crossover, time-series momentum, z-score mean reversion and a
volatility-targeted allocation, with a truncation-invariance test that proves no strategy peeks.

**Notes / what I learned**
-
