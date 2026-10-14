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

## Day 3 - Strategies (Oct 12, 2026)

**Built**
- `quantlab.strategy.base`: `Strategy` template method (`compute_targets` hook), `Target`,
  signal de-duplication with a weight tolerance, exposure-reducing-first ordering, `reset_state` hook
- `quantlab.strategy.registry`: `register_strategy`, `get_strategy`, `create_strategy`
- Four strategies: `SMACrossover`, `TimeSeriesMomentum` (with skip, threshold, rebalance cadence),
  `ZScoreMeanReversion` (entry/exit hysteresis), `VolTargetAllocation` (inverse-vol, covariance-based
  scaling to a target volatility, leverage cap, rebalance band)
- `quantlab.testing`: `frame_from_closes`, `collect_signals`, `naive_backtest` (next-open fills)
- 84 new tests (169 total) including pandas reference implementations and truncation invariance
- `scripts/demo_day3.py`

**Design decisions**
1. *Truncation invariance as the lookahead test.* Instead of trusting each indicator, run the
   strategy on the full history and on a shortened one: signals up to the cut must be identical.
   It catches whole-sample normalisation, `len(feed)` tricks, and anything else that leaks.
2. *All-or-nothing vol-target rebalance.* My first version applied the tolerance per symbol, which
   could leave the held weights as a blend of old and new targets and exceed the leverage cap by a
   few points. A band on the whole vector fixes it, and a test now asserts the cap is never breached.
3. *Exposure-reducing signals first.* Cheap to do in the base class, and it removes the
   first-come-first-served cap gotcha from Day 2 for every strategy at once.

**Observation**
Mean reversion loses money on the synthetic data. That is the correct answer: the generator is a
random walk, so there is no reversion to harvest, and the strategy pays commission 448 times. It
makes a useful negative control for the Day 6 significance tests.

**Tomorrow (Day 4)**
Execution handler replacing `naive_backtest`: commissions, slippage, square-root market impact,
market/limit/stop orders and partial fills, wired into the event queue.

**Notes / what I learned**
-

## Day 4 - Execution and the engine (Oct 13, 2026)

**Built**
- `quantlab.execution`: commission models (zero, bps, per-share with minimum and cap), slippage models
  (fixed bps, bar-range), square-root market impact, `ExecutionHandler` with market, limit and stop
  orders, shared per-bar participation cap, partial fills, order expiry and a per-fill cost log
- `quantlab.engine`: `BacktestEngine` driving everything through the `EventQueue`, and a
  `BacktestResult` (equity, returns, trades, orders, signals, costs, summary)
- 31 new tests (200 total), `scripts/demo_day4.py`

**Design decisions**
1. *Queue ordering encodes the timing rules.* Fills are queued before the bar's `MarketEvent`, so the
   recorded equity includes trades at that bar's open, and new orders only reach the handler after the
   strategy has run, so nothing can trade on the bar that produced it.
2. *The Day 3 stand-in became a regression test.* With costs off, the engine must equal
   `naive_backtest`; it does, to the cent, for three strategies.
3. *Impact uses only past bars.* The model drops the current bar before estimating volatility and ADV.

**Gotcha**
A stop or limit sees the whole bar's range, so a bar that both touches the stop and reverses is
treated as filled. OHLC data cannot say which came first; the docstring says so.

**Tomorrow (Day 5)**
Performance and risk analytics: Sharpe, Sortino, Calmar, drawdowns, VaR/CVaR, rolling metrics,
beta/alpha against a benchmark, trade statistics.

**Notes / what I learned**
-

## Day 5 - Analytics (Oct 14, 2026)

**Built**
- `quantlab.analytics.metrics`: total return, CAGR, volatility, Sharpe, Sortino (target downside
  deviation), drawdown series and episode table, Calmar, VaR and CVaR (historical, Gaussian,
  Cornish-Fisher), tail ratio, win rate, rolling Sharpe/volatility, monthly return table,
  beta/alpha/tracking error/information ratio/up-down capture, `performance_summary`
- `quantlab.analytics.trades`: closing-fill statistics, one-way turnover, leverage and time-in-market
- `quantlab.analytics.benchmark`: equal-weight benchmark (rebalanced or buy-and-hold)
- `BacktestResult.performance()`, `.trade_stats()`, `.turnover()`, `.exposure()`
- 32 new tests (232 total), `scripts/demo_day5.py`

**Design decisions**
1. *No SciPy.* The only distribution function needed is the normal quantile/density, and
   `statistics.NormalDist` has both. One fewer dependency to install and to explain.
2. *Drawdowns start from wealth 1.0.* A loss on the very first bar is a drawdown; starting the
   running peak from the first return would hide it.
3. *Trade statistics count closing fills only.* Opening fills realise nothing, so counting them would
   inflate the trade count and dilute win rate. The docstring says what a "trade" means here.

**Gotcha**
My first integration test asserted gross leverage never exceeds the 1.0 cap. It does, slightly: weights
are set at signal time, and between rebalances a rallying short lifts gross/equity above 1. That is
correct behaviour, so the test now asserts a drift bound instead of the cap.

**Tomorrow (Day 6)**
Research tooling: parameter grids, walk-forward optimisation, bootstrap significance tests,
and the deflated Sharpe ratio for correcting selection bias.

**Notes / what I learned**
-
