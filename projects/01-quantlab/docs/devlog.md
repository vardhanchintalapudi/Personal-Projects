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
