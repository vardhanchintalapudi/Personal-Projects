# QuantLab Architecture

## Event loop

```mermaid
flowchart LR
    DF[DataFeed<br/>replays bars] -->|MarketEvent| Q((Event queue))
    Q -->|MarketEvent| S[Strategy]
    S -->|SignalEvent| Q
    Q -->|SignalEvent| P[Portfolio<br/>sizing + risk]
    P -->|OrderEvent| Q
    Q -->|OrderEvent| E[ExecutionHandler<br/>costs + slippage]
    E -->|FillEvent| Q
    Q -->|FillEvent| P
    P --> A[Analytics<br/>metrics + tear sheet]
```

The queue is drained after every bar, so a strategy can only react to data that has already been
published. Orders placed on bar *t* fill no earlier than bar *t+1*.

## Components and delivery schedule

| Component | Module(s) | Day |
|-----------|-----------|-----|
| Data validation, loaders, synthetic markets | `quantlab.data` | 1 (done) |
| Return and annualisation utilities | `quantlab.utils` | 1 (done) |
| Events, queue, data feed, portfolio | `quantlab.events`, `quantlab.feed`, `quantlab.portfolio` | 2 (done) |
| Strategy interface and strategies | `quantlab.strategy` | 3 |
| Execution and cost models | `quantlab.execution` | 4 |
| Performance and risk analytics | `quantlab.analytics` | 5 |
| Research tooling | `quantlab.research` | 6 |
| Reporting and CLI | `quantlab.report`, `quantlab.cli` | 7 |

## Key design decisions

1. **Single-instrument frames, panel helpers.** The unit of storage is one validated OHLCV frame per
   symbol; `to_close_panel` builds wide views when a cross-sectional strategy needs them.
2. **Costs are a model, not a constant.** Day 4 makes commission, slippage and market impact
   pluggable, since backtested edge often disappears once costs are realistic.
3. **Analytics are pure functions.** Metrics take a return series and return numbers, so they are
   trivially testable and reusable outside the engine.
