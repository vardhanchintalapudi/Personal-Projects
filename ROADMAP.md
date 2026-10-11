# 9-Week Master Roadmap

**Rule of the game:** one industry-grade Python 3 project per week, built in 7 daily slices (~1/7 of the
code each day). Day 7 ends with a tagged release, docs and a passing CI. The next project starts the
following day.

| # | Dates (2026) | Project | Domain | Headline skill |
|---|--------------|---------|--------|----------------|
| 1 | Oct 10 - Oct 16 | **QuantLab** - event-driven backtester | Finance | Systems design, risk analytics |
| 2 | Oct 17 - Oct 23 | **StreamFlow** - medallion lakehouse pipeline | Data Engineering | ELT, data quality, orchestration |
| 3 | Oct 24 - Oct 30 | **TurboRUL** - turbofan remaining-useful-life | ML (aerospace) | Time-series deep learning, MLflow |
| 4 | Oct 31 - Nov 6 | **DocuMind** - production RAG system | AI / LLM | Retrieval, reranking, evals |
| 5 | Nov 7 - Nov 13 | **RetailPulse** - customer & experimentation analytics | Data Analytics | SQL, CLV, A/B testing |
| 6 | Nov 14 - Nov 20 | **CreditRisk** - scorecard and ML credit model | DS / Finance | WOE/IV, calibration, SHAP |
| 7 | Nov 21 - Nov 27 | **AlphaRL** - deep-RL portfolio agent | AI / Finance | PPO/SAC from scratch |
| 8 | Nov 28 - Dec 4 | **ForecastBench** - forecasting benchmark platform | ML | Probabilistic forecasting, conformal |
| 9 | Dec 5 - Dec 11 | **MLOpsForge** - feature store, registry, monitoring | MLOps / DE | Serving, drift, CI/CD capstone |

Dates are a guide: if a day is missed, shift the remaining days of that project rather than skipping
content. The point is shipped, working slices, not a perfect calendar.

## Daily slice standard

Every day's commit must:

1. Add working code (not a stub) plus tests for it.
2. Pass `ruff check .` and `pytest` locally (`tools/daily_commit.sh` enforces this).
3. Update the project's `docs/devlog.md` with what was built, one design decision, and tomorrow's goal.
4. Use the commit message `feat(<slug>): day N/7 - <summary>`.

## Definition of done (Day 7 of every project)

- README with problem statement, architecture diagram, quick start, results and limitations
- Tests green in CI, meaningful coverage of core logic
- Reproducible demo (`make demo` or `python scripts/...`) with fixed seeds
- Tagged release `<slug>-v0.1.0` and a short "what I learned" section in the devlog

---

## Project 1 - QuantLab (Oct 10 - Oct 16)

**Domain:** Finance | **Stack:** numpy, pandas, pyarrow, matplotlib | **Slug:** `quantlab`

Event-driven backtesting engine with realistic execution costs, a risk/performance analytics suite and
research tooling that guards against overfitting.

| Day | Slice |
|-----|-------|
| 1 | Project scaffold, config, OHLCV schema validation, synthetic correlated market generator with regime switching, CSV/Parquet loaders, resampling, return utilities |
| 2 | Event system (Market/Signal/Order/Fill events), event queue, data feed that replays bars without lookahead, portfolio and position accounting |
| 3 | Strategy interface and reference strategies: SMA crossover, time-series momentum, mean reversion, volatility-targeted allocation |
| 4 | Execution handler: commissions, slippage, square-root market impact, order types (market/limit/stop), partial fills |
| 5 | Performance and risk analytics: Sharpe, Sortino, Calmar, drawdowns, VaR/CVaR, rolling metrics, beta/alpha vs benchmark |
| 6 | Research tooling: parameter grid, walk-forward optimisation, bootstrap/Monte Carlo significance, deflated Sharpe ratio |
| 7 | HTML tear sheet, CLI, README with results, GitHub Actions polish, release `quantlab-v0.1.0` |

## Project 2 - StreamFlow (Oct 17 - Oct 23)

**Domain:** Data Engineering | **Stack:** DuckDB, pyarrow, pydantic, Docker | **Slug:** `streamflow`

A medallion (bronze/silver/gold) pipeline with idempotent loads, schema evolution handling, a data
quality framework and a lightweight orchestrator.

| Day | Slice |
|-----|-------|
| 1 | Scaffold, pydantic schemas, synthetic e-commerce event generator (orders, clicks, payments) with deliberate dirty data |
| 2 | Bronze layer: batch and micro-batch ingestion to partitioned Parquet, checkpoints, idempotent re-runs |
| 3 | Silver layer: cleaning, deduplication, type enforcement, SCD Type 2 dimensions, schema-drift handling |
| 4 | Gold layer: star-schema marts, incremental aggregates via DuckDB SQL, window functions |
| 5 | Data quality framework: expectations (nulls, uniqueness, ranges, freshness, referential integrity), quarantine, DQ report |
| 6 | Orchestration: DAG runner with dependencies, retries, backfills, run metadata and lineage |
| 7 | Observability, CLI, Dockerfile and compose, benchmark of rows/sec, docs, release |

## Project 3 - TurboRUL (Oct 24 - Oct 30)

**Domain:** ML (aerospace) | **Stack:** PyTorch, LightGBM, Optuna, MLflow, FastAPI | **Slug:** `turborul`

Predict remaining useful life of turbofan engines from multivariate sensor streams (NASA C-MAPSS
format, with a synthetic degradation generator so the repo runs offline).

| Day | Slice |
|-----|-------|
| 1 | Scaffold, C-MAPSS loaders (FD001-FD004), synthetic degradation generator, EDA script |
| 2 | Preprocessing: sensor selection, per-operating-condition normalisation, piecewise-linear RUL target, sliding windows |
| 3 | Feature engineering (rolling stats, health index) and gradient-boosting baselines with engine-grouped CV |
| 4 | Deep models: 1D-CNN/TCN and LSTM, training loop, early stopping, checkpoints |
| 5 | Uncertainty (MC dropout, quantile heads), NASA asymmetric score, error analysis by degradation stage |
| 6 | Experiment tracking with MLflow, Optuna search, model comparison report |
| 7 | FastAPI serving, Docker, model card, docs, release |

## Project 4 - DocuMind (Oct 31 - Nov 6)

**Domain:** AI / LLM | **Stack:** sentence-transformers, FAISS, FastAPI, Streamlit | **Slug:** `documind`

Retrieval-augmented generation done properly: hybrid search, reranking, citations and, most
importantly, an evaluation harness.

| Day | Slice |
|-----|-------|
| 1 | Scaffold, document ingestion (PDF/Markdown/HTML), cleaning, metadata model |
| 2 | Chunking strategies (fixed, recursive, semantic, sliding window) and chunk-quality metrics |
| 3 | Retrieval: BM25 from scratch, dense embeddings, vector index |
| 4 | Hybrid fusion (reciprocal rank fusion), cross-encoder reranking, query rewriting / HyDE |
| 5 | Generation: provider-agnostic LLM interface, prompt templates, citations, guardrails |
| 6 | Evaluation: synthetic QA generation, recall@k, MRR, nDCG, faithfulness, LLM-as-judge, ablation report |
| 7 | FastAPI and Streamlit UI, caching, Docker, docs, release |

## Project 5 - RetailPulse (Nov 7 - Nov 13)

**Domain:** Data Analytics | **Stack:** DuckDB/SQL, pandas, scipy, Plotly, Streamlit | **Slug:** `retailpulse`

| Day | Slice |
|-----|-------|
| 1 | Scaffold, synthetic retail transaction generator, warehouse schema |
| 2 | SQL analytics layer: KPIs, YoY, funnels, window functions |
| 3 | Segmentation: RFM, K-Means/GMM, cohort retention matrices |
| 4 | Customer lifetime value: BG/NBD and Gamma-Gamma implemented with scipy, churn modelling |
| 5 | Experimentation toolkit: power analysis, frequentist tests, CUPED, Bayesian and sequential tests, multiple-testing corrections |
| 6 | Demand forecasting and KPI anomaly detection |
| 7 | Streamlit/Plotly dashboard, insight report, docs, release |

## Project 6 - CreditRisk (Nov 14 - Nov 20)

**Domain:** Data Science / Finance | **Stack:** scikit-learn, LightGBM, SHAP | **Slug:** `creditrisk`

| Day | Slice |
|-----|-------|
| 1 | Scaffold, data loaders plus synthetic applicant generator, EDA, leakage checks |
| 2 | WOE/IV monotonic binning, feature selection, multicollinearity (VIF) |
| 3 | Logistic scorecard with points scaling (PDO) and basic reject inference |
| 4 | Gradient-boosting challenger, calibration (Platt/isotonic), cost-sensitive thresholds |
| 5 | Explainability: SHAP, reason codes, partial dependence, fairness metrics |
| 6 | Monitoring: PSI/CSI drift, stability backtests, stress tests, expected loss (PD x LGD x EAD) |
| 7 | Model card, scoring API, docs, release |

## Project 7 - AlphaRL (Nov 21 - Nov 27)

**Domain:** AI / Finance | **Stack:** PyTorch, Gymnasium | **Slug:** `alpharl`

| Day | Slice |
|-----|-------|
| 1 | Scaffold, Gymnasium portfolio environment with transaction costs (reusing QuantLab data code) |
| 2 | State features, reward shaping, baselines (buy-and-hold, equal weight, mean-variance) |
| 3 | PPO from scratch: GAE, clipped objective, minibatch updates |
| 4 | SAC / Dirichlet-policy agent for continuous portfolio weights |
| 5 | Training infrastructure: vectorised envs, seeding, logging, sweeps |
| 6 | Evaluation: walk-forward, regime stress tests, statistical comparison against baselines |
| 7 | Report, reproducibility checklist, docs, release |

## Project 8 - ForecastBench (Nov 28 - Dec 4)

**Domain:** ML | **Stack:** statsmodels, LightGBM, PyTorch | **Slug:** `forecastbench`

| Day | Slice |
|-----|-------|
| 1 | Scaffold, dataset loaders and synthetic generator, rolling-origin cross-validation framework |
| 2 | Classical models (naive, seasonal naive, ETS, ARIMA wrappers) and metrics (MASE, sMAPE, CRPS) |
| 3 | Global gradient-boosting model with lag and calendar features |
| 4 | N-BEATS and a patch-based Transformer in PyTorch |
| 5 | Probabilistic forecasts: quantile loss, conformal prediction intervals, calibration plots |
| 6 | Ensembling/stacking, leaderboard, Diebold-Mariano tests |
| 7 | CLI, report, docs, release |

## Project 9 - MLOpsForge (Dec 5 - Dec 11)

**Domain:** MLOps / DE capstone | **Stack:** FastAPI, SQLite/DuckDB, Docker, GitHub Actions | **Slug:** `mlopsforge`

| Day | Slice |
|-----|-------|
| 1 | Scaffold, architecture, feature-store design (offline Parquet plus online key-value store) |
| 2 | Feature definitions, point-in-time-correct joins, materialisation jobs |
| 3 | Model registry (versions, stages, metadata) and reproducible training pipelines |
| 4 | Serving: FastAPI with micro-batching, shadow and canary routing, latency metrics |
| 5 | Monitoring: data drift (KS/PSI), prediction drift, performance decay, alerts |
| 6 | CI/CD, docker-compose, integration and load tests |
| 7 | Final polish, architecture diagrams, profile README linking all nine projects, `v1.0.0` |

---

## Beyond week 9

Candidates once the core nine are done: LLM agent framework with tool-use evals, graph ML for fraud
detection, causal inference toolkit, streaming analytics with Kafka-compatible APIs.
