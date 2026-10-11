# Master Portfolio

Nine industry-style Python 3 projects spanning **AI, ML, Data Science, Data Engineering, Data Analytics
and Finance**, built one per week in seven daily slices. Every project ships with tests, CI, a design
write-up and a reproducible demo.

> Daily commits, one finished project every week. See [ROADMAP.md](ROADMAP.md) for the full plan and
> [docs/WORKFLOW.md](docs/WORKFLOW.md) for how the daily cadence works.

## Projects

| # | Project | Domain | Dates (2026) | Status |
|---|---------|--------|--------------|--------|
| 1 | [QuantLab](projects/01-quantlab) - event-driven backtester & risk analytics | Finance | Oct 10 - Oct 16 | In progress (Day 3/7) |
| 2 | StreamFlow - medallion lakehouse pipeline | Data Engineering | Oct 17 - Oct 23 | Planned |
| 3 | TurboRUL - turbofan remaining-useful-life | ML | Oct 24 - Oct 30 | Planned |
| 4 | DocuMind - production RAG system | AI / LLM | Oct 31 - Nov 6 | Planned |
| 5 | RetailPulse - customer & experimentation analytics | Data Analytics | Nov 7 - Nov 13 | Planned |
| 6 | CreditRisk - scorecard and ML credit model | DS / Finance | Nov 14 - Nov 20 | Planned |
| 7 | AlphaRL - deep-RL portfolio agent | AI / Finance | Nov 21 - Nov 27 | Planned |
| 8 | ForecastBench - forecasting benchmark platform | ML | Nov 28 - Dec 4 | Planned |
| 9 | MLOpsForge - feature store, registry, monitoring | MLOps / DE | Dec 5 - Dec 11 | Planned |

## Repository layout

```
.
├── README.md                 # this file
├── ROADMAP.md                # 9-week plan, day-by-day
├── docs/                     # cross-project docs (workflow, conventions)
├── tools/daily_commit.sh     # lint + test + commit + push helper
├── .github/workflows/ci.yml  # discovers every project and runs lint + tests
└── projects/
    └── 01-quantlab/          # each project is self-contained
        ├── pyproject.toml
        ├── README.md
        ├── src/<package>/
        ├── tests/
        ├── docs/             # architecture + devlog
        └── scripts/          # runnable demos
```

## Working on a project

```bash
cd projects/01-quantlab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && pytest
python scripts/demo_day1.py
```

## Conventions

- Python 3.10+, type hints, docstrings on public APIs
- `ruff` for linting, `pytest` for tests, fixed random seeds in every demo
- Commit messages: `feat(<slug>): day N/7 - <summary>` (use `fix`, `docs`, `test`, `refactor` as needed)
- Large datasets never enter git; each project documents how to download or regenerate its data

## License

MIT, see [LICENSE](LICENSE).
