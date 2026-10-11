"""Day 1 demo: generate a synthetic market, validate it, persist it, summarise it.

Run from the project root:

    python scripts/demo_day1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.config import DEFAULT_SETTINGS
from quantlab.data import (
    DEFAULT_REGIMES,
    generate_ohlcv_panel,
    load_directory,
    save_parquet,
    to_close_panel,
    validate_ohlcv,
)
from quantlab.utils import (
    annualization_factor,
    annualize_return,
    annualize_vol,
    simple_returns,
)

SYMBOLS = ["AAPL_SIM", "MSFT_SIM", "GLD_SIM", "TLT_SIM"]


def main() -> None:
    market = generate_ohlcv_panel(
        SYMBOLS,
        start="2018-01-01",
        periods=1500,
        annual_drift=[0.12, 0.10, 0.04, 0.02],
        annual_vol=[0.28, 0.25, 0.15, 0.12],
        correlation=0.35,
        regimes=DEFAULT_REGIMES,
        seed=DEFAULT_SETTINGS.seed,
    )

    out_dir = Path("sample_data")
    for symbol, frame in market.items():
        report = validate_ohlcv(frame)
        print(f"{symbol:<9} validation: {report}")
        save_parquet(frame, out_dir / f"{symbol}.parquet")

    reloaded = load_directory(out_dir, "*.parquet")
    closes = to_close_panel(reloaded)
    rets = simple_returns(closes)
    ppy = annualization_factor("B")

    summary = pd.DataFrame(
        {
            "ann_return": rets.apply(annualize_return, periods_per_year=ppy),
            "ann_vol": rets.apply(annualize_vol, periods_per_year=ppy),
        }
    )
    summary["return/vol"] = summary["ann_return"] / summary["ann_vol"]

    print("\nAnnualised statistics (synthetic data)")
    print(summary.round(3).to_string())
    print("\nReturn correlation matrix")
    print(rets.corr().round(2).to_string())


if __name__ == "__main__":
    main()
