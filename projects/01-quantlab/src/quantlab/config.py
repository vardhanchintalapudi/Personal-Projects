"""Central, immutable configuration shared across QuantLab modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Bars per year for the common pandas frequency aliases. Used to annualise
#: returns and volatility. ``"B"``/``"D"`` assume the equity convention of 252.
PERIODS_PER_YEAR: dict[str, int] = {
    "B": 252,
    "D": 252,
    "W": 52,
    "ME": 12,
    "M": 12,
    "QE": 4,
    "Q": 4,
    "YE": 1,
    "Y": 1,
    "h": 252 * 7,  # ~7 trading hours per session
    "H": 252 * 7,
    "min": 252 * 390,  # 390 one-minute bars per US equity session
    "T": 252 * 390,
}

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Settings:
    """Project-wide defaults. Override by constructing your own instance."""

    seed: int = 42
    trading_days_per_year: int = 252
    data_dir: Path = Path("data")
    sample_dir: Path = Path("sample_data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


DEFAULT_SETTINGS = Settings()
