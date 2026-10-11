"""OHLCV schema validation.

Backtests are only as trustworthy as their inputs. Every loader in QuantLab
funnels data through :func:`validate_ohlcv`, which returns a structured
:class:`ValidationReport` (or raises :class:`DataValidationError` in strict mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.config import OHLCV_COLUMNS


class DataValidationError(ValueError):
    """Raised when market data violates the OHLCV contract."""


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a validation pass."""

    n_rows: int
    issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise DataValidationError("; ".join(self.issues))

    def __str__(self) -> str:
        if self.ok:
            return f"OK ({self.n_rows} rows)"
        bullets = "\n  - ".join(self.issues)
        return f"FAILED ({self.n_rows} rows):\n  - {bullets}"


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    strict: bool = True,
    allow_zero_volume: bool = True,
) -> ValidationReport:
    """Validate a single-instrument OHLCV frame.

    Checks performed:

    * ``DatetimeIndex`` that is strictly increasing with no duplicates
    * all of ``open, high, low, close, volume`` present
    * no NaN / infinite values
    * strictly positive prices, non-negative volume
    * bar consistency: ``low <= min(open, close)`` and ``high >= max(open, close)``

    Parameters
    ----------
    df:
        Frame indexed by timestamp with lower-case OHLCV columns.
    strict:
        If True (default) raise :class:`DataValidationError` on any issue.
    allow_zero_volume:
        Set False to flag bars with zero volume (e.g. for liquid instruments).
    """
    issues: list[str] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("index must be a DatetimeIndex")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")

    report_rows = len(df)
    if missing or not isinstance(df.index, pd.DatetimeIndex):
        # The remaining checks need a well-formed frame.
        report = ValidationReport(report_rows, tuple(issues))
        if strict:
            report.raise_if_failed()
        return report

    if df.index.has_duplicates:
        issues.append(f"{int(df.index.duplicated().sum())} duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        issues.append("index is not sorted in increasing order")

    values = df.loc[:, list(OHLCV_COLUMNS)].astype("float64")

    n_nan = int(values.isna().to_numpy().sum())
    if n_nan:
        per_col = values.isna().sum()
        detail = {c: int(n) for c, n in per_col.items() if n}
        issues.append(f"{n_nan} NaN values {detail}")

    n_inf = int(np.isinf(values.to_numpy()).sum())
    if n_inf:
        issues.append(f"{n_inf} infinite values")

    prices = values[["open", "high", "low", "close"]]
    n_nonpos = int((prices <= 0).to_numpy().sum())
    if n_nonpos:
        issues.append(f"{n_nonpos} non-positive prices")

    n_neg_vol = int((values["volume"] < 0).sum())
    if n_neg_vol:
        issues.append(f"{n_neg_vol} negative volumes")
    if not allow_zero_volume:
        n_zero_vol = int((values["volume"] == 0).sum())
        if n_zero_vol:
            issues.append(f"{n_zero_vol} zero-volume bars")

    body_hi = values[["open", "close"]].max(axis=1)
    body_lo = values[["open", "close"]].min(axis=1)
    n_bad_high = int((values["high"] < body_hi).sum())
    n_bad_low = int((values["low"] > body_lo).sum())
    if n_bad_high:
        issues.append(f"{n_bad_high} bars with high below open/close")
    if n_bad_low:
        issues.append(f"{n_bad_low} bars with low above open/close")

    report = ValidationReport(report_rows, tuple(issues))
    if strict:
        report.raise_if_failed()
    return report
