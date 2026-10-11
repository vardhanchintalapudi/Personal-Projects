"""Read, write and reshape OHLCV market data.

All loaders return frames with a sorted ``DatetimeIndex`` named ``date`` and
lower-case snake_case columns, and (by default) validate them with
:func:`quantlab.data.schema.validate_ohlcv`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from quantlab.data.schema import validate_ohlcv

_DATE_CANDIDATES = ("date", "datetime", "timestamp", "time")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and convert spaces/punctuation to underscores.

    ``"Adj Close"`` becomes ``"adj_close"``; ``"Volume "`` becomes ``"volume"``.
    """

    def clean(name: object) -> str:
        return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")

    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


def _prepare_frame(
    df: pd.DataFrame,
    *,
    date_col: str | None,
    tz: str | None,
) -> pd.DataFrame:
    df = standardize_columns(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        col = date_col.lower() if date_col else next(
            (c for c in _DATE_CANDIDATES if c in df.columns), None
        )
        if col is None or col not in df.columns:
            raise ValueError(
                f"could not find a date column; tried {_DATE_CANDIDATES}. "
                "Pass date_col explicitly."
            )
        df = df.set_index(pd.to_datetime(df[col])).drop(columns=[col])

    if tz is not None:
        df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)

    df.index.name = "date"
    return df.sort_index()


def load_csv(
    path: str | Path,
    *,
    date_col: str | None = None,
    tz: str | None = None,
    validate: bool = True,
    drop_duplicates: bool = False,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Load one instrument's OHLCV history from a CSV file.

    Parameters
    ----------
    date_col:
        Name of the timestamp column. Auto-detected among
        ``date / datetime / timestamp / time`` when omitted.
    tz:
        Localise naive timestamps to this timezone (or convert aware ones).
    validate:
        Run strict OHLCV validation on the result.
    drop_duplicates:
        Keep the last row for any repeated timestamp instead of failing validation.
    """
    raw = pd.read_csv(path, **read_csv_kwargs)
    df = _prepare_frame(raw, date_col=date_col, tz=tz)
    if drop_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    if validate:
        validate_ohlcv(df, strict=True)
    return df


def load_parquet(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load OHLCV data previously stored with :func:`save_parquet`."""
    df = pd.read_parquet(path)
    df = _prepare_frame(df, date_col=None, tz=None)
    if validate:
        validate_ohlcv(df, strict=True)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path, *, validate: bool = True) -> Path:
    """Write a validated OHLCV frame to Parquet (creating parent folders)."""
    if validate:
        validate_ohlcv(df, strict=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")
    return path


def load_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    *,
    validate: bool = True,
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Load every file matching ``pattern`` into ``{file_stem: frame}``.

    ``.parquet`` files are read with :func:`load_parquet`, everything else with
    :func:`load_csv`.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")

    market: dict[str, pd.DataFrame] = {}
    for file in files:
        if file.suffix == ".parquet":
            market[file.stem] = load_parquet(file, validate=validate)
        else:
            market[file.stem] = load_csv(file, validate=validate, **kwargs)
    return market


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate bars to a coarser frequency.

    ``open`` is the first value, ``high`` the max, ``low`` the min, ``close``
    the last and ``volume`` the sum of the window. Any extra columns (such as
    ``adj_close``) take the last value. Empty windows are dropped.
    """
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in df.columns:
        agg.setdefault(col, "last")
    out = df.resample(rule).agg(agg)
    out = out.dropna(subset=["close"])
    out["volume"] = out["volume"].astype(df["volume"].dtype)
    return out


def to_close_panel(
    market: Mapping[str, pd.DataFrame],
    column: str = "close",
) -> pd.DataFrame:
    """Stack one column from many instruments into a wide ``date x symbol`` frame.

    Timestamps are the union of all indices; missing observations stay NaN so
    the caller decides how to fill them.
    """
    if not market:
        raise ValueError("market is empty")
    panel = pd.concat({sym: df[column] for sym, df in market.items()}, axis=1)
    panel.index.name = "date"
    return panel.sort_index()
