"""Bar-replay data feed with structural protection against lookahead bias.

The feed owns the full history but only ever exposes what has already been
published. Strategies receive the feed and query it through :meth:`history`,
which slices at an internal cursor, so there is no API through which future
bars can be read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce

import numpy as np
import pandas as pd

from quantlab.config import OHLCV_COLUMNS
from quantlab.data.schema import validate_ohlcv
from quantlab.events import Bar, MarketEvent


class HistoricalDataFeed:
    """Replays a multi-instrument market one timestamp at a time.

    Parameters
    ----------
    market:
        ``{symbol: OHLCV DataFrame}`` as produced by
        :func:`quantlab.data.generate_ohlcv_panel` or the loaders.
    validate:
        Validate every frame with :func:`quantlab.data.validate_ohlcv`.

    The timeline is the union of all symbols' timestamps. A symbol without a
    bar at a given timestamp is absent from that :class:`MarketEvent`.

    Iterate with ``for event in feed`` or call :meth:`step` manually.
    """

    def __init__(self, market: Mapping[str, pd.DataFrame], *, validate: bool = True) -> None:
        if not market:
            raise ValueError("market is empty")

        frames: dict[str, pd.DataFrame] = {}
        for symbol, df in market.items():
            if validate:
                validate_ohlcv(df, strict=True)
            frames[symbol] = df.loc[:, list(OHLCV_COLUMNS)]

        tz_flags = {df.index.tz is None for df in frames.values()}
        if len(tz_flags) > 1:
            raise ValueError("cannot mix timezone-aware and naive indices")

        self._frames = frames
        self._symbols = tuple(frames)
        self._timeline: pd.DatetimeIndex = reduce(
            lambda a, b: a.union(b), (df.index for df in frames.values())
        )
        self._rows = {s: df.to_numpy(dtype="float64") for s, df in frames.items()}
        self._locator = {s: df.index.get_indexer(self._timeline) for s, df in frames.items()}
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Rewind to before the first bar."""
        self._step = 0
        self._cursor = {s: 0 for s in self._symbols}
        self._last_close: dict[str, float] = {}

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def timeline(self) -> pd.DatetimeIndex:
        """Every timestamp the feed will publish (known in advance, like a calendar)."""
        return self._timeline

    @property
    def current_time(self) -> pd.Timestamp | None:
        """Timestamp of the latest published bar, or ``None`` before the first step."""
        return self._timeline[self._step - 1] if self._step else None

    def has_next(self) -> bool:
        return self._step < len(self._timeline)

    def __len__(self) -> int:
        return len(self._timeline)

    # -------------------------------------------------------------- iteration

    def step(self) -> MarketEvent:
        """Publish the next timestamp and return it as a :class:`MarketEvent`."""
        if not self.has_next():
            raise IndexError("feed exhausted")
        i = self._step
        bars: dict[str, Bar] = {}
        for symbol in self._symbols:
            pos = self._locator[symbol][i]
            if pos >= 0:
                o, h, low, c, v = self._rows[symbol][pos]
                bars[symbol] = Bar(o, h, low, c, v)
                self._cursor[symbol] = int(pos) + 1
                self._last_close[symbol] = float(c)
        self._step += 1
        return MarketEvent(timestamp=self._timeline[i], bars=bars)

    def __iter__(self) -> HistoricalDataFeed:
        return self

    def __next__(self) -> MarketEvent:
        if not self.has_next():
            raise StopIteration
        return self.step()

    # ---------------------------------------------------------------- queries

    def bars_seen(self, symbol: str) -> int:
        """Number of bars already published for ``symbol``."""
        self._check_symbol(symbol)
        return self._cursor[symbol]

    def last_close(self, symbol: str) -> float | None:
        """Most recent published close (carried forward across missing bars)."""
        self._check_symbol(symbol)
        return self._last_close.get(symbol)

    def last_prices(self) -> dict[str, float]:
        """Latest known close for every symbol that has traded so far."""
        return dict(self._last_close)

    def history(self, symbol: str, n: int | None = None, field: str = "close") -> pd.Series:
        """Last ``n`` published values of ``field`` (all of them when ``n`` is None).

        The current bar is included; nothing after it ever is.
        """
        self._check_symbol(symbol)
        if field not in OHLCV_COLUMNS:
            raise ValueError(f"field must be one of {OHLCV_COLUMNS}, got {field!r}")
        if n is not None and n <= 0:
            raise ValueError("n must be positive")
        end = self._cursor[symbol]
        start = 0 if n is None else max(0, end - n)
        return self._frames[symbol][field].iloc[start:end]

    def history_frame(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Last ``n`` published OHLCV rows for ``symbol``."""
        self._check_symbol(symbol)
        if n is not None and n <= 0:
            raise ValueError("n must be positive")
        end = self._cursor[symbol]
        start = 0 if n is None else max(0, end - n)
        return self._frames[symbol].iloc[start:end]

    def history_panel(
        self,
        symbols: Sequence[str] | None = None,
        n: int | None = None,
        field: str = "close",
        how: str = "inner",
    ) -> pd.DataFrame:
        """Wide ``timestamp x symbol`` view of recent history.

        ``n`` bars are taken per symbol before joining, so with ``how="inner"``
        the result can have fewer than ``n`` rows when timelines differ.
        """
        symbols = list(symbols) if symbols is not None else list(self._symbols)
        if not symbols:
            raise ValueError("symbols is empty")
        series = {s: self.history(s, n, field) for s in symbols}
        panel = pd.concat(series, axis=1, join=how)
        return panel.sort_index()

    def window(self, symbol: str, n: int, field: str = "close") -> np.ndarray | None:
        """Last ``n`` values as a numpy array, or ``None`` if fewer than ``n`` exist.

        Convenience for indicator code that needs a full lookback window.
        """
        values = self.history(symbol, n, field)
        if len(values) < n:
            return None
        return values.to_numpy(dtype="float64")

    # ---------------------------------------------------------------- helpers

    def _check_symbol(self, symbol: str) -> None:
        if symbol not in self._frames:
            raise KeyError(f"unknown symbol: {symbol!r}")
