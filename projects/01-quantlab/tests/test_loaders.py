import pandas as pd
import pytest

from quantlab.data import (
    DataValidationError,
    load_csv,
    load_directory,
    load_parquet,
    resample_ohlcv,
    save_parquet,
    standardize_columns,
    to_close_panel,
)


def test_standardize_columns():
    df = pd.DataFrame(columns=["Date", "Adj Close", " Volume ", "Open-Price"])
    assert list(standardize_columns(df).columns) == ["date", "adj_close", "volume", "open_price"]


def test_csv_roundtrip_with_messy_headers(tmp_path, aaa):
    messy = aaa.reset_index()
    messy.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    path = tmp_path / "AAA.csv"
    messy.to_csv(path, index=False)

    loaded = load_csv(path)
    assert loaded.index.name == "date"
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(loaded, aaa, check_freq=False, atol=1e-8, rtol=1e-8)


def test_csv_unsorted_is_sorted_and_duplicates_handled(tmp_path, aaa):
    shuffled = aaa.sample(frac=1.0, random_state=0).reset_index()
    dup = pd.concat([shuffled, shuffled.iloc[[0]]])
    path = tmp_path / "dup.csv"
    dup.to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="duplicate"):
        load_csv(path)
    clean = load_csv(path, drop_duplicates=True)
    assert clean.index.is_monotonic_increasing
    assert len(clean) == len(aaa)


def test_csv_timezone_localisation(tmp_path, aaa):
    path = tmp_path / "tz.csv"
    aaa.reset_index().to_csv(path, index=False)
    loaded = load_csv(path, tz="Asia/Kolkata")
    assert str(loaded.index.tz) == "Asia/Kolkata"


def test_missing_date_column_errors(tmp_path):
    path = tmp_path / "nodate.csv"
    pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="date column"):
        load_csv(path)


def test_parquet_roundtrip(tmp_path, aaa):
    path = save_parquet(aaa, tmp_path / "nested" / "AAA.parquet")
    assert path.exists()
    loaded = load_parquet(path)
    pd.testing.assert_frame_equal(loaded, aaa, check_freq=False)


def test_save_parquet_rejects_invalid(tmp_path, aaa):
    aaa.iloc[0, aaa.columns.get_loc("close")] = -1
    with pytest.raises(DataValidationError):
        save_parquet(aaa, tmp_path / "bad.parquet")


def test_load_directory_mixed(tmp_path, market):
    market["AAA"].reset_index().to_csv(tmp_path / "AAA.csv", index=False)
    market["BBB"].reset_index().to_csv(tmp_path / "BBB.csv", index=False)
    loaded = load_directory(tmp_path)
    assert set(loaded) == {"AAA", "BBB"}

    save_parquet(market["CCC"], tmp_path / "CCC.parquet")
    pq = load_directory(tmp_path, "*.parquet")
    assert set(pq) == {"CCC"}

    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path, "*.nothing")


def test_resample_ohlcv_semantics(aaa):
    weekly = resample_ohlcv(aaa, "W-FRI")
    first_week = aaa.iloc[:5]
    row = weekly.iloc[0]
    assert row["open"] == first_week["open"].iloc[0]
    assert row["close"] == first_week["close"].iloc[-1]
    assert row["high"] == first_week["high"].max()
    assert row["low"] == first_week["low"].min()
    assert row["volume"] == first_week["volume"].sum()
    assert weekly["volume"].dtype == aaa["volume"].dtype
    assert weekly["volume"].sum() == aaa["volume"].sum()


def test_to_close_panel(market):
    panel = to_close_panel(market)
    assert list(panel.columns) == ["AAA", "BBB", "CCC"]
    assert panel.index.name == "date"
    assert panel.shape == (500, 3)
    with pytest.raises(ValueError):
        to_close_panel({})
