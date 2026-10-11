import numpy as np
import pandas as pd
import pytest

from quantlab.data import DataValidationError, validate_ohlcv


def test_generated_data_is_valid(aaa):
    report = validate_ohlcv(aaa)
    assert report.ok
    assert report.n_rows == len(aaa)


def test_missing_columns_flagged(aaa):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_ohlcv(aaa.drop(columns=["volume"]))


def test_non_datetime_index_flagged(aaa):
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        validate_ohlcv(aaa.reset_index(drop=True))


def test_duplicates_and_unsorted_flagged(aaa):
    dup = pd.concat([aaa, aaa.iloc[[0]]])
    report = validate_ohlcv(dup, strict=False)
    assert not report.ok
    assert any("duplicate" in i for i in report.issues)
    assert any("not sorted" in i for i in report.issues)


def test_nan_and_inf_flagged(aaa):
    aaa.iloc[3, aaa.columns.get_loc("close")] = np.nan
    aaa.iloc[5, aaa.columns.get_loc("open")] = np.inf
    report = validate_ohlcv(aaa, strict=False)
    assert any("NaN" in i for i in report.issues)
    assert any("infinite" in i for i in report.issues)


def test_non_positive_price_and_negative_volume(aaa):
    aaa.iloc[0, aaa.columns.get_loc("low")] = 0.0
    aaa.iloc[1, aaa.columns.get_loc("volume")] = -5
    report = validate_ohlcv(aaa, strict=False)
    assert any("non-positive" in i for i in report.issues)
    assert any("negative volumes" in i for i in report.issues)


def test_inconsistent_bar_flagged(aaa):
    aaa.iloc[10, aaa.columns.get_loc("high")] = aaa["close"].iloc[10] * 0.5
    aaa.iloc[11, aaa.columns.get_loc("low")] = aaa["open"].iloc[11] * 1.5
    report = validate_ohlcv(aaa, strict=False)
    assert any("high below" in i for i in report.issues)
    assert any("low above" in i for i in report.issues)


def test_zero_volume_policy(aaa):
    aaa.iloc[2, aaa.columns.get_loc("volume")] = 0
    assert validate_ohlcv(aaa, strict=False).ok
    report = validate_ohlcv(aaa, strict=False, allow_zero_volume=False)
    assert any("zero-volume" in i for i in report.issues)


def test_report_str_and_non_strict_return(aaa):
    bad = aaa.drop(columns=["close"])
    report = validate_ohlcv(bad, strict=False)
    assert "FAILED" in str(report)
    assert "OK" in str(validate_ohlcv(aaa))
