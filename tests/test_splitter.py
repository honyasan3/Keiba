"""時系列分割およびデータリーク検証の単体テスト"""

import pandas as pd
import pytest

from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.dataset.leak_validator import DataLeakageValidator
from src.common.exceptions import DataLeakageError


def test_time_series_data_splitter():
    """日付軸に基づく分割テスト"""
    df = pd.DataFrame([
        {"race_date": "2021-06-01", "val": 10},
        {"race_date": "2023-01-15", "val": 20},
        {"race_date": "2024-02-01", "val": 30},
    ])

    train, val, test = TimeSeriesDataSplitter.split_by_date(
        df, train_end="2022-12-31", val_end="2023-12-31"
    )

    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1


def test_data_leakage_validator_detects_leak():
    """禁止列が特徴量に含まれている場合にエラーが発生するかのテスト"""
    df = pd.DataFrame([{"horse_weight": 480, "rank": 1}])
    
    # 'rank'（確定着順）を特徴量に混ぜた場合は例外がスローされるべき
    illegal_features = ["horse_weight", "rank"]

    with pytest.raises(DataLeakageError):
        DataLeakageValidator.validate_features(df, illegal_features)