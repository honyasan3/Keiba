"""時系列データ分割モジュール"""

from typing import Tuple
import pandas as pd

from src.common.exceptions import ConfigurationError
from src.common.logger import setup_logger

logger = setup_logger("time_splitter")


class TimeSeriesDataSplitter:
    """開催日付に基づいてデータセットを厳格に分割するクラス"""

    @staticmethod
    def split_by_date(
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
        date_col: str = "race_date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        指定された日付境界でデータフレームをTrain / Validation / Testへ分割します。

        Parameters:
            df (pd.DataFrame): 分割対象のデータフレーム
            train_end (str): 訓練データの終了日 (例: '2022-12-31')
            val_end (str): 検証データの終了日 (例: '2023-12-31')
            date_col (str): 日付が格納されている列名

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        if date_col not in df.columns:
            raise ConfigurationError(f"指定された日付列 '{date_col}' がデータフレームに存在しません。")

        # 日付型の変換・ソート
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        t_end = pd.to_datetime(train_end)
        v_end = pd.to_datetime(val_end)

        if t_end >= v_end:
            raise ConfigurationError("train_end は val_end よりも前の日付である必要があります。")

        # 日付による分割
        train_df = df[df[date_col] <= t_end].copy()
        val_df = df[(df[date_col] > t_end) & (df[date_col] <= v_end)].copy()
        test_df = df[df[date_col] > v_end].copy()

        logger.info(
            f"時系列分割完了 - Train: {len(train_df)}件 (~{train_end}), "
            f"Val: {len(val_df)}件 ({train_end}~{val_end}), "
            f"Test: {len(test_df)}件 ({val_end}~)"
        )

        return train_df, val_df, test_df