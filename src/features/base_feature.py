"""特徴量エンジニアリング基底モジュール"""
from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureExtractor(ABC):
    """特徴量抽出器の抽象基底クラス"""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生データ（または結合済みDataFrame）を受け取り、特徴量を付与したDataFrameを返します。
        """
        pass