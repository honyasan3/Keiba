"""機械学習モデル基底インターフェース"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """予想モデル共通の基底クラス"""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """モデルの学習を実行"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """勝率または複勝圏内率（確率値）を予測"""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """学習済みモデルの保存"""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """学習済みモデルの読み込み"""
        pass