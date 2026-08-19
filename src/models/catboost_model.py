"""CatBoost 分類モデルモジュール（BaseModel準拠・trainメソッド実装版）"""
import os
from typing import List, Optional
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.models.base_model import BaseModel

logger = setup_logger("catboost_model")


class CatBoostRacePredictor(BaseModel):
    """CatBoostを用いた複勝予測モデルクラス"""

    def __init__(self, cat_features: Optional[List[str]] = None) -> None:
        self.cat_features = cat_features or []
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
    ) -> None:
        """BaseModelの抽象メソッドを実装"""
        self.feature_names = list(X_train.columns)
        
        # 実際にデータフレームに存在するカテゴリ変数のみ抽出
        cat_cols = [c for c in self.cat_features if c in self.feature_names]
        
        # 欠損値補完（CatBoost用にカテゴリ型/数値型を安定化）
        X_train_clean = X_train.copy()
        for c in cat_cols:
            X_train_clean[c] = X_train_clean[c].fillna(-1).astype(int)

        train_pool = Pool(X_train_clean, y_train, cat_features=cat_cols)

        val_pool = None
        if X_val is not None and y_val is not None:
            X_val_clean = X_val.copy()
            for c in cat_cols:
                X_val_clean[c] = X_val_clean[c].fillna(-1).astype(int)
            val_pool = Pool(X_val_clean, y_val, cat_features=cat_cols)

        logger.info(f"CatBoost学習開始 (特徴量数: {len(self.feature_names)}, 訓練件数: {len(X_train)})")

        self.model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100,
        )

        self.model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        logger.info("CatBoostの学習が正常に完了しました。")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("モデルが学習されていないか、読み込まれていません。")

        X_input = X[self.feature_names].copy()
        cat_cols = [c for c in self.cat_features if c in self.feature_names]
        for c in cat_cols:
            X_input[c] = X_input[c].fillna(-1).astype(int)

        # 3着以内（クラス1）の予測確率を返す
        return self.model.predict_proba(X_input)[:, 1]

    def save(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("保存するモデルが存在しません。")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save_model(file_path)
        logger.info(f"CatBoostモデルを保存しました: {file_path}")

    def load(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"モデルファイルが存在しません: {file_path}")
        self.model = CatBoostClassifier()
        self.model.load_model(file_path)
        self.feature_names = self.model.feature_names_
        logger.info(f"CatBoostモデルを読み込みました: {file_path}")