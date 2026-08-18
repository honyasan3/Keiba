"""LightGBMによる競馬予想モデル実装"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.models.base_model import BaseModel

logger = setup_logger("lgbm_model")


class LGBMRacePredictor(BaseModel):
    """LightGBMを用いた着順・入着確率予測クラス"""

    def __init__(self, custom_params: Optional[Dict[str, Any]] = None) -> None:
        # デフォルトハイパーパラメータ（2値分類・不均衡データ考慮）
        self.params: Dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": 42,
        }
        if custom_params:
            self.params.update(custom_params)

        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        """LightGBMモデルの学習を実行します。"""
        self.feature_names = list(X_train.columns)
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        logger.info(f"LightGBM学習開始 (特徴量数: {len(self.feature_names)}, 訓練件数: {len(X_train)})")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        logger.info("LightGBMの学習が正常に完了しました。")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """各出走馬が目的変数（例: 3着以内）を満たす確率を出力します。"""
        if self.model is None:
            raise RuntimeError("モデルが学習されていません。train() または load() を実行してください。")
        X_input = X[self.feature_names]
        return self.model.predict(X_input, num_iteration=self.model.best_iteration)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """特徴量重要度を取得"""
        if self.model is None:
            raise RuntimeError("モデルが未学習です。")
        importance = self.model.feature_importance(importance_type="gain")
        df_importance = (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return df_importance

    def save(self, filepath: str) -> None:
        """モデルをファイルへ保存"""
        if self.model is None:
            raise RuntimeError("モデルが未学習です。")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(filepath)
        logger.info(f"モデルを保存しました: {filepath}")

    def load(self, filepath: str) -> None:
        """モデルをファイルから読み込み"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {filepath}")
        self.model = lgb.Booster(model_file=filepath)
        self.feature_names = self.model.feature_name()
        logger.info(f"モデルを読み込みました: {filepath}")