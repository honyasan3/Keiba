"""順位学習（Learning to Rank / LambdaMART）モデルモジュール"""
import os
from typing import List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("ranker_model")


class LGBMRankPredictor:
    """LightGBMのLambdaMARTアルゴリズムを用いたレース内順位学習モデル"""

    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params or {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 1.0,
            "lambda_l2": 5.0,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def _convert_rank_to_relevance(self, rank_series: pd.Series) -> np.ndarray:
        """着順をNDCG計算用の関連度スコア（Relevance）に変換"""
        r = pd.to_numeric(rank_series, errors="coerce").fillna(99)
        relevance = np.where(
            r == 1, 4,
            np.where(r == 2, 3,
            np.where(r == 3, 2,
            np.where(r <= 5, 1, 0)))
        )
        return relevance.astype(int)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.feature_names = feature_cols
        logger.info(f"LambdaMART (LGBMRanker) 学習開始 (特徴量数: {len(feature_cols)}, 訓練件数: {len(train_df)})")

        # 順位学習ではレースIDでソートされている必要がある
        train_sorted = train_df.sort_values(["race_date", "race_id"]).copy()
        val_sorted = val_df.sort_values(["race_date", "race_id"]).copy()

        X_train = train_sorted[feature_cols]
        y_train = self._convert_rank_to_relevance(train_sorted["rank"])
        group_train = train_sorted.groupby("race_id", sort=False).size().values

        X_val = val_sorted[feature_cols]
        y_val = self._convert_rank_to_relevance(val_sorted["rank"])
        group_val = val_sorted.groupby("race_id", sort=False).size().values

        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            callbacks=callbacks,
        )
        logger.info("LambdaMARTモデルの学習が正常に完了しました。")

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """各馬の相対的な強さスコアを出力（同一レース内で高いほど上位着順）"""
        if self.model is None:
            raise ValueError("モデルが学習または読み込まれていません。")
        X_input = df[self.feature_names]
        raw_scores = self.model.predict(X_input, num_iteration=self.model.best_iteration)
        return raw_scores

    def predict_rank_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """レース単位でSoftmaxを適用し、勝率・好走率の擬似確率に正規化"""
        df_temp = df.copy()
        df_temp["_raw_score"] = self.predict_score(df)
        
        # レース内でSoftmax計算（数値安定化のため最大値を減算）
        def _softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probs = df_temp.groupby("race_id")["_raw_score"].transform(_softmax).values
        return probs

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("保存するモデルが存在しません。")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save_model(file_path)
        logger.info(f"LambdaMARTモデルを保存しました: {file_path}")

    def load(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"モデルファイルが存在しません: {file_path}")
        self.model = lgb.Booster(model_file=file_path)
        self.feature_names = self.model.feature_name()
        logger.info(f"LambdaMARTモデルを読み込みました: {file_path}")