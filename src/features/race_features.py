"""レース属性およびレース内相対・ペース予測特徴量生成モジュール"""
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.features.base_feature import BaseFeatureExtractor

logger = setup_logger("race_features")


class RaceFeatureExtractor(BaseFeatureExtractor):
    """レース基本情報・出走頭数・斤量差・想定ペース特徴量を抽出するクラス"""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("レース属性・相対特徴量・想定ペース特徴量の生成を開始します。")
        df_out = df.copy()

        # 競馬場コード
        df_out["venue_code"] = (
            df_out["race_id"].astype(str).str[4:6].astype(int, errors="ignore")
        )

        # カテゴリ変数のマッピング
        course_type_map = {"芝": 1, "ダート": 2, "障害": 3}
        df_out["course_type_cat"] = (
            df_out["course_type"].map(course_type_map).fillna(0).astype(int)
        )

        weather_map = {"晴": 1, "曇": 2, "小雨": 3, "雨": 4, "雪": 5}
        df_out["weather_cat"] = (
            df_out["weather"].map(weather_map).fillna(0).astype(int)
        )

        track_condition_map = {"良": 1, "稍": 2, "稍重": 2, "重": 3, "不良": 4}
        df_out["track_condition_cat"] = (
            df_out["track_condition"].map(track_condition_map).fillna(0).astype(int)
        )

        # レース出走頭数
        race_horse_counts = (
            df_out.groupby("race_id")["horse_num"].transform("count")
        )
        df_out["race_horse_count"] = race_horse_counts

        # レース内平均斤量との差
        race_mean_weight = df_out.groupby("race_id")["jockey_weight"].transform("mean")
        df_out["jockey_weight_diff_from_race_mean"] = (
            df_out["jockey_weight"] - race_mean_weight
        ).round(2)

        # 距離の数値化
        df_out["distance"] = pd.to_numeric(df_out["distance"], errors="coerce").fillna(1600)
        df_out["race_round"] = pd.to_numeric(df_out["race_round"], errors="coerce").fillna(11)

        logger.info("レース属性・相対特徴量の生成が完了しました。")
        return df_out