"""競走馬・騎手・コース相性の過去実績特徴量抽出モジュール"""
import re
from typing import Any, List
import numpy as np
import pandas as pd
from src.features.base_feature import BaseFeatureExtractor
from src.common.logger import setup_logger

logger = setup_logger("horse_features")


class PastPerformanceExtractor(BaseFeatureExtractor):
    """過去走実績、脚質傾向、騎手×競馬場相性などの特徴量を抽出するクラス"""

    def __init__(self, recent_runs: int = 3) -> None:
        self.recent_runs = recent_runs

    def _calc_passage_rate(self, passage_str: Any, horse_count: int) -> float:
        """通過順文字列（例: '1-1-2-2'）から最終コーナー付近の位置割合(0.0:先頭 ~ 1.0:最後方)を算出"""
        if not isinstance(passage_str, str) or not passage_str or horse_count <= 1:
            return np.nan
        try:
            parts = re.findall(r"\d+", passage_str)
            if parts:
                last_pos = int(parts[-1])
                return round(last_pos / horse_count, 3)
        except Exception:
            pass
        return np.nan

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("競走馬・騎手・ドメイン特徴量の生成を開始します。")
        data = df.copy()

        # 日付ソート（時系列整合性の保証）
        data["race_date_dt"] = pd.to_datetime(data["race_date"])
        data = data.sort_values(["race_date_dt", "race_id", "horse_num"]).reset_index(drop=True)

        # レース内頭数
        if "race_horse_count" not in data.columns:
            data["race_horse_count"] = data.groupby("race_id")["horse_num"].transform("count")

        # 競馬場コード (数値型 int に変換)
        data["venue_code"] = data["race_id"].astype(str).str[4:6].astype(int)

        # 1. 競走馬ごとの過去走集計
        logger.info("競走馬の過去走および脚質特徴量を算出中...")
        
        # 1着フラグ・3着以内フラグ・通過順割合の算出
        data["is_win"] = (data["rank"] == 1).astype(float)
        data["is_place"] = (data["rank"].between(1, 3)).astype(float)
        data["passage_rate"] = [
            self._calc_passage_rate(p, c) for p, c in zip(data["passage_order"], data["race_horse_count"])
        ]

        horse_grouped = data.groupby("horse_id")

        # 過去全走の累積集計 (shift(1)で未来リーク遮断)
        data["horse_past_runs"] = horse_grouped.cumcount()
        data["horse_past_wins"] = horse_grouped["is_win"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        data["horse_past_places"] = horse_grouped["is_place"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        data["horse_past_sum_rank"] = horse_grouped["rank"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        data["horse_past_sum_passage"] = horse_grouped["passage_rate"].transform(lambda x: x.shift(1).cumsum()).fillna(0)

        # 過去平均値
        data["horse_past_win_rate"] = np.where(data["horse_past_runs"] > 0, data["horse_past_wins"] / data["horse_past_runs"], 0.0)
        data["horse_past_place_rate"] = np.where(data["horse_past_runs"] > 0, data["horse_past_places"] / data["horse_past_runs"], 0.0)
        data["horse_past_avg_rank"] = np.where(data["horse_past_runs"] > 0, data["horse_past_sum_rank"] / data["horse_past_runs"], np.nan)
        
        # 脚質傾向: 過去平均通過割合
        data["horse_avg_passage_rate"] = np.where(
            data["horse_past_runs"] > 0, data["horse_past_sum_passage"] / data["horse_past_runs"], 0.5
        )

        # 距離変化（前走との距離差: 延長>0, 短縮<0）
        data["prev_distance"] = horse_grouped["distance"].shift(1)
        data["distance_diff"] = (data["distance"] - data["prev_distance"]).fillna(0)

        # 直近N走集計
        data["horse_recent3_avg_rank"] = (
            horse_grouped["rank"]
            .transform(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
        )
        data["horse_recent3_avg_last3f"] = (
            horse_grouped["last_3f_time"]
            .transform(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
        )

        # 前走からの経過日数 (レース間隔)
        prev_date = horse_grouped["race_date_dt"].shift(1)
        data["days_since_prev_race"] = (data["race_date_dt"] - prev_date).dt.days.fillna(999)

        # 2. 騎手の過去実績集計
        logger.info("騎手実績およびコース相性特徴量を算出中...")
        jockey_grouped = data.groupby("jockey_name")
        data["jockey_past_rides"] = jockey_grouped.cumcount()
        data["jockey_past_wins"] = jockey_grouped["is_win"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        data["jockey_past_places"] = jockey_grouped["is_place"].transform(lambda x: x.shift(1).cumsum()).fillna(0)

        data["jockey_past_win_rate"] = np.where(data["jockey_past_rides"] > 0, data["jockey_past_wins"] / data["jockey_past_rides"], 0.0)
        data["jockey_past_place_rate"] = np.where(data["jockey_past_rides"] > 0, data["jockey_past_places"] / data["jockey_past_rides"], 0.0)

        # 騎手 × 競馬場の過去複勝率
        jv_grouped = data.groupby(["jockey_name", "venue_code"])
        data["jv_past_rides"] = jv_grouped.cumcount()
        data["jv_past_places"] = jv_grouped["is_place"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        data["jockey_venue_place_rate"] = np.where(data["jv_past_rides"] > 0, data["jv_past_places"] / data["jv_past_rides"], data["jockey_past_place_rate"])

        # 不要な中間列を削除
        drop_cols = [
            "race_date_dt", "is_win", "is_place", "passage_rate", "prev_distance",
            "horse_past_wins", "horse_past_places", "horse_past_sum_rank", "horse_past_sum_passage",
            "jockey_past_rides", "jockey_past_wins", "jockey_past_places", "jv_past_rides", "jv_past_places"
        ]
        data = data.drop(columns=drop_cols, errors="ignore")

        logger.info("競走馬・騎手・ドメイン特徴量の生成が完了しました。")
        return data