"""競走馬・騎手・ドメイン特徴量生成モジュール（タイム指数・展開ペース対応完全版）"""
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.features.base_feature import BaseFeatureExtractor

logger = setup_logger("horse_features")


class PastPerformanceExtractor(BaseFeatureExtractor):
    """競走馬の過去走、タイム指数、騎手相性、展開ペース特徴量を算出するクラス"""

    def __init__(self, recent_runs: int = 3) -> None:
        self.recent_runs = recent_runs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("競走馬・騎手・ドメイン特徴量の生成を開始します。")
        df = df.copy()

        # 日付型変換と時系列ソート
        df["race_date_dt"] = pd.to_datetime(df["race_date"])
        df = df.sort_values(["race_date_dt", "race_id", "horse_num"]).reset_index(drop=True)

        # ターゲット変数（3着以内フラグ）
        if "rank" in df.columns:
            df["is_placed"] = (df["rank"] <= 3).astype(float)
            df.loc[df["rank"].isna(), "is_placed"] = np.nan
        else:
            df["is_placed"] = np.nan

        # ----------------------------------------------------
        # 1. タイム指数の算出（同日・同コース・同距離の偏差値）
        # ----------------------------------------------------
        if "finish_time_sec" in df.columns:
            group_keys = ["race_date", "course_type", "distance"]
            race_mean_time = df.groupby(group_keys)["finish_time_sec"].transform("mean")
            race_std_time = df.groupby(group_keys)["finish_time_sec"].transform("std").fillna(1.0)
            
            # タイム指数（平均より速いほどプラス値になるよう反転）
            df["speed_index"] = -((df["finish_time_sec"] - race_mean_time) / race_std_time.clip(lower=0.5)) * 10 + 50
        else:
            df["speed_index"] = np.nan

        # ----------------------------------------------------
        # 2. 通過順位から脚質（通過割合・1角位置）の算出
        # ----------------------------------------------------
        def _calc_passage_metrics(passage: str):
            if not passage or pd.isna(passage):
                return np.nan, np.nan
            parts = str(passage).split("-")
            try:
                nums = [float(p) for p in parts if p.isdigit() or p.replace(".", "", 1).isdigit()]
                if not nums:
                    return np.nan, np.nan
                first_corner = nums[0]
                avg_pos = np.mean(nums)
                return first_corner, avg_pos
            except Exception:
                return np.nan, np.nan

        metrics = df["passage_order"].apply(_calc_passage_metrics)
        df["first_corner_pos"] = [m[0] for m in metrics]
        df["avg_passage_pos"] = [m[1] for m in metrics]
        
        # レース出走頭数に対する通過割合（0に近いほど逃げ・先行、1に近いほど追込）
        df["horse_avg_passage_rate"] = df["avg_passage_pos"] / df["race_horse_count"].clip(lower=1)
        # 先行馬フラグ（1コーナー3番手以内）
        df["is_front_runner"] = (df["first_corner_pos"] <= 3).astype(float)

        # ----------------------------------------------------
        # 3. 競走馬ごとの過去走集計 (時系列リーク防止: shift(1))
        # ----------------------------------------------------
        logger.info("競走馬の過去走・タイム指数を算出中...")
        grouped_horse = df.groupby("horse_id")

        # 過去出走数
        df["horse_past_runs"] = grouped_horse.cumcount()

        # 過去平均着順・勝率・複勝率・脚質平均
        past_rank_sum = grouped_horse["rank"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        valid_rank_count = grouped_horse["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_avg_rank"] = past_rank_sum / valid_rank_count.replace(0, np.nan)

        past_placed_sum = grouped_horse["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_place_rate"] = past_placed_sum / valid_rank_count.replace(0, np.nan)

        past_win_sum = grouped_horse["rank"].apply(lambda x: (x.shift(1) == 1).astype(float).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_win_rate"] = past_win_sum / valid_rank_count.replace(0, np.nan)

        # 過去平均通過割合
        past_pass_sum = grouped_horse["horse_avg_passage_rate"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        valid_pass_count = grouped_horse["horse_avg_passage_rate"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        df["horse_avg_passage_rate"] = past_pass_sum / valid_pass_count.replace(0, np.nan)

        # 直近3走の平均着順・上がり3F・タイム指数
        df["horse_recent3_avg_rank"] = (
            grouped_horse["rank"]
            .apply(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df["horse_recent3_avg_last3f"] = (
            grouped_horse["last_3f_time"]
            .apply(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df["horse_recent3_avg_speed_index"] = (
            grouped_horse["speed_index"]
            .apply(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

        # レース間隔日数
        df["prev_race_date"] = grouped_horse["race_date_dt"].shift(1)
        df["days_since_prev_race"] = (df["race_date_dt"] - df["prev_race_date"]).dt.days

        # 距離変化
        df["prev_distance"] = grouped_horse["distance"].shift(1)
        df["distance_diff"] = df["distance"] - df["prev_distance"]

        # 体重増減率（%）
        if "horse_weight" in df.columns and "horse_weight_diff" in df.columns:
            df["horse_weight_diff_rate"] = (df["horse_weight_diff"] / df["horse_weight"].replace(0, np.nan)) * 100
        else:
            df["horse_weight_diff_rate"] = np.nan

        # ----------------------------------------------------
        # 4. 騎手実績およびコース・枠相性
        # ----------------------------------------------------
        logger.info("騎手実績およびコース・枠順バイアスを算出中...")
        grouped_jockey = df.groupby("jockey_name")

        jockey_valid_runs = grouped_jockey["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        jockey_placed_sum = grouped_jockey["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["jockey_past_place_rate"] = jockey_placed_sum / jockey_valid_runs.replace(0, np.nan)

        jockey_win_sum = grouped_jockey["rank"].apply(lambda x: (x.shift(1) == 1).astype(float).cumsum()).reset_index(level=0, drop=True)
        df["jockey_past_win_rate"] = jockey_win_sum / jockey_valid_runs.replace(0, np.nan)

        # 騎手 × 競馬場相性
        df["jockey_venue"] = df["jockey_name"] + "_" + df["race_id"].str[4:6]
        grouped_jv = df.groupby("jockey_venue")
        jv_valid_runs = grouped_jv["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        jv_placed_sum = grouped_jv["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["jockey_venue_place_rate"] = jv_placed_sum / jv_valid_runs.replace(0, np.nan)

        # コース種別 × 枠番の好走率（枠順バイアス）
        df["course_bracket"] = df["course_type"] + "_" + df["bracket_num"].astype(str)
        grouped_cb = df.groupby("course_bracket")
        cb_valid_runs = grouped_cb["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        cb_placed_sum = grouped_cb["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["course_bracket_place_rate"] = cb_placed_sum / cb_valid_runs.replace(0, np.nan)

        # ----------------------------------------------------
        # 5. レース展開（先行馬頭数）
        # ----------------------------------------------------
        # 前走先行していた馬が対象レース内に何頭いるか
        df["race_front_runner_count"] = df.groupby("race_id")["is_front_runner"].transform("sum")

        logger.info("高度特徴量の生成が完了しました。")
        return df