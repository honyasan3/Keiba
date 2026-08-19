"""競走馬・騎手・ドメイン特徴量生成モジュール（走破力・展開負荷・PCI・休養・Elo対戦レーティング完全統合版）"""
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.features.base_feature import BaseFeatureExtractor

logger = setup_logger("horse_features")


class PastPerformanceExtractor(BaseFeatureExtractor):
    """競走馬の過去走、タイム指数、騎手相性、展開ペース、ローテーション、PCI、Eloレーティングを算出するクラス"""

    def __init__(self, recent_runs: int = 3, elo_k_factor: float = 16.0) -> None:
        self.recent_runs = recent_runs
        self.elo_k_factor = elo_k_factor

    def _calc_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """全過去レースを時系列順に走査し、各馬の発走前Eloレートを時系列リークなしで算出"""
        logger.info("競走馬Eloレーティング（多頭数直接対決ネットワーク）の算出を開始します。")
        initial_rating = 1500.0
        ratings = {}  # {horse_id: current_rating}
        pre_race_ratings = {}

        # レース順に反復
        grouped = df.groupby("race_id", sort=False)
        for race_id, race_df in grouped:
            race_horses = race_df["horse_id"].tolist()
            current_race_ratings = {
                h_id: ratings.get(h_id, initial_rating) for h_id in race_horses
            }

            for h_id in race_horses:
                pre_race_ratings[(race_id, h_id)] = current_race_ratings[h_id]

            # 確定着順が存在する場合にレートを更新
            valid_results = race_df[race_df["rank"].notnull() & (race_df["rank"] > 0)].copy()
            if len(valid_results) >= 2:
                valid_results["rank_num"] = pd.to_numeric(valid_results["rank"], errors="coerce")
                valid_results = valid_results.sort_values("rank_num")

                n_horses = len(valid_results)
                deltas = {h_id: 0.0 for h_id in valid_results["horse_id"]}
                horses = valid_results["horse_id"].values
                ranks = valid_results["rank_num"].values

                for i in range(n_horses):
                    for j in range(i + 1, n_horses):
                        h_i, h_j = horses[i], horses[j]
                        r_i, r_j = current_race_ratings[h_i], current_race_ratings[h_j]
                        rank_i, rank_j = ranks[i], ranks[j]

                        # 期待勝率（ロジスティック曲線）
                        exp_i = 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))
                        exp_j = 1.0 - exp_i

                        if rank_i < rank_j:
                            act_i, act_j = 1.0, 0.0
                        elif rank_i > rank_j:
                            act_i, act_j = 0.0, 1.0
                        else:
                            act_i, act_j = 0.5, 0.5

                        k_adj = self.elo_k_factor / (n_horses - 1)
                        deltas[h_i] += k_adj * (act_i - exp_i)
                        deltas[h_j] += k_adj * (act_j - exp_j)

                for h_id, delta in deltas.items():
                    ratings[h_id] = current_race_ratings[h_id] + delta

        # データフレームに結合
        df["horse_elo_rating"] = [
            pre_race_ratings.get((r_id, h_id), initial_rating)
            for r_id, h_id in zip(df["race_id"], df["horse_id"])
        ]
        race_mean_elo = df.groupby("race_id")["horse_elo_rating"].transform("mean")
        df["race_elo_diff_from_mean"] = (df["horse_elo_rating"] - race_mean_elo).round(1)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("競走馬・騎手・ドメイン特徴量（ピュア能力＆Eloレーティング版）の生成を開始します。")
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
        # 1. タイム指数 & 単走PCI（Pace Change Index）の算出
        # ----------------------------------------------------
        if "finish_time_sec" in df.columns:
            group_keys = ["race_date", "course_type", "distance"]
            race_mean_time = df.groupby(group_keys)["finish_time_sec"].transform("mean")
            race_std_time = df.groupby(group_keys)["finish_time_sec"].transform("std").fillna(1.0)
            
            # タイム指数（平均より速いほどプラス値）
            df["speed_index"] = -((df["finish_time_sec"] - race_mean_time) / race_std_time.clip(lower=0.5)) * 10 + 50
        else:
            df["speed_index"] = np.nan

        # 単走PCIの算出 (前半速度 / 後半速度 * 100)
        if "finish_time_sec" in df.columns and "last_3f_time" in df.columns:
            first_dist = (df["distance"] - 600.0).clip(lower=200.0)
            first_time = df["finish_time_sec"] - df["last_3f_time"]
            v_first = first_dist / first_time.clip(lower=1.0)
            v_last = 600.0 / df["last_3f_time"].clip(lower=1.0)
            pci_raw = (v_first / v_last.clip(lower=0.01)) * 100.0

            df["pci"] = np.where(
                df["finish_time_sec"].notnull() & df["last_3f_time"].notnull() & (first_time > 0),
                pci_raw.clip(lower=50.0, upper=150.0),
                np.nan,
            )
            # レース全体のペース前傾度判定
            race_avg_pci = df.groupby("race_id")["pci"].transform("mean")
            df["is_race_high_pace"] = (race_avg_pci >= 104.0).astype(float)
            df["is_race_slow_pace"] = (race_avg_pci <= 96.0).astype(float)
        else:
            df["pci"] = np.nan
            df["is_race_high_pace"] = np.nan
            df["is_race_slow_pace"] = np.nan

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
        
        # レース出走頭数に対する通過割合
        df["horse_avg_passage_rate"] = df["avg_passage_pos"] / df["race_horse_count"].clip(lower=1)
        # 先行馬フラグ
        df["is_front_runner"] = (df["first_corner_pos"] <= 3).astype(float)

        # ----------------------------------------------------
        # 3. 競走馬ごとの過去走集計 (時系列リーク防止: shift(1))
        # ----------------------------------------------------
        logger.info("競走馬の過去走・タイム指数・ローテーション・PCIを算出中...")
        grouped_horse = df.groupby("horse_id")

        df["horse_past_runs"] = grouped_horse.cumcount()

        past_rank_sum = grouped_horse["rank"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        valid_rank_count = grouped_horse["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_avg_rank"] = past_rank_sum / valid_rank_count.replace(0, np.nan)

        past_placed_sum = grouped_horse["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_place_rate"] = past_placed_sum / valid_rank_count.replace(0, np.nan)

        past_win_sum = grouped_horse["rank"].apply(lambda x: (x.shift(1) == 1).astype(float).cumsum()).reset_index(level=0, drop=True)
        df["horse_past_win_rate"] = past_win_sum / valid_rank_count.replace(0, np.nan)

        past_pass_sum = grouped_horse["horse_avg_passage_rate"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        valid_pass_count = grouped_horse["horse_avg_passage_rate"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        df["horse_avg_passage_rate"] = past_pass_sum / valid_pass_count.replace(0, np.nan)

        # 直近3走の平均着順・上がり3F・タイム指数・PCI
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
        df["horse_recent3_avg_pci"] = (
            grouped_horse["pci"]
            .apply(lambda x: x.shift(1).rolling(self.recent_runs, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        ).fillna(100.0)

        # ----------------------------------------------------
        # 4. 展開不利・余力フラグ
        # ----------------------------------------------------
        prev_high_pace = grouped_horse["is_race_high_pace"].shift(1).fillna(0)
        prev_front_pos = grouped_horse["horse_avg_passage_rate"].shift(1).fillna(0.5)
        prev_rank = grouped_horse["rank"].shift(1).fillna(1)
        df["prev_pace_disadvantage_front"] = (
            (prev_high_pace == 1.0) & (prev_front_pos <= 0.30) & (prev_rank >= 5)
        ).astype(float)

        prev_slow_pace = grouped_horse["is_race_slow_pace"].shift(1).fillna(0)
        df["prev_pace_disadvantage_back"] = (
            (prev_slow_pace == 1.0) & (prev_front_pos >= 0.70)
        ).astype(float)

        # ----------------------------------------------------
        # 5. ローテーション・休養・距離ショック・性齢特徴量
        # ----------------------------------------------------
        gender_map = {"牡": 1, "牝": 2, "セ": 3}
        df["gender_cat"] = df["gender"].map(gender_map).fillna(1).astype(int)

        df["age_gender"] = df["gender"].fillna("牡") + df["age"].fillna(3).astype(str)
        df["age_gender_cat"] = df["age_gender"].astype("category").cat.codes

        df["prev_race_date"] = grouped_horse["race_date_dt"].shift(1)
        df["days_since_prev_race"] = (df["race_date_dt"] - df["prev_race_date"]).dt.days

        def _categorize_rest(days):
            if pd.isna(days):
                return 0
            if days <= 7:
                return 1
            if days <= 28:
                return 2
            if days <= 90:
                return 3
            return 4

        df["rest_category_cat"] = df["days_since_prev_race"].apply(_categorize_rest)

        prev2_race_date = grouped_horse["race_date_dt"].shift(2)
        days_prev2_to_prev = (df["prev_race_date"] - prev2_race_date).dt.days
        df["is_second_run_after_rest"] = (
            (days_prev2_to_prev > 90) & (df["days_since_prev_race"] <= 35)
        ).astype(float)

        df["prev_distance"] = grouped_horse["distance"].shift(1)
        df["distance_diff"] = df["distance"] - df["prev_distance"]
        
        df["distance_shock_cat"] = 0
        df.loc[df["distance_diff"] <= -200, "distance_shock_cat"] = -1
        df.loc[df["distance_diff"] >= 200, "distance_shock_cat"] = 1

        df["prev_jockey"] = grouped_horse["jockey_name"].shift(1)
        df["is_jockey_changed"] = (
            (df["prev_jockey"].notna()) & (df["jockey_name"] != df["prev_jockey"])
        ).astype(float)

        if "horse_weight" in df.columns and "horse_weight_diff" in df.columns:
            df["horse_weight_diff_rate"] = (df["horse_weight_diff"] / df["horse_weight"].replace(0, np.nan)) * 100
        else:
            df["horse_weight_diff_rate"] = np.nan

        # ----------------------------------------------------
        # 6. 騎手実績およびコース・枠相性
        # ----------------------------------------------------
        logger.info("騎手実績およびコース・枠順バイアスを算出中...")
        grouped_jockey = df.groupby("jockey_name")

        jockey_valid_runs = grouped_jockey["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        jockey_placed_sum = grouped_jockey["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["jockey_past_place_rate"] = jockey_placed_sum / jockey_valid_runs.replace(0, np.nan)

        jockey_win_sum = grouped_jockey["rank"].apply(lambda x: (x.shift(1) == 1).astype(float).cumsum()).reset_index(level=0, drop=True)
        df["jockey_past_win_rate"] = jockey_win_sum / jockey_valid_runs.replace(0, np.nan)

        df["jockey_venue"] = df["jockey_name"] + "_" + df["race_id"].str[4:6]
        grouped_jv = df.groupby("jockey_venue")
        jv_valid_runs = grouped_jv["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        jv_placed_sum = grouped_jv["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["jockey_venue_place_rate"] = jv_placed_sum / jv_valid_runs.replace(0, np.nan)

        df["course_bracket"] = df["course_type"] + "_" + df["bracket_num"].astype(str)
        grouped_cb = df.groupby("course_bracket")
        cb_valid_runs = grouped_cb["rank"].apply(lambda x: (~x.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
        cb_placed_sum = grouped_cb["is_placed"].apply(lambda x: x.shift(1).cumsum()).reset_index(level=0, drop=True)
        df["course_bracket_place_rate"] = cb_placed_sum / cb_valid_runs.replace(0, np.nan)

        # ----------------------------------------------------
        # 7. レース展開
        # ----------------------------------------------------
        df["race_front_runner_count"] = df.groupby("race_id")["is_front_runner"].transform("sum")

        front_ratio = df["race_front_runner_count"] / df["race_horse_count"].clip(lower=1)
        df["race_expected_pace_cat"] = np.where(
            front_ratio >= 0.30, 3,
            np.where(front_ratio <= 0.15, 1, 2)
        )

        df["pace_match_score"] = np.where(
            df["race_expected_pace_cat"] == 3,
            df["horse_avg_passage_rate"].fillna(0.5),
            np.where(
                df["race_expected_pace_cat"] == 1,
                1.0 - df["horse_avg_passage_rate"].fillna(0.5),
                0.5,
            ),
        ).round(3)

        # ----------------------------------------------------
        # 8. 【フェーズD】競走馬Eloレーティング（直接対決ネットワーク）
        # ----------------------------------------------------
        df = self._calc_elo_ratings(df)

        logger.info("ピュア走破能力＆Eloドメイン特徴量（全46特徴量）の生成が完了しました。")
        return df