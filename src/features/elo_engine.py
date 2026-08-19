"""競走馬・騎手の多頭数対戦レーティング（Elo Rating）計算エンジン"""
import numpy as np
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("elo_engine")


class EloRatingEngine:
    """レース結果（直接対決）から時系列リークなしで競走馬のEloレーティングを更新・算出するクラス"""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 16.0) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = {}  # {horse_id: current_rating}

    def compute_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレーム全体を日付順に走査し、発走前Eloレートとレース内相対レートを付与"""
        logger.info("競走馬Eloレーティング（直接対決ネットワーク）の時系列算出を開始します。")
        df_out = df.copy()
        
        # 日付とレースIDで時系列ソート
        df_out["_dt"] = pd.to_datetime(df_out["race_date"])
        df_out = df_out.sort_values(["_dt", "race_id", "horse_num"]).reset_index(drop=True)

        horse_elo_list = []
        
        # レースごとにグループ化して時系列処理
        grouped = df_out.groupby("race_id", sort=False)
        
        # 結果格納用辞書 {(race_id, horse_id): pre_race_rating}
        pre_race_ratings = {}

        for race_id, race_df in grouped:
            # 1. 各馬の発走前レートを取得（初出走は初期値 1500）
            race_horses = race_df["horse_id"].tolist()
            current_race_ratings = {
                h_id: self.ratings.get(h_id, self.initial_rating) for h_id in race_horses
            }

            for h_id in race_horses:
                pre_race_ratings[(race_id, h_id)] = current_race_ratings[h_id]

            # 2. レース結果（確定着順）がある場合のみレートを更新
            valid_results = race_df[race_df["rank"].notnull() & (race_df["rank"] > 0)].copy()
            if len(valid_results) >= 2:
                valid_results["rank_num"] = pd.to_numeric(valid_results["rank"], errors="coerce")
                valid_results = valid_results.sort_values("rank_num")
                
                n_horses = len(valid_results)
                # 各馬のレート変動量（デルタ）を計算
                deltas = {h_id: 0.0 for h_id in valid_results["horse_id"]}
                
                # ペアワイズ直接対決（1対1の対戦の総和）
                horses = valid_results["horse_id"].values
                ranks = valid_results["rank_num"].values
                
                for i in range(n_horses):
                    for j in range(i + 1, n_horses):
                        h_i, h_j = horses[i], horses[j]
                        r_i, r_j = current_race_ratings[h_i], current_race_ratings[h_j]
                        rank_i, rank_j = ranks[i], ranks[j]
                        
                        # 期待勝率 (Logistic)
                        exp_i = 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))
                        exp_j = 1.0 - exp_i
                        
                        # 実際の結果 (同着は0.5)
                        if rank_i < rank_j:
                            act_i, act_j = 1.0, 0.0
                        elif rank_i > rank_j:
                            act_i, act_j = 0.0, 1.0
                        else:
                            act_i, act_j = 0.5, 0.5
                            
                        # 出走頭数で正規化したKファクター
                        k_adj = self.k_factor / (n_horses - 1)
                        deltas[h_i] += k_adj * (act_i - exp_i)
                        deltas[h_j] += k_adj * (act_j - exp_j)

                # レーティングの永続更新
                for h_id, delta in deltas.items():
                    self.ratings[h_id] = current_race_ratings[h_id] + delta

        # 発走前Eloレートをデータフレームに結合
        df_out["horse_elo_rating"] = df_out.apply(
            lambda row: pre_race_ratings.get((row["race_id"], row["horse_id"]), self.initial_rating),
            axis=1
        )
        
        # レース内平均Eloとの差分（突出度）
        race_mean_elo = df_out.groupby("race_id")["horse_elo_rating"].transform("mean")
        df_out["race_elo_diff_from_mean"] = (df_out["horse_elo_rating"] - race_mean_elo).round(1)

        df_out = df_out.drop(columns=["_dt"], errors="ignore")
        logger.info("競走馬Eloレーティングの算出が完了しました。")
        return df_out