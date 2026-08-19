"""モンテカルロ・レースシミュレータモジュール（コース種別適性・スケーリング補正版）"""
import numpy as np
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("race_simulator")


class MonteCarloRaceSimulator:
    """各馬の走破タイム分布から1万回仮想レースを実行するシミュレータ"""

    def __init__(self, n_simulations: int = 10000) -> None:
        self.n_simulations = n_simulations

    def simulate_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        出走馬データから仮想走破タイムを生成し、着順確率・勝率を算出
        :param race_df: 1レース分の出走馬DataFrame
        """
        n_horses = len(race_df)
        if n_horses < 2:
            logger.warning("出走頭数が少なすぎるためシミュレーションをスキップします。")
            return race_df

        horses = race_df.copy().reset_index(drop=True)

        # 1. 各馬の能力値（平均スピード指数）
        # 欠損値は全体の平均的なベース(48.0)で補完
        base_speed = horses["horse_recent3_avg_speed_index"].fillna(50.0).values

        # 異常値クリッピング (極端な外れ値を 35〜65 の間に丸める)
        base_speed = np.clip(base_speed, 38.0, 62.0)

        # 騎手実績（勝率・複勝率）による加点
        jockey_win = horses["jockey_past_win_rate"].fillna(0.05).values
        jockey_place = horses["jockey_past_place_rate"].fillna(0.15).values
        jockey_boost = (jockey_win * 5.0) + (jockey_place * 3.0)

        # 直近着順による補正
        recent_rank = horses["horse_recent3_avg_rank"].fillna(8.0).values
        rank_penalty = (recent_rank - 1.0) * 0.4

        # 平均能力値 (mu)
        mu = base_speed + jockey_boost - rank_penalty

        # 2. タイムのばらつき (sigma): 過去走数が多い馬ほど安定
        runs = horses["horse_past_runs"].fillna(1).values
        sigma = np.clip(2.5 / np.sqrt(np.maximum(runs, 1)), 1.5, 3.0)

        # 3. モンテカルロシミュレーション実行 (10,000回 × 出走頭数)
        rng = np.random.default_rng(seed=42)
        simulated_speeds = rng.normal(loc=mu, scale=sigma, size=(self.n_simulations, n_horses))

        # スピード指数が大きい順に着順決定
        ranks = n_horses - np.argsort(np.argsort(simulated_speeds, axis=1), axis=1)

        # 4. 着順確率の集計
        win_counts = np.sum(ranks == 1, axis=0)
        top2_counts = np.sum(ranks <= 2, axis=0)
        top3_counts = np.sum(ranks <= 3, axis=0)

        horses["sim_win_prob"] = win_counts / self.n_simulations
        horses["sim_top2_prob"] = top2_counts / self.n_simulations
        horses["sim_place_prob"] = top3_counts / self.n_simulations
        horses["sim_rank"] = horses["sim_win_prob"].rank(ascending=False, method="min").astype(int)

        # 5. アンサンブル複勝率（LightGBM 70% : シミュレータ 30%）
        if "pred_place_prob" in horses.columns:
            horses["ensemble_place_prob"] = (
                horses["pred_place_prob"] * 0.70 + horses["sim_place_prob"] * 0.30
            )
        else:
            horses["ensemble_place_prob"] = horses["sim_place_prob"]

        return horses