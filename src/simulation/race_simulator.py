"""モンテカルロ・レースシミュレータモジュール（ワイド・馬連・ケリー基準対応版）"""
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("race_simulator")


class MonteCarloRaceSimulator:
    """各馬の走破タイム分布から1万回仮想レースを実行し、複合券種確率とケリー資金配分を算出"""

    def __init__(self, n_simulations: int = 10000) -> None:
        self.n_simulations = n_simulations

    def simulate_race(
        self, race_df: pd.DataFrame, bankroll: int = 10000, kelly_fraction: float = 0.25
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        :param race_df: 1レース分の出走馬DataFrame
        :param bankroll: レース軍資金 (デフォルト1万円)
        :param kelly_fraction: フラクショナル・ケリー係数 (安全運用のため0.25)
        :return: (単複結果DF, ワイド組み合わせDF, 馬連組み合わせDF)
        """
        n_horses = len(race_df)
        if n_horses < 2:
            logger.warning("出走頭数が少なすぎるためシミュレーションをスキップします。")
            return race_df, pd.DataFrame(), pd.DataFrame()

        horses = race_df.copy().reset_index(drop=True)

        # 1. 各馬の能力値 (mu) と ばらつき (sigma) の設定
        base_speed = horses["horse_recent3_avg_speed_index"].fillna(50.0).values
        base_speed = np.clip(base_speed, 38.0, 62.0)

        jockey_win = horses["jockey_past_win_rate"].fillna(0.05).values
        jockey_place = horses["jockey_past_place_rate"].fillna(0.15).values
        jockey_boost = (jockey_win * 5.0) + (jockey_place * 3.0)

        recent_rank = horses["horse_recent3_avg_rank"].fillna(8.0).values
        rank_penalty = (recent_rank - 1.0) * 0.4

        mu = base_speed + jockey_boost - rank_penalty

        runs = horses["horse_past_runs"].fillna(1).values
        sigma = np.clip(2.5 / np.sqrt(np.maximum(runs, 1)), 1.5, 3.0)

        # 2. 1万回シミュレーション実行
        rng = np.random.default_rng(seed=42)
        simulated_speeds = rng.normal(loc=mu, scale=sigma, size=(self.n_simulations, n_horses))
        # 降順で着順決定 (1位〜n_horses位)
        ranks = n_horses - np.argsort(np.argsort(simulated_speeds, axis=1), axis=1)

        # 3. 単勝・複勝確率集計
        win_counts = np.sum(ranks == 1, axis=0)
        top2_counts = np.sum(ranks <= 2, axis=0)
        top3_counts = np.sum(ranks <= 3, axis=0)

        horses["sim_win_prob"] = win_counts / self.n_simulations
        horses["sim_top2_prob"] = top2_counts / self.n_simulations
        horses["sim_place_prob"] = top3_counts / self.n_simulations
        horses["sim_rank"] = horses["sim_win_prob"].rank(ascending=False, method="min").astype(int)

        # アンサンブル複勝率
        if "pred_place_prob" in horses.columns:
            horses["ensemble_place_prob"] = (
                horses["pred_place_prob"] * 0.70 + horses["sim_place_prob"] * 0.30
            )
        else:
            horses["ensemble_place_prob"] = horses["sim_place_prob"]

        # 4. フラクショナル・ケリー基準による複勝推奨金額計算
        # ケリー基準式: f* = (p * b - q) / b (p:勝率, b:純オッズ(odds-1), q:敗率(1-p))
        def calc_kelly_bet(row) -> int:
            p = row["ensemble_place_prob"]
            b = max(row["place_odds_est"] - 1.0, 0.1)
            q = 1.0 - p
            f_star = (p * b - q) / b
            if f_star <= 0:
                return 0
            # 安全率（kelly_fraction: 1/4ケリー）を掛けて100円単位に丸める
            bet = int(bankroll * f_star * kelly_fraction / 100) * 100
            return max(bet, 0)

        # 複勝オッズは、オッズAPIの実測レンジ(real_place_odds_min/max)が入力DataFrameにあれば
        # それを優先する（保守的に見て下限値を採用）。無ければ単勝オッズからの近似式にフォールバックする。
        approx_odds_est = (horses["odds"].fillna(1.0) ** 0.45).clip(lower=1.1)
        if "real_place_odds_min" in horses.columns:
            horses["place_odds_est"] = horses["real_place_odds_min"].fillna(approx_odds_est)
        else:
            horses["place_odds_est"] = approx_odds_est
        horses["ev_place"] = horses["ensemble_place_prob"] * horses["place_odds_est"]
        horses["kelly_bet_place"] = horses.apply(calc_kelly_bet, axis=1)

        # 5. ワイド (2頭とも3着以内) & 馬連 (2頭が1-2着) の組み合わせ集計
        wide_list = []
        umaren_list = []

        horse_nums = horses["horse_num"].values
        horse_names = horses["horse_name"].values
        odds_vals = horses["odds"].values

        for (i, j) in combinations(range(n_horses), 2):
            h1_num, h2_num = horse_nums[i], horse_nums[j]
            h1_name, h2_name = horse_names[i], horse_names[j]

            # 試行ごとに両馬が3着以内か (ワイド)
            both_top3 = np.sum((ranks[:, i] <= 3) & (ranks[:, j] <= 3))
            wide_prob = both_top3 / self.n_simulations

            # 試行ごとに両馬が1着＆2着か (馬連)
            both_top2 = np.sum((ranks[:, i] <= 2) & (ranks[:, j] <= 2))
            umaren_prob = both_top2 / self.n_simulations

            # 推定オッズ（幾何平均ベースの近似値）
            est_wide_odds = round((horses.loc[i, "place_odds_est"] * horses.loc[j, "place_odds_est"] * 1.5), 1)
            est_umaren_odds = round(np.sqrt(odds_vals[i] * odds_vals[j]) * 3.5, 1)

            wide_ev = wide_prob * est_wide_odds
            umaren_ev = umaren_prob * est_umaren_odds

            if wide_prob >= 0.08:  # 確率8%以上の組み合わせのみ保持
                wide_list.append({
                    "pair": f"{h1_num}-{h2_num}",
                    "names": f"{h1_name} × {h2_name}",
                    "prob": wide_prob,
                    "est_odds": est_wide_odds,
                    "ev": wide_ev
                })

            if umaren_prob >= 0.05:  # 確率5%以上の組み合わせのみ保持
                umaren_list.append({
                    "pair": f"{h1_num}-{h2_num}",
                    "names": f"{h1_name} × {h2_name}",
                    "prob": umaren_prob,
                    "est_odds": est_umaren_odds,
                    "ev": umaren_ev
                })

        wide_df = pd.DataFrame(wide_list)
        if not wide_df.empty:
            wide_df = wide_df.sort_values("ev", ascending=False).reset_index(drop=True)

        umaren_df = pd.DataFrame(umaren_list)
        if not umaren_df.empty:
            umaren_df = umaren_df.sort_values("ev", ascending=False).reset_index(drop=True)

        return horses, wide_df, umaren_df