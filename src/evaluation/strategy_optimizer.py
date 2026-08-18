"""ベッティング戦略最適化（グリッドサーチ）モジュール"""
from itertools import product
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("strategy_optimizer")


class BettingStrategyOptimizer:
    """予測確率・オッズ・EVの組み合わせから最高回収率ルールを探索するクラス"""

    @staticmethod
    def optimize_place_strategy(
        df: pd.DataFrame, min_bets: int = 50
    ) -> pd.DataFrame:
        """複勝ベッティングの最適パラメータを探索"""
        ev_thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        min_probs = [0.25, 0.30, 0.35, 0.40, 0.45]
        max_ranks = [1, 2, 3, 5]
        min_odds_list = [1.5, 3.0, 5.0]

        records = []
        df_eval = df.dropna(subset=["rank", "odds", "pred_place_prob"]).copy()
        df_eval["place_odds_est"] = (df_eval["odds"] ** 0.45).clip(lower=1.1)
        df_eval["ev_place"] = df_eval["pred_place_prob"] * df_eval["place_odds_est"]
        df_eval["is_hit"] = df_eval["rank"] <= 3

        for ev_th, min_p, max_r, min_o in product(ev_thresholds, min_probs, max_ranks, min_odds_list):
            cond = (
                (df_eval["ev_place"] >= ev_th)
                & (df_eval["pred_place_prob"] >= min_p)
                & (df_eval["pred_rank"] <= max_r)
                & (df_eval["odds"] >= min_o)
            )
            filtered = df_eval[cond]
            count = len(filtered)

            if count < min_bets:
                continue

            hits = filtered["is_hit"].sum()
            hit_rate = hits / count if count > 0 else 0
            # 概算回収（的中時は推定複勝オッズで配当計算）
            returns = filtered[filtered["is_hit"]]["place_odds_est"].sum() * 100
            investment = count * 100
            roi = (returns / investment) * 100 if investment > 0 else 0

            records.append({
                "ev_threshold": ev_th,
                "min_prob": min_p,
                "max_rank": max_r,
                "min_odds": min_o,
                "bet_count": count,
                "hit_count": hits,
                "hit_rate": round(hit_rate * 100, 2),
                "roi": round(roi, 2),
            })

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("roi", ascending=False).reset_index(drop=True)
        return result_df

    @staticmethod
    def optimize_win_strategy(
        df: pd.DataFrame, min_bets: int = 50
    ) -> pd.DataFrame:
        """単勝ベッティングの最適パラメータを探索"""
        ev_thresholds = [1.0, 1.2, 1.4, 1.6, 1.8]
        min_probs = [0.20, 0.25, 0.30, 0.35, 0.40]
        max_ranks = [1, 2, 3]
        odds_ranges = [(1.5, 999.0), (3.0, 20.0), (5.0, 30.0), (10.0, 999.0)]

        records = []
        df_eval = df.dropna(subset=["rank", "odds", "pred_place_prob"]).copy()
        # 単勝勝率概算（複勝率からの推定値）
        df_eval["pred_win_prob"] = (df_eval["pred_place_prob"] ** 1.6).clip(upper=0.9)
        df_eval["ev_win"] = df_eval["pred_win_prob"] * df_eval["odds"]
        df_eval["is_hit"] = df_eval["rank"] == 1

        for ev_th, min_p, max_r, (min_o, max_o) in product(ev_thresholds, min_probs, max_ranks, odds_ranges):
            cond = (
                (df_eval["ev_win"] >= ev_th)
                & (df_eval["pred_place_prob"] >= min_p)
                & (df_eval["pred_rank"] <= max_r)
                & (df_eval["odds"] >= min_o)
                & (df_eval["odds"] <= max_o)
            )
            filtered = df_eval[cond]
            count = len(filtered)

            if count < min_bets:
                continue

            hits = filtered["is_hit"].sum()
            hit_rate = hits / count if count > 0 else 0
            returns = filtered[filtered["is_hit"]]["odds"].sum() * 100
            investment = count * 100
            roi = (returns / investment) * 100 if investment > 0 else 0

            records.append({
                "ev_threshold": ev_th,
                "min_prob": min_p,
                "max_rank": max_r,
                "odds_range": f"{min_o}〜{max_o}倍",
                "bet_count": count,
                "hit_count": hits,
                "hit_rate": round(hit_rate * 100, 2),
                "roi": round(roi, 2),
            })

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("roi", ascending=False).reset_index(drop=True)
        return result_df