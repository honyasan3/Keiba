"""馬券購入・回収率（ROI）バックテストシミュレータ"""
from typing import Any, Dict
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("simulator")


class BettingSimulator:
    """予測スコアに基づいた単勝・複勝の回収率シミュレータ"""

    @staticmethod
    def simulate_single_bet(
        df_test: pd.DataFrame,
        pred_col: str = "pred_prob",
        min_ev_threshold: float = 1.2,
        min_prob_threshold: float = 0.10,
        max_rank_in_race: int = 2,
        bet_amount: int = 100,
    ) -> Dict[str, Any]:
        """【厳格化版 単勝シミュレーション】"""
        df = df_test.copy()
        df["expected_value"] = df[pred_col] * df["odds"].fillna(0.0)
        df["pred_rank_in_race"] = df.groupby("race_id")[pred_col].rank(ascending=False, method="min").astype(int)

        bet_condition = (
            (df["expected_value"] >= min_ev_threshold)
            & (df[pred_col] >= min_prob_threshold)
            & (df["pred_rank_in_race"] <= max_rank_in_race)
        )
        bet_targets = df[bet_condition].copy()
        total_bets = len(bet_targets)

        if total_bets == 0:
            logger.warning("単勝条件を満たす購入対象馬がいませんでした。")
            return {"total_bets": 0, "hit_count": 0, "hit_rate": 0.0, "total_invest": 0, "total_return": 0, "recovery_rate": 0.0}

        hit_targets = bet_targets[bet_targets["rank"] == 1]
        hit_count = len(hit_targets)
        total_invest = total_bets * bet_amount
        total_return = int((hit_targets["odds"] * bet_amount).sum())
        recovery_rate = round((total_return / total_invest) * 100, 2)
        hit_rate = round((hit_count / total_bets) * 100, 2)

        logger.info(
            f"【単勝シミュレーション】 購入数: {total_bets}件, 的中: {hit_count}件 (的中率: {hit_rate}%), "
            f"投資: {total_invest}円, 回収: {total_return}円, 回収率: {recovery_rate}%"
        )
        return {"total_bets": total_bets, "hit_count": hit_count, "hit_rate": hit_rate, "total_invest": total_invest, "total_return": total_return, "recovery_rate": recovery_rate}

    @staticmethod
    def simulate_place_bet(
        df_test: pd.DataFrame,
        pred_col: str = "pred_prob",
        min_ev_threshold: float = 1.1,
        min_prob_threshold: float = 0.35,
        max_rank_in_race: int = 3,
        bet_amount: int = 100,
    ) -> Dict[str, Any]:
        """
        【複勝シミュレーション (3着以内)】
        ※ 簡易複勝オッズ推定: 単勝オッズから複勝下限オッズを推計 (odds ** 0.45 程度)
        """
        df = df_test.copy()
        # 複勝オッズの下限概算値 (例: 単勝10倍 → 複勝約2.8倍)
        df["place_odds_est"] = (df["odds"].fillna(1.0) ** 0.45).clip(lower=1.1)
        df["expected_value"] = df[pred_col] * df["place_odds_est"]
        df["pred_rank_in_race"] = df.groupby("race_id")[pred_col].rank(ascending=False, method="min").astype(int)

        bet_condition = (
            (df["expected_value"] >= min_ev_threshold)
            & (df[pred_col] >= min_prob_threshold)
            & (df["pred_rank_in_race"] <= max_rank_in_race)
        )
        bet_targets = df[bet_condition].copy()
        total_bets = len(bet_targets)

        if total_bets == 0:
            logger.warning("複勝条件を満たす購入対象馬がいませんでした。")
            return {"total_bets": 0, "hit_count": 0, "hit_rate": 0.0, "total_invest": 0, "total_return": 0, "recovery_rate": 0.0}

        # 的中判定: 1〜3着
        hit_targets = bet_targets[bet_targets["rank"].between(1, 3)]
        hit_count = len(hit_targets)
        total_invest = total_bets * bet_amount
        total_return = int((hit_targets["place_odds_est"] * bet_amount).sum())
        recovery_rate = round((total_return / total_invest) * 100, 2)
        hit_rate = round((hit_count / total_bets) * 100, 2)

        logger.info(
            f"【複勝シミュレーション】 購入数: {total_bets}件, 的中: {hit_count}件 (的中率: {hit_rate}%), "
            f"投資: {total_invest}円, 回収: {total_return}円, 回収率: {recovery_rate}%"
        )
        return {"total_bets": total_bets, "hit_count": hit_count, "hit_rate": hit_rate, "total_invest": total_invest, "total_return": total_return, "recovery_rate": recovery_rate}