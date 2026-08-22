"""ベッティング戦略最適化（グリッドサーチ）モジュール"""
from itertools import product
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.common.logger import setup_logger
from src.simulation.race_simulator import MonteCarloRaceSimulator

logger = setup_logger("strategy_optimizer")


class BettingStrategyOptimizer:
    """予測確率・オッズ・EVの組み合わせから最高回収率ルールを探索するクラス"""

    @staticmethod
    def prep_place_eval_df(df: pd.DataFrame) -> pd.DataFrame:
        """複勝戦略評価用に、EV・的中フラグ等を付与した評価用DataFrameを作成"""
        df_eval = df.dropna(subset=["rank", "odds", "pred_place_prob"]).copy()
        df_eval["place_odds_est"] = (df_eval["odds"] ** 0.45).clip(lower=1.1)
        df_eval["ev_place"] = df_eval["pred_place_prob"] * df_eval["place_odds_est"]
        df_eval["is_hit"] = df_eval["rank"] <= 3
        return df_eval

    @staticmethod
    def evaluate_place_strategy(
        df_eval: pd.DataFrame, ev_th: float, min_p: float, max_r: int, odds_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """`prep_place_eval_df`済みのDataFrameに対し、指定した1つの複勝ルールを適用して指標を算出"""
        min_o, max_o = odds_range
        cond = (
            (df_eval["ev_place"] >= ev_th)
            & (df_eval["pred_place_prob"] >= min_p)
            & (df_eval["pred_rank"] <= max_r)
            & (df_eval["odds"] >= min_o)
            & (df_eval["odds"] <= max_o)
        )
        filtered = df_eval[cond]
        count = len(filtered)
        hits = int(filtered["is_hit"].sum())
        hit_rate = hits / count if count > 0 else 0.0
        # 概算回収（的中時は推定複勝オッズで配当計算）
        returns = filtered[filtered["is_hit"]]["place_odds_est"].sum() * 100
        investment = count * 100
        roi = (returns / investment) * 100 if investment > 0 else 0.0

        return {
            "ev_threshold": ev_th,
            "min_prob": min_p,
            "max_rank": max_r,
            "odds_range": f"{min_o}〜{max_o}倍",
            "min_odds": min_o,
            "max_odds": max_o,
            "bet_count": count,
            "hit_count": hits,
            "hit_rate": round(hit_rate * 100, 2),
            "roi": round(roi, 2),
        }

    @staticmethod
    def optimize_place_strategy(
        df: pd.DataFrame, min_bets: int = 50
    ) -> pd.DataFrame:
        """複勝ベッティングの最適パラメータを探索

        odds_rangesには従来の下限のみのオープンレンジに加え、確率較正チェック（evaluate_calibration.py）
        で歪みが最も小さいと分かったオッズ帯（3〜5倍付近）に絞った候補も加えている。
        """
        ev_thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        min_probs = [0.25, 0.30, 0.35, 0.40, 0.45]
        max_ranks = [1, 2, 3, 5]
        odds_ranges = [
            (1.5, 999.0), (3.0, 999.0), (5.0, 999.0),  # 従来の下限のみレンジ
            (2.0, 6.0), (3.0, 5.0),                     # 確率較正の歪みが小さいオッズ帯に絞った候補
        ]

        df_eval = BettingStrategyOptimizer.prep_place_eval_df(df)

        records = []
        for ev_th, min_p, max_r, odds_range in product(ev_thresholds, min_probs, max_ranks, odds_ranges):
            result = BettingStrategyOptimizer.evaluate_place_strategy(df_eval, ev_th, min_p, max_r, odds_range)
            if result["bet_count"] < min_bets:
                continue
            records.append(result)

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("roi", ascending=False).reset_index(drop=True)
        return result_df

    @staticmethod
    def prep_wide_eval_df(df: pd.DataFrame, n_simulations: int = 3000) -> pd.DataFrame:
        """ワイド戦略評価用の候補ペアDataFrameを作成する（place/winと異なりコストが高い処理）。

        複勝・単勝は1頭ごとの`pred_place_prob`だけで判定できるが、ワイドは「2頭とも3着以内」という
        組み合わせ確率が必要で、これは predict.py の本番推論でも `MonteCarloRaceSimulator` による
        1万回シミュレーションで算出している。過去データの検証でも同じ仕組みを使わないと、実際に
        ライブで計算されるワイドEVとは異なる基準を検証してしまうことになるため、レースごとに
        シミュレーションを実行し、実際の着順と突き合わせて的中フラグを付与する。
        レース数が多いと相応に時間がかかるため、既定では有効試行数を本番の1万回より減らしている
        （グリッドサーチ自体はこのメソッドの戻り値に対して行うため、このメソッドはVal/Testそれぞれ
        1回ずつ呼べば十分）。
        """
        simulator = MonteCarloRaceSimulator(n_simulations=n_simulations)
        records = []

        for race_id, group in df.groupby("race_id"):
            race_df = group.dropna(subset=["rank"]).copy()
            if len(race_df) < 2:
                continue
            rank_by_num = dict(zip(race_df["horse_num"], race_df["rank"]))

            try:
                _, wide_df, _ = simulator.simulate_race(race_df)
            except Exception as e:
                logger.warning(f"ワイドシミュレーションに失敗しました (Race ID: {race_id}): {e}")
                continue
            if wide_df.empty:
                continue

            for _, row in wide_df.iterrows():
                try:
                    h1, h2 = (int(float(x)) for x in str(row["pair"]).split("-"))
                except (ValueError, AttributeError):
                    continue
                r1, r2 = rank_by_num.get(h1), rank_by_num.get(h2)
                if r1 is None or r2 is None:
                    continue
                records.append({
                    "race_id": race_id,
                    "pair": row["pair"],
                    "prob": row["prob"],
                    "est_odds": row["est_odds"],
                    "ev": row["ev"],
                    "is_hit": bool(r1 <= 3 and r2 <= 3),
                })

        return pd.DataFrame(records)

    @staticmethod
    def evaluate_wide_strategy(
        df_eval: pd.DataFrame, ev_th: float, min_p: float, odds_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """`prep_wide_eval_df`済みのDataFrameに対し、指定した1つのワイドルールを適用して指標を算出"""
        min_o, max_o = odds_range
        cond = (
            (df_eval["ev"] >= ev_th)
            & (df_eval["prob"] >= min_p)
            & (df_eval["est_odds"] >= min_o)
            & (df_eval["est_odds"] <= max_o)
        )
        filtered = df_eval[cond]
        count = len(filtered)
        hits = int(filtered["is_hit"].sum())
        hit_rate = hits / count if count > 0 else 0.0
        returns = filtered[filtered["is_hit"]]["est_odds"].sum() * 100
        investment = count * 100
        roi = (returns / investment) * 100 if investment > 0 else 0.0

        return {
            "ev_threshold": ev_th,
            "min_prob": min_p,
            "odds_range": f"{min_o}〜{max_o}倍",
            "min_odds": min_o,
            "max_odds": max_o,
            "bet_count": count,
            "hit_count": hits,
            "hit_rate": round(hit_rate * 100, 2),
            "roi": round(roi, 2),
        }

    @staticmethod
    def optimize_wide_strategy(
        df_eval: pd.DataFrame, min_bets: int = 50
    ) -> pd.DataFrame:
        """ワイドベッティングの最適パラメータを探索。

        引数は生データではなく`prep_wide_eval_df`で事前作成した候補ペアDataFrameを渡すこと
        （place/winのoptimize_*と異なり、シミュレーションが必要なためprepを毎回呼ぶと非常に遅くなる）。
        """
        ev_thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
        min_probs = [0.10, 0.15, 0.20, 0.25, 0.30]
        odds_ranges = [(1.0, 999.0), (3.0, 999.0), (5.0, 999.0), (3.0, 10.0), (5.0, 15.0)]

        records = []
        for ev_th, min_p, odds_range in product(ev_thresholds, min_probs, odds_ranges):
            result = BettingStrategyOptimizer.evaluate_wide_strategy(df_eval, ev_th, min_p, odds_range)
            if result["bet_count"] < min_bets:
                continue
            records.append(result)

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("roi", ascending=False).reset_index(drop=True)
        return result_df

    @staticmethod
    def prep_win_eval_df(df: pd.DataFrame) -> pd.DataFrame:
        """単勝戦略評価用に、EV・的中フラグ等を付与した評価用DataFrameを作成"""
        df_eval = df.dropna(subset=["rank", "odds", "pred_place_prob"]).copy()
        # 単勝勝率概算（複勝率からの推定値）
        df_eval["pred_win_prob"] = (df_eval["pred_place_prob"] ** 1.6).clip(upper=0.9)
        df_eval["ev_win"] = df_eval["pred_win_prob"] * df_eval["odds"]
        df_eval["is_hit"] = df_eval["rank"] == 1
        return df_eval

    @staticmethod
    def evaluate_win_strategy(
        df_eval: pd.DataFrame, ev_th: float, min_p: float, max_r: int, odds_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """`prep_win_eval_df`済みのDataFrameに対し、指定した1つの単勝ルールを適用して指標を算出"""
        min_o, max_o = odds_range
        cond = (
            (df_eval["ev_win"] >= ev_th)
            & (df_eval["pred_place_prob"] >= min_p)
            & (df_eval["pred_rank"] <= max_r)
            & (df_eval["odds"] >= min_o)
            & (df_eval["odds"] <= max_o)
        )
        filtered = df_eval[cond]
        count = len(filtered)
        hits = int(filtered["is_hit"].sum())
        hit_rate = hits / count if count > 0 else 0.0
        returns = filtered[filtered["is_hit"]]["odds"].sum() * 100
        investment = count * 100
        roi = (returns / investment) * 100 if investment > 0 else 0.0

        return {
            "ev_threshold": ev_th,
            "min_prob": min_p,
            "max_rank": max_r,
            "odds_range": f"{min_o}〜{max_o}倍",
            "min_odds": min_o,
            "max_odds": max_o,
            "bet_count": count,
            "hit_count": hits,
            "hit_rate": round(hit_rate * 100, 2),
            "roi": round(roi, 2),
        }

    @staticmethod
    def optimize_win_strategy(
        df: pd.DataFrame, min_bets: int = 50
    ) -> pd.DataFrame:
        """単勝ベッティングの最適パラメータを探索

        odds_rangesには確率較正チェックで歪みが最も小さいと分かったオッズ帯（3〜5倍付近）に
        絞った候補(2.0〜6.0, 3.0〜5.0)も加えている。
        """
        ev_thresholds = [1.0, 1.2, 1.4, 1.6, 1.8]
        min_probs = [0.20, 0.25, 0.30, 0.35, 0.40]
        max_ranks = [1, 2, 3]
        odds_ranges = [
            (1.5, 999.0), (3.0, 20.0), (5.0, 30.0), (10.0, 999.0),
            (2.0, 6.0), (3.0, 5.0),
        ]

        df_eval = BettingStrategyOptimizer.prep_win_eval_df(df)

        records = []
        for ev_th, min_p, max_r, odds_range in product(ev_thresholds, min_probs, max_ranks, odds_ranges):
            result = BettingStrategyOptimizer.evaluate_win_strategy(df_eval, ev_th, min_p, max_r, odds_range)
            if result["bet_count"] < min_bets:
                continue
            records.append(result)

        result_df = pd.DataFrame(records)
        if not result_df.empty:
            result_df = result_df.sort_values("roi", ascending=False).reset_index(drop=True)
        return result_df
