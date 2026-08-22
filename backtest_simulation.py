"""長期バックテストシミュレーション（全46特徴量: トラックバイアス & Elo & トリプルアンサンブル & ケリー基準資金管理）"""
import os
from typing import List
import numpy as np
import pandas as pd

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.metrics import MetricsEvaluator
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.models.ranker_model import LGBMRankPredictor
from src.pipeline.repository import RaceModel, RaceResultModel
from src.simulation.race_simulator import MonteCarloRaceSimulator

logger = setup_logger("backtest_simulation")


def load_dataset_from_db(connector: DatabaseConnector) -> pd.DataFrame:
    with connector.get_session() as session:
        query = (
            session.query(
                RaceModel.race_id,
                RaceModel.race_title,
                RaceModel.race_date,
                RaceModel.race_round,
                RaceModel.course_type,
                RaceModel.distance,
                RaceModel.weather,
                RaceModel.track_condition,
                RaceResultModel.rank,
                RaceResultModel.bracket_num,
                RaceResultModel.horse_num,
                RaceResultModel.horse_name,
                RaceResultModel.horse_id,
                RaceResultModel.gender,
                RaceResultModel.age,
                RaceResultModel.jockey_weight,
                RaceResultModel.jockey_name,
                RaceResultModel.finish_time_sec,
                RaceResultModel.margin,
                RaceResultModel.passage_order,
                RaceResultModel.last_3f_time,
                RaceResultModel.odds,
                RaceResultModel.popularity,
                RaceResultModel.horse_weight,
                RaceResultModel.horse_weight_diff,
            )
            .join(RaceResultModel, RaceModel.race_id == RaceResultModel.race_id)
        )
        results = query.all()
        return pd.DataFrame(results)


def run_backtest():
    logger.info("=== 長期バックテストシミュレーションを開始します ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    connector = DatabaseConnector(config.db.connection_string)

    df_raw = load_dataset_from_db(connector)
    logger.info(f"データ取得完了: 合計 {len(df_raw)} レコード")

    # 1. 特徴量エンジニアリング（展開・Elo・当日トラックバイアス）
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3, elo_k_factor=16.0)
    bias_fe = TrackBiasFeatureExtractor()

    df_featured = race_fe.transform(df_raw)
    df_featured = horse_fe.transform(df_featured)
    df_featured = bias_fe.transform(df_featured)

    # 2. 目的変数の設定
    df_featured["target_place"] = df_featured["rank"].apply(
        lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0
    )
    df_featured["target_win"] = df_featured["rank"].apply(
        lambda r: 1 if pd.notnull(r) and r == 1 else 0
    )

    # 全46特徴量リスト
    feature_cols = [
        "venue_code", "race_round", "distance", "course_type_cat", "weather_cat",
        "track_condition_cat", "bracket_num", "horse_num", "gender_cat", "age", "age_gender_cat",
        "jockey_weight", "jockey_weight_diff_from_race_mean", "race_horse_count",
        "horse_weight", "horse_weight_diff", "horse_weight_diff_rate",
        "horse_past_runs", "horse_past_avg_rank", "horse_past_win_rate", "horse_past_place_rate",
        "horse_avg_passage_rate", "distance_diff", "distance_shock_cat", "horse_recent3_avg_rank",
        "horse_recent3_avg_last3f", "horse_recent3_avg_speed_index", "days_since_prev_race",
        "rest_category_cat", "is_second_run_after_rest", "is_jockey_changed",
        "jockey_past_win_rate", "jockey_past_place_rate", "jockey_venue_place_rate",
        "course_bracket_place_rate", "race_front_runner_count",
        # 展開負荷・ラップペース特徴量
        "horse_recent3_avg_pci", "prev_pace_disadvantage_front", "prev_pace_disadvantage_back",
        "race_expected_pace_cat", "pace_match_score",
        # Eloレーティング特徴量
        "horse_elo_rating", "race_elo_diff_from_mean",
        # 当日トラックバイアス特徴量
        "bias_inner_bracket_advantage", "bias_front_runner_advantage", "bias_horse_match_score"
    ]

    # 3. 時系列分割（テスト期間の抽出）
    unique_dates = sorted(df_featured["race_date"].unique())
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)

    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    splitter = TimeSeriesDataSplitter()
    train_df, val_df, test_df = splitter.split_by_date(
        df_featured, train_end=train_end, val_end=val_end
    )
    test_df = test_df.copy().reset_index(drop=True)

    # 4. 保存済みモデルのロード
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.load("models_saved/lgbm_model.txt")
    lgbm_preds = lgbm_predictor.predict_proba(test_df[feature_cols])

    cb_predictor = CatBoostRacePredictor()
    cb_predictor.load("models_saved/catboost_model.cbm")
    cb_preds = cb_predictor.predict_proba(test_df[feature_cols])

    rank_predictor = LGBMRankPredictor()
    rank_predictor.load("models_saved/lambdarank_model.txt")
    rank_scores = rank_predictor.predict_score(test_df)
    test_df["_rank_score"] = rank_scores
    rank_norm_scores = test_df.groupby("race_id")["_rank_score"].rank(pct=True).values

    # 5. トリプルアンサンブル予測
    ensemble_preds = (lgbm_preds * 0.40) + (cb_preds * 0.40) + (rank_norm_scores * 0.20)
    test_df["ensemble_prob"] = ensemble_preds

    # 6. ケリー基準・期待値シミュレーション
    # 複勝とワイドは独立した戦略として、それぞれ別のシミュレーション軍資金で評価する
    # （合算すると片方の勝敗がもう片方の資金効率に影響してしまい、券種ごとの実力が見えなくなるため）。
    initial_bankroll = 100000.0
    stats = {
        "place": {"bankroll": initial_bankroll, "history": [initial_bankroll], "bet_count": 0, "hit_count": 0, "invest": 0.0, "return": 0.0},
        "wide": {"bet_count": 0, "hit_count": 0, "invest": 0.0, "return": 0.0},
    }

    RACE_MAX_EXPOSURE = 0.05  # 1レースあたりの合計ベット比率の上限（bankroll比、複勝のケリー配分にのみ使用）
    WIDE_FLAT_STAKE = 100  # ワイドは1点固定ステーク（理由はワイド購入基準フィルタ直前のコメント参照）

    def _size_race_bets(bankroll: float, candidates: list) -> List[int]:
        """レース単位でケリー基準のベット額を算出する。

        candidatesは(p, odds_est)のリスト。個々の候補を独立に最大5%ずつ賭けると、
        特にワイドで見られるように「同じ馬を含む複数ペアが同時に条件を満たし、しかもそれらは
        結果が強く連動する（同じ馬が絡むため一緒に的中/落選しやすい）」場合に、実質的な
        1レースあたりのエクスポージャーが際限なく積み上がってしまう。そのため個々のケリー比率を
        求めたうえで、レース合計がRACE_MAX_EXPOSUREを超える場合は全体を比例縮小する。
        """
        raw_ratios = []
        for p, odds_est in candidates:
            b = odds_est - 1.0
            q = 1.0 - p
            kelly_f = max(0.0, (b * p - q) / b) if b > 0 else 0.0
            raw_ratios.append(min(0.05, kelly_f * 0.10))

        total_ratio = sum(raw_ratios)
        scale = (RACE_MAX_EXPOSURE / total_ratio) if total_ratio > RACE_MAX_EXPOSURE else 1.0
        return [int(bankroll * ratio * scale // 100) * 100 for ratio in raw_ratios]

    simulator = MonteCarloRaceSimulator(n_simulations=10000)
    test_races = test_df.groupby("race_id", sort=False)
    logger.info(f"バックテスト対象レース数: {len(test_races)} レース (期間: {test_df['race_date'].min()} ~ {test_df['race_date'].max()})")

    for race_id, group in test_races:
        group = group.copy()
        # レース内順位と推定複勝オッズ
        group["pred_rank"] = group["ensemble_prob"].rank(ascending=False)
        # 簡易複勝オッズ推定: race_simulator.py / strategy_optimizer.py と同じ近似式に統一
        # （以前は odds/3.5 という別式を使っており、EVしきい値がstrategy_optimizer.py側の
        # グリッドサーチ結果と整合しなくなっていた）
        group["est_place_odds"] = (group["odds"] ** 0.45).clip(lower=1.1)
        group["ev_place"] = group["ensemble_prob"] * group["est_place_odds"]

        # 購入基準フィルタ: rolling_walk_forward.py による3fold独立検証（各foldでモデルを再学習し
        # Validation期間のみで選定→一度も選定に使っていないTest期間に適用）で一貫して選ばれた
        # 複勝ルール（EV >= 1.4, 複勝率 >= 0.45, 予測3位以内, 単勝オッズ3.0〜5.0倍）。
        # 3fold中2foldが完全一致でこの単勝オッズ3〜5倍帯を独立選定しており（残り1foldも2〜6倍で近い）、
        # 3fold合算326件でのプールROIは110.34%と、単一テスト期間の循環検証だった旧版とは異なり
        # 複数の独立した期間で再現性が確認できている（単勝は同様の検証で優位性が確認できず対象外とした）。
        place_candidates = group[
            (group["ev_place"] >= 1.4)
            & (group["ensemble_prob"] >= 0.45)
            & (group["pred_rank"] <= 3)
            & (group["odds"] >= 3.0)
            & (group["odds"] <= 5.0)
        ]

        # 同一レース内の複数ベットは実際には同時に行うため、ベット額は「そのレース開始時点の
        # bankroll」を基準に、レース単位の合計エクスポージャー上限内で算出する
        # （レース結果が出る前に全ベット額を決めるのと同じ）。bankroll自体はベット確定のたびに
        # 加算・減算し、次のレースには正しく反映する。
        s = stats["place"]
        race_start_bankroll_place = s["bankroll"]
        place_rows = list(place_candidates.itertuples(index=False))
        place_amounts = _size_race_bets(
            race_start_bankroll_place, [(r.ensemble_prob, r.est_place_odds) for r in place_rows]
        )
        for row, bet_amount in zip(place_rows, place_amounts):
            if bet_amount < 100:
                continue
            s["bet_count"] += 1
            s["invest"] += bet_amount
            s["bankroll"] -= bet_amount
            if pd.notnull(row.rank) and 1 <= row.rank <= 3:
                s["hit_count"] += 1
                return_amount = bet_amount * row.est_place_odds
                s["return"] += return_amount
                s["bankroll"] += return_amount
            s["history"].append(s["bankroll"])

        # ワイド購入基準: rolling_walk_forward.pyの3fold独立検証で3fold中2foldが完全一致で選定した
        # ルール（EV >= 1.0, 的中確率 >= 0.3, 想定オッズ3.0〜10.0倍）。3fold合算15,078件でのプール
        # ROIは120.00%と、この一連の検証の中で最も件数が多く安定した結果だった。
        race_df_for_sim = group.dropna(subset=["rank"]).copy()
        if len(race_df_for_sim) >= 2:
            try:
                _, wide_df, _ = simulator.simulate_race(race_df_for_sim)
            except Exception as e:
                logger.warning(f"ワイドシミュレーションに失敗しました (Race ID: {race_id}): {e}")
                wide_df = pd.DataFrame()

            if not wide_df.empty:
                rank_by_num = dict(zip(race_df_for_sim["horse_num"], race_df_for_sim["rank"]))
                wide_candidates = wide_df[
                    (wide_df["ev"] >= 1.0) & (wide_df["prob"] >= 0.3)
                    & (wide_df["est_odds"] >= 3.0) & (wide_df["est_odds"] <= 10.0)
                ]
                s = stats["wide"]
                # ワイドは1レースあたりの該当件数が複勝よりずっと多く（平均2件以上/レース、かつ
                # 同じ馬を含むペア同士は結果が強く連動する）、bankroll比例のKelly複利をそのまま
                # 適用すると、たった8ヶ月のTest期間でも理論上は数百億円規模まで際限なく複利成長して
                # しまい、現実の馬券市場の資金吸収力（大口になるほどオッズが不利に動く）を無視した
                # 非現実的な数字になる。複勝と同じ枠組みでは意味のある金額を出せないため、ワイドは
                # optimize_betting.py / rolling_walk_forward.py と同じ「1点100円のフラットステーク」
                # で集計する（bankroll/ドローダウンの概念は使わない）。
                for row in wide_candidates.itertuples(index=False):
                    s["bet_count"] += 1
                    s["invest"] += WIDE_FLAT_STAKE
                    try:
                        h1, h2 = (int(float(x)) for x in str(row.pair).split("-"))
                        is_hit = rank_by_num.get(h1, 999) <= 3 and rank_by_num.get(h2, 999) <= 3
                    except (ValueError, AttributeError):
                        is_hit = False
                    if is_hit:
                        s["hit_count"] += 1
                        s["return"] += WIDE_FLAT_STAKE * row.est_odds

    # 7. パフォーマンス指標の集計・表示
    logger.info("=" * 60)
    logger.info("★ 【長期バックテスト結果サマリー】 ★")
    logger.info(f"・対象レース数: {len(test_races)} レース")

    # 複勝: ケリー資金配分でのbankroll推移・ドローダウンまで含めて表示
    s = stats["place"]
    net_profit = s["return"] - s["invest"]
    recovery_rate = (s["return"] / s["invest"] * 100) if s["invest"] > 0 else 0.0
    hit_rate = (s["hit_count"] / s["bet_count"] * 100) if s["bet_count"] > 0 else 0.0
    peaks = np.maximum.accumulate(s["history"])
    drawdowns = (peaks - s["history"]) / peaks * 100
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    logger.info("--- 複勝（ケリー資金配分） ---")
    logger.info(f"・総購入件数: {s['bet_count']} 件 (的中: {s['hit_count']} 件, 的中率: {hit_rate:.2f}%)")
    logger.info(f"・総投資額: {s['invest']:,.0f} 円")
    logger.info(f"・総払戻額: {s['return']:,.0f} 円")
    logger.info(f"・純利益: {net_profit:+,.0f} 円")
    logger.info(f"・回収率: {recovery_rate:.2f} %")
    logger.info(f"・最終資金: {s['bankroll']:,.0f} 円 (初期資金: {initial_bankroll:,.0f} 円)")
    logger.info(f"・最大ドローダウン: {max_drawdown:.2f} %")

    # ワイド: フラットステーク集計（理由は購入基準フィルタ直前のコメント参照）。
    # bankroll複利・ドローダウンは意味を持たないため表示しない。
    s = stats["wide"]
    net_profit = s["return"] - s["invest"]
    recovery_rate = (s["return"] / s["invest"] * 100) if s["invest"] > 0 else 0.0
    hit_rate = (s["hit_count"] / s["bet_count"] * 100) if s["bet_count"] > 0 else 0.0

    logger.info(f"--- ワイド（1点{WIDE_FLAT_STAKE}円のフラットステーク、Kelly複利は非現実的なため不使用） ---")
    logger.info(f"・総購入件数: {s['bet_count']} 件 (的中: {s['hit_count']} 件, 的中率: {hit_rate:.2f}%)")
    logger.info(f"・総投資額: {s['invest']:,.0f} 円")
    logger.info(f"・総払戻額: {s['return']:,.0f} 円")
    logger.info(f"・純利益: {net_profit:+,.0f} 円")
    logger.info(f"・回収率: {recovery_rate:.2f} %")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_backtest()