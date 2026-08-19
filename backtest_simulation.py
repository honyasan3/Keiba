"""長期バックテストシミュレーション（全46特徴量: トラックバイアス & Elo & トリプルアンサンブル & ケリー基準資金管理）"""
import os
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
    initial_bankroll = 100000.0
    current_bankroll = initial_bankroll
    bankroll_history = [initial_bankroll]

    bet_count = 0
    hit_count = 0
    total_bet_amount = 0.0
    total_return_amount = 0.0

    test_races = test_df.groupby("race_id", sort=False)
    logger.info(f"バックテスト対象レース数: {len(test_races)} レース (期間: {test_df['race_date'].min()} ~ {test_df['race_date'].max()})")

    for race_id, group in test_races:
        group = group.copy()
        # レース内順位と推定複勝オッズ
        group["pred_rank"] = group["ensemble_prob"].rank(ascending=False)
        # 簡易複勝オッズ推定: 単勝オッズから算出（1.1〜単勝/3.5）
        group["est_place_odds"] = np.clip(group["odds"] / 3.5, 1.1, 15.0)
        group["ev_place"] = group["ensemble_prob"] * group["est_place_odds"]

        # 購入基準フィルタ: EV >= 1.15, 複勝率 >= 0.38, 予測3位以内, 単勝オッズ >= 4.5
        candidates = group[
            (group["ev_place"] >= 1.15)
            & (group["ensemble_prob"] >= 0.38)
            & (group["pred_rank"] <= 3)
            & (group["odds"] >= 4.5)
        ]

        for _, row in candidates.iterrows():
            # ケリー基準ベット額計算（安全係数 0.10 のフラクショナルケリー）
            b = row["est_place_odds"] - 1.0
            p = row["ensemble_prob"]
            q = 1.0 - p
            kelly_f = max(0.0, (b * p - q) / b) if b > 0 else 0.0
            bet_ratio = min(0.05, kelly_f * 0.10)  # 最大ベット上限 5%
            bet_amount = int(current_bankroll * bet_ratio // 100) * 100

            if bet_amount < 100:
                continue

            bet_count += 1
            total_bet_amount += bet_amount
            current_bankroll -= bet_amount

            # 着順判定 (1~3着で的中)
            if pd.notnull(row["rank"]) and 1 <= row["rank"] <= 3:
                hit_count += 1
                return_amount = bet_amount * row["est_place_odds"]
                total_return_amount += return_amount
                current_bankroll += return_amount

            bankroll_history.append(current_bankroll)

    # 7. パフォーマンス指標の集計
    net_profit = total_return_amount - total_bet_amount
    recovery_rate = (total_return_amount / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
    hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0.0

    # 最大ドローダウンの算出
    peaks = np.maximum.accumulate(bankroll_history)
    drawdowns = (peaks - bankroll_history) / peaks * 100
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    logger.info("=" * 60)
    logger.info("★ 【長期バックテスト結果サマリー】 ★")
    logger.info(f"・対象レース数: {len(test_races)} レース")
    logger.info(f"・総購入件数: {bet_count} 件 (的中: {hit_count} 件, 的中率: {hit_rate:.2f}%)")
    logger.info(f"・総投資額: {total_bet_amount:,.0f} 円")
    logger.info(f"・総払戻額: {total_return_amount:,.0f} 円")
    logger.info(f"・純利益: {net_profit:+,.0f} 円")
    logger.info(f"・回収率: {recovery_rate:.2f} %")
    logger.info(f"・最終資金: {current_bankroll:,.0f} 円 (初期資金: {initial_bankroll:,.0f} 円)")
    logger.info(f"・最大ドローダウン: {max_drawdown:.2f} %")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_backtest()