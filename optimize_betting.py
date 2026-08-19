"""ベッティング戦略最適化スクリプト（全46特徴量: トラックバイアス & Elo & トリプルアンサンブル統合版）"""
import os
import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.strategy_optimizer import BettingStrategyOptimizer
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.models.ranker_model import LGBMRankPredictor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("optimize_betting")


def get_all_race_data(db_connector: DatabaseConnector) -> pd.DataFrame:
    with db_connector.get_session() as session:
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
        return pd.DataFrame(query.all())


def run_optimization() -> None:
    logger.info("=== 買い目戦略最適化グリッドサーチ（トリプルアンサンブル & 46特徴量版）を開始します ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    raw_df = get_all_race_data(db_connector)
    if raw_df.empty:
        logger.error("データベースからデータを取得できませんでした。")
        return

    # 1. 特徴量生成（ピュア能力・Eloレーティング・当日トラックバイアス）
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3, elo_k_factor=16.0)
    bias_fe = TrackBiasFeatureExtractor()

    featured_df = race_fe.transform(raw_df)
    featured_df = horse_fe.transform(featured_df)
    featured_df = bias_fe.transform(featured_df)

    # 2. 時系列分割 (Train: 70% / Val: 15% / Test: 15%)
    unique_dates = sorted(featured_df["race_date"].unique())
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)

    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    splitter = TimeSeriesDataSplitter()
    _, _, test_df = splitter.split_by_date(
        featured_df, train_end=train_end, val_end=val_end
    )
    test_df = test_df.copy().reset_index(drop=True)

    logger.info(f"評価データ期間: {test_df['race_date'].min()} ~ {test_df['race_date'].max()} のテストデータを対象に検証します")

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

    # 3. トリプルアンサンブルモデル読み込みと推論 (LGBM 40% + CatBoost 40% + Ranker 20%)
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.load("models_saved/lgbm_model.txt")
    lgbm_probs = lgbm_predictor.predict_proba(test_df[feature_cols])

    if os.path.exists("models_saved/catboost_model.cbm"):
        cb_predictor = CatBoostRacePredictor()
        cb_predictor.load("models_saved/catboost_model.cbm")
        cb_probs = cb_predictor.predict_proba(test_df[feature_cols])
    else:
        cb_probs = lgbm_probs

    if os.path.exists("models_saved/lambdarank_model.txt"):
        rank_predictor = LGBMRankPredictor()
        rank_predictor.load("models_saved/lambdarank_model.txt")
        rank_scores = rank_predictor.predict_score(test_df)
        test_df["_rank_score"] = rank_scores
        rank_norm_scores = test_df.groupby("race_id")["_rank_score"].rank(pct=True).values
    else:
        rank_norm_scores = lgbm_probs

    test_df["pred_place_prob"] = (lgbm_probs * 0.40) + (cb_probs * 0.40) + (rank_norm_scores * 0.20)

    # レースごとの予測順位 (pred_rank) を算出
    test_df["pred_rank"] = (
        test_df.groupby("race_id")["pred_place_prob"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 4. グリッドサーチ最適化
    optimizer = BettingStrategyOptimizer()

    logger.info("=== 複勝ベッティングルールの最適化中 ===")
    place_results = optimizer.optimize_place_strategy(test_df)
    print("\n【複勝 最適戦略 Top 5】")
    if not place_results.empty:
        print(tabulate(place_results.head(5), headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")

    logger.info("=== 単勝ベッティングルールの最適化中 ===")
    win_results = optimizer.optimize_win_strategy(test_df)
    print("\n【単勝 最適戦略 Top 5】")
    if not win_results.empty:
        print(tabulate(win_results.head(5), headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")


if __name__ == "__main__":
    run_optimization()