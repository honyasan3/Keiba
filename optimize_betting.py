"""ベッティング戦略最適化 実行スクリプト"""
import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.strategy_optimizer import BettingStrategyOptimizer
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.models.lgbm_model import LGBMRacePredictor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("optimize_betting")


def run_optimization():
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    logger.info("DBから学習・検証用データを読み込み中...")
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
        df = pd.DataFrame(query.all())

    # 特徴量生成
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3)
    df = race_fe.transform(df)
    df = horse_fe.transform(df)

    # 日付でソートして時系列分割点を自動計算 (Train: 70%, Val: 15%, Test: 15%)
    unique_dates = sorted(df["race_date"].dropna().unique())
    n_dates = len(unique_dates)
    train_end = unique_dates[int(n_dates * 0.70)]
    val_end = unique_dates[int(n_dates * 0.85)]

    logger.info(f"評価データ期間: {val_end} 以降のテストデータを対象に検証します")
    _, _, test_df = TimeSeriesDataSplitter.split_by_date(
        df, train_end=train_end, val_end=val_end
    )

    feature_cols = [
            "venue_code", "race_round", "distance", "course_type_cat", "weather_cat",
            "track_condition_cat", "bracket_num", "horse_num", "gender_cat", "age",
            "jockey_weight", "jockey_weight_diff_from_race_mean", "race_horse_count",
            "horse_weight", "horse_weight_diff", "horse_weight_diff_rate",
            "horse_past_runs", "horse_past_avg_rank", "horse_past_win_rate", "horse_past_place_rate",
            "horse_avg_passage_rate", "distance_diff", "horse_recent3_avg_rank",
            "horse_recent3_avg_last3f", "horse_recent3_avg_speed_index", "days_since_prev_race",
            "jockey_past_win_rate", "jockey_past_place_rate", "jockey_venue_place_rate",
            "course_bracket_place_rate", "race_front_runner_count"
        ]

    predictor = LGBMRacePredictor()
    predictor.load("models_saved/lgbm_model.txt")

    test_df["pred_place_prob"] = predictor.predict_proba(test_df[feature_cols])
    test_df["pred_rank"] = test_df.groupby("race_id")["pred_place_prob"].rank(ascending=False, method="min")

    logger.info("=== 複勝ベッティングルールの最適化中 ===")
    place_opt = BettingStrategyOptimizer.optimize_place_strategy(test_df, min_bets=30)
    print("\n【複勝 最適戦略 Top 5】")
    if not place_opt.empty:
        print(tabulate(place_opt.head(5), headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print("基準を満たす戦略が見つかりませんでした。")

    logger.info("=== 単勝ベッティングルールの最適化中 ===")
    win_opt = BettingStrategyOptimizer.optimize_win_strategy(test_df, min_bets=30)
    print("\n【単勝 最適戦略 Top 5】")
    if not win_opt.empty:
        print(tabulate(win_opt.head(5), headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print("基準を満たす戦略が見つかりませんでした。")


if __name__ == "__main__":
    run_optimization()