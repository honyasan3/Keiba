"""過去テストデータを用いた長期運用シミュレーションスクリプト（アンサンブル × ケリー資金管理版）"""
import numpy as np
import pandas as pd
from tabulate import tabulate
from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("backtest_simulation")


def load_dataset(db_connector: DatabaseConnector) -> pd.DataFrame:
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


def run_backtest() -> None:
    logger.info("=== 長期運用バックテストを開始します ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    connector = DatabaseConnector(config.db.connection_string)

    df_raw = load_dataset(connector)
    logger.info(f"データ取得完了: 合計 {len(df_raw)} 件")

    # 1. 特徴量生成
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3)

    df_featured = race_fe.transform(df_raw)
    df_featured = horse_fe.transform(df_featured)

    # 2. 時系列データ分割（テストセット抽出）
    splitter = TimeSeriesDataSplitter()
    _, _, test_df = splitter.split_by_date(
        df_featured, train_end="2025-11-08", val_end="2026-03-22"
    )

    logger.info(f"バックテスト対象期間: {test_df['race_date'].min()} 〜 {test_df['race_date'].max()} ({len(test_df)} 件)")

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
        "course_bracket_place_rate", "race_front_runner_count"
    ]

    # 3. アンサンブル推論
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.load("models_saved/lgbm_model.txt")
    lgbm_preds = lgbm_predictor.predict_proba(test_df[feature_cols])

    cb_predictor = CatBoostRacePredictor()
    cb_predictor.load("models_saved/catboost_model.cbm")
    cb_preds = cb_predictor.predict_proba(test_df[feature_cols])

    test_df = test_df.copy()
    test_df["pred_place_prob"] = (lgbm_preds * 0.5) + (cb_preds * 0.5)

    # 推定複勝オッズおよびEV算出
    test_df["est_place_odds"] = np.clip(1.1 + (test_df["odds"] - 1.0) * 0.28, 1.1, 15.0)
    test_df["ev_place"] = test_df["pred_place_prob"] * test_df["est_place_odds"]
    test_df["pred_rank"] = (
        test_df.groupby("race_id")["pred_place_prob"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # 4. 戦略ルール抽出（複勝・単勝）
    # 複勝: EV >= 1.0, 確率 >= 45%, 予測3位以内, オッズ >= 5.0倍
    place_bets = test_df[
        (test_df["ev_place"] >= 1.0)
        & (test_df["pred_place_prob"] >= 0.45)
        & (test_df["pred_rank"] <= 3)
        & (test_df["odds"] >= 5.0)
    ].copy()

    # ケリー基準による賭け金（100円〜1000円）
    b = np.maximum(place_bets["est_place_odds"] - 1.0, 0.01)
    p = place_bets["pred_place_prob"]
    q = 1.0 - p
    f_star = np.clip((b * p - q) / b, 0.0, 1.0)
    fractional_kelly = f_star * 0.15
    place_bets["bet_amount"] = (np.clip(fractional_kelly * 10000, 100, 1000) // 100 * 100).astype(int)

    # 的中判定（3着以内）
    place_bets["is_hit"] = place_bets["rank"].apply(lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0)
    place_bets["payout"] = np.where(
        place_bets["is_hit"] == 1,
        place_bets["bet_amount"] * place_bets["est_place_odds"],
        0.0
    ).astype(int)

    # 日付・月別集計
    place_bets["month"] = pd.to_datetime(place_bets["race_date"]).dt.strftime("%Y-%m")

    monthly_summary = place_bets.groupby("month").agg(
        bet_count=("is_hit", "count"),
        hit_count=("is_hit", "sum"),
        total_invest=("bet_amount", "sum"),
        total_return=("payout", "sum"),
    ).reset_index()

    monthly_summary["hit_rate"] = (monthly_summary["hit_count"] / monthly_summary["bet_count"] * 100).round(1).astype(str) + "%"
    monthly_summary["profit"] = monthly_summary["total_return"] - monthly_summary["total_invest"]
    monthly_summary["roi"] = (monthly_summary["total_return"] / monthly_summary["total_invest"] * 100).round(2).astype(str) + "%"

    # 全体サマリー
    total_invest = place_bets["bet_amount"].sum()
    total_return = place_bets["payout"].sum()
    total_profit = total_return - total_invest
    total_roi = (total_return / total_invest * 100) if total_invest > 0 else 0.0
    hit_rate = (place_bets["is_hit"].sum() / len(place_bets) * 100) if len(place_bets) > 0 else 0.0

    print("\n" + "=" * 70)
    print(" 📈 【テスト期間（2026年3月〜8月）複勝アンサンブル・バックテスト結果】")
    print("=" * 70)
    print(tabulate(monthly_summary, headers=["月度", "購入数", "的中数", "投資額", "払戻額", "的中率", "収支", "回収率(ROI)"], tablefmt="fancy_grid", showindex=False))

    print("\n" + "-" * 50)
    print(f" 🎯 総購入件数: {len(place_bets)} レース")
    print(f" 🎯 的中数: {place_bets['is_hit'].sum()} 件 (的中率: {hit_rate:.1f}%)")
    print(f" 💰 総投資額: {total_invest:,} 円")
    print(f" 💰 総払戻額: {total_return:,} 円")
    print(f" 📊 純利益: {'＋' if total_profit >= 0 else ''}{total_profit:,} 円")
    print(f" 🚀 最終回収率 (ROI): 【{total_roi:.2f}%】")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    run_backtest()