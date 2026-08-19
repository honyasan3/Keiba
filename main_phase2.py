"""フェーズ2: 特徴量エンジニアリング、アンサンブルモデル学習 (LightGBM × CatBoost)、バックテスト実行パイプライン"""
import os
import pandas as pd
from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.leak_validator import DataLeakageValidator
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.metrics import MetricsEvaluator
from src.evaluation.simulator import BettingSimulator
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("main_phase2")


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


def run_pipeline() -> None:
    logger.info("=== フェーズ2 モデル学習・検証パイプラインを開始します ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    connector = DatabaseConnector(config.db.connection_string)

    df_raw = load_dataset_from_db(connector)
    logger.info(f"データ取得完了: 合計 {len(df_raw)} レコード")

    # 1. 特徴量エンジニアリング
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3)

    df_featured = race_fe.transform(df_raw)
    df_featured = horse_fe.transform(df_featured)

    # 2. 目的変数の設定 (3着以内 = 1, それ以外 = 0)
    df_featured["target_place"] = df_featured["rank"].apply(
        lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0
    )
    df_featured["target_win"] = df_featured["rank"].apply(
        lambda r: 1 if pd.notnull(r) and r == 1 else 0
    )

    # カテゴリ特徴量 & 全特徴量リスト
    cat_cols = [
        "venue_code", "course_type_cat", "weather_cat", "track_condition_cat",
        "gender_cat", "age_gender_cat", "rest_category_cat", "distance_shock_cat"
    ]
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

    # リーク検証
    if not DataLeakageValidator.validate_features(feature_cols):
        logger.error("リーク検証エラーにより中断します。")
        return

    # 3. 時系列データ分割（70% / 15% / 15% の境界日付を自動算出して分割）
    unique_dates = sorted(df_featured["race_date"].unique())
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)

    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    splitter = TimeSeriesDataSplitter()
    train_df, val_df, test_df = splitter.split_by_date(
        df_featured, train_end=train_end, val_end=val_end
    )

    X_train = train_df[feature_cols]
    y_train = train_df["target_place"]
    X_val = val_df[feature_cols]
    y_val = val_df["target_place"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_place"]

    os.makedirs("models_saved", exist_ok=True)

    # 4. LightGBM学習（複勝モデル target_place を学習）
    logger.info("--- [1/2] LightGBM モデルの学習を開始 ---")
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=30,
    )

    importance_df = lgbm_predictor.get_feature_importance()
    logger.info(f"【LightGBM 特徴量重要度 Top 10】\n{importance_df.head(10)}")
    lgbm_predictor.save("models_saved/lgbm_model.txt")

    lgbm_test_preds = lgbm_predictor.predict_proba(X_test)
    lgbm_metrics = MetricsEvaluator.calculate_all_metrics(y_test, lgbm_test_preds)
    logger.info(f"LightGBM 単体評価 - AUC: {lgbm_metrics['auc']}, LogLoss: {lgbm_metrics['logloss']}")

    # 5. CatBoost学習
    logger.info("--- [2/2] CatBoost モデルの学習を開始 ---")
    cb_predictor = CatBoostRacePredictor(cat_features=cat_cols)
    cb_predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=30,
    )
    cb_predictor.save("models_saved/catboost_model.cbm")

    cb_test_preds = cb_predictor.predict_proba(X_test)
    cb_metrics = MetricsEvaluator.calculate_all_metrics(y_test, cb_test_preds)
    logger.info(f"CatBoost 単体評価 - AUC: {cb_metrics['auc']}, LogLoss: {cb_metrics['logloss']}")

    # 6. アンサンブル推論 (LightGBM 50% + CatBoost 50%)
    logger.info("--- [3/3] アンサンブル (LightGBM × CatBoost) 評価 ---")
    ensemble_preds = (lgbm_test_preds * 0.5) + (cb_test_preds * 0.5)

    ensemble_metrics = MetricsEvaluator.calculate_all_metrics(y_test, ensemble_preds)
    logger.info(
        f"★ アンサンブル総合評価 - AUC: {ensemble_metrics['auc']}, LogLoss: {ensemble_metrics['logloss']}, Accuracy: {ensemble_metrics['accuracy']}"
    )

    # 7. バックテスト（アンサンブル確率で実行）
    test_df = test_df.copy()
    test_df["pred_prob"] = ensemble_preds

    # 複勝シミュレーション
    BettingSimulator.simulate_place_bet(
        test_df,
        pred_col="pred_prob",
        min_ev_threshold=1.1,
        min_prob_threshold=0.35,
        max_rank_in_race=3,
    )

    # 単勝シミュレーション
    BettingSimulator.simulate_single_bet(
        test_df,
        pred_col="pred_prob",
        min_ev_threshold=1.2,
        min_prob_threshold=0.15,
        max_rank_in_race=2,
    )

    logger.info("=== フェーズ2 パイプライン処理が正常に完了しました ===")


if __name__ == "__main__":
    run_pipeline()