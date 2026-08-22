"""トリプルアンサンブル（LGBM×CatBoost×LambdaMART）推論・学習の共通ヘルパー

optimize_betting.py / evaluate_calibration.py / rolling_walk_forward.py など、DB上の全レコードに
対してまとめて推論・検証したいスクリプトから共通利用する。predict.py（単一レースのライブ推論）とは
目的が異なるため独立させている。
"""
import os
from typing import List, Optional, Tuple
import pandas as pd
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.models.ranker_model import LGBMRankPredictor

# 全46特徴量リスト（main_phase2.py / predict.py / backtest_simulation.py と完全一致させること）
FEATURE_COLS: List[str] = [
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

# CatBoostのカテゴリ特徴量リスト（main_phase2.py と完全一致させること）
CAT_COLS: List[str] = [
    "venue_code", "course_type_cat", "weather_cat", "track_condition_cat",
    "gender_cat", "age_gender_cat", "rest_category_cat", "distance_shock_cat",
    "race_expected_pace_cat",
]


def train_ensemble_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    cat_cols: List[str] = CAT_COLS,
) -> Tuple[LGBMRacePredictor, CatBoostRacePredictor, LGBMRankPredictor]:
    """train_df/val_dfからトリプルアンサンブルの3モデルを学習する（main_phase2.pyと同一ロジック）。

    学習したモデルはメモリ上のオブジェクトとして返すのみで、models_saved/ への保存は行わない
    （rolling_walk_forward.py のような検証目的での複数回学習で、本番デプロイ済みモデルを
    誤って上書きしないようにするため）。
    """
    if "target_place" not in train_df.columns:
        train_df = train_df.copy()
        train_df["target_place"] = train_df["rank"].apply(lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0)
    if "target_place" not in val_df.columns:
        val_df = val_df.copy()
        val_df["target_place"] = val_df["rank"].apply(lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0)

    X_train, y_train = train_df[feature_cols], train_df["target_place"]
    X_val, y_val = val_df[feature_cols], val_df["target_place"]

    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=30)

    cb_predictor = CatBoostRacePredictor(cat_features=cat_cols)
    cb_predictor.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=30)

    rank_predictor = LGBMRankPredictor()
    rank_predictor.train(train_df=train_df, val_df=val_df, feature_cols=feature_cols, early_stopping_rounds=50)

    return lgbm_predictor, cb_predictor, rank_predictor


def run_ensemble_inference(
    df: pd.DataFrame,
    lgbm_predictor: Optional[LGBMRacePredictor] = None,
    cb_predictor: Optional[CatBoostRacePredictor] = None,
    rank_predictor: Optional[LGBMRankPredictor] = None,
    lgbm_model_path: str = "models_saved/lgbm_model.txt",
    catboost_model_path: str = "models_saved/catboost_model.cbm",
    ranker_model_path: str = "models_saved/lambdarank_model.txt",
) -> pd.DataFrame:
    """トリプルアンサンブル推論を実行し、pred_place_prob / pred_rank を付与して返す（40:40:20 Blend）

    lgbm_predictor等をメモリ上のモデルオブジェクトとして渡した場合はそちらを使用し（rolling
    walk-forwardでfoldごとに学習したモデルを使う場合など）、渡さなければ models_saved/ の
    本番デプロイ済みモデルをディスクから読み込む（optimize_betting.py等の既存挙動と同じ）。
    """
    df = df.copy()

    if lgbm_predictor is None:
        lgbm_predictor = LGBMRacePredictor()
        lgbm_predictor.load(lgbm_model_path)
    lgbm_probs = lgbm_predictor.predict_proba(df[FEATURE_COLS])

    if cb_predictor is None and os.path.exists(catboost_model_path):
        cb_predictor = CatBoostRacePredictor()
        cb_predictor.load(catboost_model_path)
    cb_probs = cb_predictor.predict_proba(df[FEATURE_COLS]) if cb_predictor is not None else lgbm_probs

    if rank_predictor is None and os.path.exists(ranker_model_path):
        rank_predictor = LGBMRankPredictor()
        rank_predictor.load(ranker_model_path)

    if rank_predictor is not None:
        rank_scores = rank_predictor.predict_score(df)
        df["_rank_score"] = rank_scores
        rank_norm_scores = df.groupby("race_id")["_rank_score"].rank(pct=True).values
    else:
        rank_norm_scores = lgbm_probs

    df["pred_place_prob"] = (lgbm_probs * 0.40) + (cb_probs * 0.40) + (rank_norm_scores * 0.20)
    df["pred_rank"] = (
        df.groupby("race_id")["pred_place_prob"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return df
