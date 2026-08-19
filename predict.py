"""リアルタイムレース予想・推論スクリプト（LightGBM × CatBoost アンサンブル × モンテカルロシミュレータ統合版）"""
import argparse
import os
import re
import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.crawler.race_scraper import RaceScraper
from src.crawler.shutuba_parser import ShutubaHtmlParser
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.notification.discord_notifier import DiscordNotifier
from src.pipeline.cleaner import DataCleaner
from src.pipeline.repository import RaceModel, RaceResultModel
from src.simulation.race_simulator import MonteCarloRaceSimulator

logger = setup_logger("predict")


def get_historical_data(db_connector: DatabaseConnector) -> pd.DataFrame:
    """DBから過去のレース実績データを抽出"""
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


def predict_race(
    race_id: str,
    lgbm_model_path: str = "models_saved/lgbm_model.txt",
    catboost_model_path: str = "models_saved/catboost_model.cbm",
    notify: bool = True,
) -> None:
    logger.info(f"=== Race ID: {race_id} のアンサンブル推論を開始します ===")

    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    scraper = RaceScraper(config.crawler)
    try:
        html_content = scraper.fetch_race_card(race_id, use_cache=True)
        raw_card = ShutubaHtmlParser.parse_shutuba_card(html_content, race_id)
    except Exception as e:
        logger.error(f"出馬表の取得に失敗しました: {e}")
        return

    cleaned_entries = []
    for item in raw_card.get("entries", []):
        gender, age = None, None
        sex_age = item.get("sex_age_raw", "")
        if sex_age:
            gender = sex_age[0]
            age_m = re.search(r"\d+", sex_age)
            age = int(age_m.group()) if age_m else None

        hw, hw_diff = None, None
        hw_raw = item.get("horse_weight_raw", "")
        hw_m = re.search(r"(\d+)(?:\(([-+]?\d+)\))?", hw_raw)
        if hw_m:
            hw = int(hw_m.group(1))
            if hw_m.group(2):
                hw_diff = int(hw_m.group(2))

        cleaned_entries.append({
            "race_id": race_id,
            "race_title": raw_card.get("race_title"),
            "race_date": raw_card.get("race_date") or pd.Timestamp.now().strftime("%Y-%m-%d"),
            "race_round": raw_card.get("race_round") or 11,
            "course_type": raw_card.get("course_type") or "芝",
            "distance": raw_card.get("distance") or 1600,
            "weather": raw_card.get("weather") or "晴",
            "track_condition": raw_card.get("track_condition") or "良",
            "rank": None,
            "bracket_num": DataCleaner._extract_int(item.get("bracket_num_raw")),
            "horse_num": DataCleaner._extract_int(item.get("horse_num_raw")),
            "horse_name": item.get("horse_name", "").strip(),
            "horse_id": item.get("horse_id", ""),
            "gender": gender,
            "age": age,
            "jockey_weight": DataCleaner._extract_float(item.get("jockey_weight_raw")),
            "jockey_name": item.get("jockey_name", "").strip(),
            "finish_time_sec": None,
            "margin": None,
            "passage_order": None,
            "last_3f_time": None,
            "odds": DataCleaner._extract_float(item.get("odds_raw")) or 10.0,
            "popularity": DataCleaner._extract_int(item.get("popularity_raw")),
            "horse_weight": hw or 470,
            "horse_weight_diff": hw_diff or 0,
        })

    target_df = pd.DataFrame(cleaned_entries)
    if target_df.empty:
        logger.error("出走馬データが抽出できませんでした。")
        return

    hist_df = get_historical_data(db_connector)
    # DB内に既に存在する同じrace_idのレコードを過去データ側から除外して重複を防ぐ
    hist_df = hist_df[hist_df["race_id"] != race_id].copy()
    combined_df = pd.concat([hist_df, target_df], ignore_index=True)

    # 特徴量抽出
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3)

    combined_df = race_fe.transform(combined_df)
    combined_df = horse_fe.transform(combined_df)

    infer_df = combined_df[combined_df["race_id"] == race_id].copy().reset_index(drop=True)

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

    # 1. LightGBM & CatBoost によるアンサンブル複勝確率予測
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.load(lgbm_model_path)
    lgbm_probs = lgbm_predictor.predict_proba(infer_df[feature_cols])

    if os.path.exists(catboost_model_path):
        cb_predictor = CatBoostRacePredictor()
        cb_predictor.load(catboost_model_path)
        cb_probs = cb_predictor.predict_proba(infer_df[feature_cols])
        # アンサンブル (LightGBM 50% + CatBoost 50%)
        infer_df["pred_place_prob"] = (lgbm_probs * 0.5) + (cb_probs * 0.5)
    else:
        logger.warning("CatBoostモデルが見つからないため、LightGBM単体で推論します。")
        infer_df["pred_place_prob"] = lgbm_probs

    # 2. モンテカルロシミュレーション（ワイド・馬連・ケリー対応）
    simulator = MonteCarloRaceSimulator(n_simulations=10000)
    result_df, wide_df, umaren_df = simulator.simulate_race(infer_df, bankroll=10000)

    # 総合複勝率でソート
    result_df = result_df.sort_values("ensemble_place_prob", ascending=False).reset_index(drop=True)
    result_df["pred_rank"] = result_df.index + 1

    # 全頭一覧テーブル表示
    print("\n" + "=" * 75)
    print(f" レース予想: {raw_card.get('race_title', '')} (ID: {race_id})")
    print(f" 条件: {raw_card.get('course_type')} {raw_card.get('distance')}m 天候:{raw_card.get('weather')} 馬場:{raw_card.get('track_condition')}")
    print("=" * 75 + "\n")

    display_cols = [
        "pred_rank", "horse_num", "bracket_num", "horse_name", "jockey_name",
        "odds", "place_odds_est", "pred_place_prob", "sim_win_prob", "ensemble_place_prob", "ev_place", "kelly_bet_place"
    ]
    table_view = result_df[display_cols].copy()
    table_view["pred_place_prob"] = (table_view["pred_place_prob"] * 100).round(1).astype(str) + "%"
    table_view["sim_win_prob"] = (table_view["sim_win_prob"] * 100).round(1).astype(str) + "%"
    table_view["ensemble_place_prob"] = (table_view["ensemble_place_prob"] * 100).round(1).astype(str) + "%"
    table_view["place_odds_est"] = table_view["place_odds_est"].round(1)
    table_view["ev_place"] = table_view["ev_place"].round(2)
    table_view["kelly_bet_place"] = table_view["kelly_bet_place"].astype(str) + "円"
    table_view.columns = ["総合順位", "馬番", "枠", "馬名", "騎手", "単勝オッズ", "推定複勝", "AI複勝率", "シミュ勝率", "総合複勝率", "複勝EV", "ケリー推奨額"]

    print(tabulate(table_view, headers="keys", tablefmt="fancy_grid", showindex=False))

    # 買い目判定（テスト検証済みの最適化ルール）
    place_rec = result_df[
        (result_df["ev_place"] >= 1.0)
        & (result_df["ensemble_place_prob"] >= 0.45)
        & (result_df["pred_rank"] <= 3)
        & (result_df["odds"] >= 5.0)
    ]

    win_candidate = result_df[
        (result_df["pred_rank"] == 1)
        & (result_df["sim_win_prob"] >= 0.20)
        & (result_df["odds"] >= 10.0)
    ]

    # ワイド推奨 (EV >= 1.25 かつ 確率 >= 15%)
    wide_rec = wide_df[(wide_df["ev"] >= 1.25) & (wide_df["prob"] >= 0.15)].head(3) if not wide_df.empty else pd.DataFrame()

    print("\n" + "=" * 65)
    print(" 🎯 【AI × シミュレーション 厳選推奨買い目（資金傾斜付き）】")
    print("=" * 65)
    if not place_rec.empty:
        for _, row in place_rec.iterrows():
            print(
                f" 🟢 複勝推奨: [{row['horse_num']}番] {row['horse_name']} "
                f"(総合複勝率: {row['ensemble_place_prob']*100:.1f}%, 想定: {row['place_odds_est']:.1f}倍, EV: {row['ev_place']:.2f}) "
                f"👉 推奨賭け金: 【{row['kelly_bet_place']}円】"
            )
    else:
        print(" ⏸️ 複勝: 基準を満たす馬がいないため【見送り (KEN)】")

    if not win_candidate.empty:
        for _, row in win_candidate.iterrows():
            print(
                f" 🟠 単勝穴狙い: [{row['horse_num']}番] {row['horse_name']} "
                f"(単勝: {row['odds']:.1f}倍, シミュ勝率: {row['sim_win_prob']*100:.1f}%, 総合複勝率: {row['ensemble_place_prob']*100:.1f}%)"
            )

    if not wide_rec.empty:
        print("-" * 65)
        for _, row in wide_rec.iterrows():
            print(
                f" 🔵 ワイド推奨: [{row['pair']}] {row['names']} "
                f"(的中率: {row['prob']*100:.1f}%, 想定: {row['est_odds']}倍, EV: {row['ev']:.2f})"
            )
    print("=" * 65 + "\n")

    # 3. Discord通知
    if notify:
        notif_cfg = getattr(config, "notification", None)
        if notif_cfg and getattr(notif_cfg, "enabled", False):
            notifier = DiscordNotifier(webhook_url=notif_cfg.discord_webhook_url, enabled=True)
            notifier.send_prediction_report(
                race_info=raw_card,
                top_entries=result_df.to_dict(orient="records"),
                place_recommendations=place_rec.to_dict(orient="records"),
                win_recommendations=win_candidate.to_dict(orient="records"),
                wide_recommendations=wide_rec.to_dict(orient="records") if not wide_rec.empty else [],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬予想AI 推論スクリプト")
    parser.add_argument("race_id", type=str, help="推論対象の12桁レースID (例: 202405020811)")
    parser.add_argument("--no-notify", action="store_true", help="Discord通知をスキップする場合に指定")
    args = parser.parse_args()

    predict_race(race_id=args.race_id, notify=not args.no_notify)