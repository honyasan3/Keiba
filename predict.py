"""リアルタイムレース予想・推論スクリプト（単勝・複勝・Discord通知対応版）"""
import argparse
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
from src.models.lgbm_model import LGBMRacePredictor
from src.notification.discord_notifier import DiscordNotifier
from src.pipeline.cleaner import DataCleaner
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("predict")


def get_historical_data(db_connector: DatabaseConnector) -> pd.DataFrame:
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
    model_path: str = "models_saved/lgbm_model.txt",
    notify: bool = True,
) -> None:
    logger.info(f"=== Race ID: {race_id} の推論を開始します ===")

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
    combined_df = pd.concat([hist_df, target_df], ignore_index=True)

    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3)

    combined_df = race_fe.transform(combined_df)
    combined_df = horse_fe.transform(combined_df)

    infer_df = combined_df[combined_df["race_id"] == race_id].copy()

    predictor = LGBMRacePredictor()
    predictor.load(model_path)

    feature_cols = [
        "venue_code", "race_round", "distance", "course_type_cat", "weather_cat",
        "track_condition_cat", "bracket_num", "horse_num", "gender_cat", "age",
        "jockey_weight", "jockey_weight_diff_from_race_mean", "race_horse_count",
        "horse_weight", "horse_weight_diff", "horse_past_runs", "horse_past_avg_rank",
        "horse_past_win_rate", "horse_past_place_rate", "horse_avg_passage_rate",
        "distance_diff", "horse_recent3_avg_rank", "horse_recent3_avg_last3f",
        "days_since_prev_race", "jockey_past_win_rate", "jockey_past_place_rate",
        "jockey_venue_place_rate"
    ]

    infer_df["pred_place_prob"] = predictor.predict_proba(infer_df[feature_cols])
    infer_df["place_odds_est"] = (infer_df["odds"].fillna(1.0) ** 0.45).clip(lower=1.1)
    infer_df["ev_place"] = infer_df["pred_place_prob"] * infer_df["place_odds_est"]

    result_df = infer_df.sort_values("pred_place_prob", ascending=False).reset_index(drop=True)
    result_df["pred_rank"] = result_df.index + 1

    # 1. 全頭一覧テーブル表示
    print(f"\n=======================================================")
    print(f" レース予想: {raw_card.get('race_title', '')} (ID: {race_id})")
    print(f" 条件: {raw_card.get('course_type')} {raw_card.get('distance')}m 天候:{raw_card.get('weather')} 馬場:{raw_card.get('track_condition')}")
    print(f"=======================================================\n")

    display_cols = ["pred_rank", "horse_num", "bracket_num", "horse_name", "jockey_name", "odds", "place_odds_est", "pred_place_prob", "ev_place"]
    table_view = result_df[display_cols].copy()
    table_view["pred_place_prob"] = (table_view["pred_place_prob"] * 100).round(1).astype(str) + "%"
    table_view["place_odds_est"] = table_view["place_odds_est"].round(1)
    table_view["ev_place"] = table_view["ev_place"].round(2)
    table_view.columns = ["予想順位", "馬番", "枠", "馬名", "騎手", "単勝オッズ", "推定複勝", "予測複勝率", "複勝EV"]

    print(tabulate(table_view, headers="keys", tablefmt="fancy_grid", showindex=False))

    # 2. 買い目判定（最適化された回収率121%ルール）
    # 複勝: EV >= 1.5, 複勝率 >= 30%, 予測2位以内, 単勝オッズ >= 3.0倍
    place_rec = result_df[
        (result_df["ev_place"] >= 1.5)
        & (result_df["pred_place_prob"] >= 0.30)
        & (result_df["pred_rank"] <= 2)
        & (result_df["odds"] >= 3.0)
    ]

    # 単勝狙い目: 予測1位かつ複勝率40%以上かつ単勝3〜20倍（参考枠）
    win_candidate = result_df[
        (result_df["pred_rank"] == 1)
        & (result_df["pred_place_prob"] >= 0.40)
        & (result_df["odds"] >= 3.0)
        & (result_df["odds"] <= 20.0)
    ]

    # コンソール側への買い目出力
    print("\n" + "=" * 55)
    print(" 🎯 【AI厳選推奨買い目判定】")
    print("=" * 55)
    if not place_rec.empty:
        for _, row in place_rec.iterrows():
            print(f" 🟢 複勝推奨: [{row['horse_num']}番] {row['horse_name']} "
                  f"(複勝率: {row['pred_place_prob']*100:.1f}%, 想定: {row['place_odds_est']:.1f}倍, EV: {row['ev_place']:.2f})")
    else:
        print(" ⏸️ 複勝: 基準（EV≧1.5, 複勝率≧30%, 予測2位以内）を満たす馬がいないため【見送り (KEN)】")

    if not win_candidate.empty:
        for _, row in win_candidate.iterrows():
            print(f" 🟠 単勝狙い: [{row['horse_num']}番] {row['horse_name']} "
                  f"(単勝オッズ: {row['odds']:.1f}倍, 複勝圏内率: {row['pred_place_prob']*100:.1f}%)")
    print("=" * 55 + "\n")

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
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬予想AI 推論スクリプト")
    parser.add_argument("race_id", type=str, help="推論対象の12桁レースID (例: 202405020811)")
    parser.add_argument("--no-notify", action="store_true", help="Discord通知をスキップする場合に指定")
    args = parser.parse_args()

    predict_race(race_id=args.race_id, notify=not args.no_notify)