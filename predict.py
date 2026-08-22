"""リアルタイムレース予想・推論スクリプト（トリプルアンサンブル: LGBM × CatBoost × LambdaMART & Elo & 当日トラックバイアス統合版）"""
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
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.models.catboost_model import CatBoostRacePredictor
from src.models.lgbm_model import LGBMRacePredictor
from src.models.ranker_model import LGBMRankPredictor
from src.notification.discord_notifier import DiscordNotifier
from src.pipeline.cleaner import DataCleaner
from src.pipeline.repository import PredictionRepository, RaceModel, RaceResultModel
from src.simulation.race_simulator import MonteCarloRaceSimulator

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
    lgbm_model_path: str = "models_saved/lgbm_model.txt",
    catboost_model_path: str = "models_saved/catboost_model.cbm",
    ranker_model_path: str = "models_saved/lambdarank_model.txt",
    notify: bool = True,
) -> None:
    logger.info(f"=== Race ID: {race_id} のトリプルアンサンブル推論を開始します ===")

    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)
    db_connector.create_tables()

    scraper = RaceScraper(config.crawler)
    try:
        html_content = scraper.fetch_race_card(race_id, use_cache=True)
        raw_card = ShutubaHtmlParser.parse_shutuba_card(html_content, race_id)
    except Exception as e:
        logger.error(f"出馬表の取得に失敗しました: {e}")
        return

    # オッズは出馬表HTMLに静的に埋め込まれておらず、別APIから取得する必要がある。
    # JRAはレース直前までオッズを公表しないため、未公表の場合は空dictが返る。
    live_odds = scraper.fetch_live_odds(race_id)
    odds_published = bool(live_odds)
    if not odds_published:
        logger.warning(
            f"オッズが未公表のため、本レースの買い目判定（EV・ケリー推奨額）は行いません "
            f"(Race ID: {race_id})。発走が近づいてから再実行してください。"
        )

    cleaned_entries = []
    weight_missing_count = 0
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
        else:
            weight_missing_count += 1

        horse_num = DataCleaner._extract_int(item.get("horse_num_raw"))
        odds_info = live_odds.get(horse_num) if horse_num is not None else None

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
            "horse_num": horse_num,
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
            # オッズ未公表の場合はNone（架空の値で代替しない。EV計算は自然に対象外になる）
            "odds": odds_info["odds"] if odds_info else None,
            "popularity": odds_info["popularity"] if odds_info else None,
            # 複勝オッズの実測レンジ（取得できない場合はNone。シミュレータ側で近似式にフォールバックする）
            "real_place_odds_min": odds_info.get("place_odds_min") if odds_info else None,
            "real_place_odds_max": odds_info.get("place_odds_max") if odds_info else None,
            "horse_weight": hw or 470,
            "horse_weight_diff": hw_diff or 0,
        })

    target_df = pd.DataFrame(cleaned_entries)
    if target_df.empty:
        logger.error("出走馬データが抽出できませんでした。")
        return

    if weight_missing_count > 0:
        logger.warning(
            f"馬体重が未発表の馬が{weight_missing_count}/{len(target_df)}頭います "
            f"(Race ID: {race_id})。JRAは通常発走1時間前に発表するため、それまでは仮値(470kg・増減0)で"
            f"補完されており、AI予測の精度がわずかに低下する可能性があります。発表後の再実行を推奨します。"
        )

    # 枠番は出馬表HTMLから取得できず全行Noneになるため、Noneのままだと列がobject型になり
    # concat時に既存データ(int/float)と型が衝突してLightGBM等が受け付けなくなる。
    # 明示的にfloat64化してNaNとして扱う（モデルはNaNを欠損として正しく処理できる）。
    for col in ("bracket_num", "odds", "popularity", "real_place_odds_min", "real_place_odds_max"):
        target_df[col] = pd.to_numeric(target_df[col], errors="coerce")

    hist_df = get_historical_data(db_connector)
    hist_df = hist_df[hist_df["race_id"] != race_id].copy()
    combined_df = pd.concat([hist_df, target_df], ignore_index=True)

    # 特徴量抽出（ピュア能力・Eloレーティング・当日トラックバイアス）
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3, elo_k_factor=16.0)
    bias_fe = TrackBiasFeatureExtractor()

    combined_df = race_fe.transform(combined_df)
    combined_df = horse_fe.transform(combined_df)
    combined_df = bias_fe.transform(combined_df)

    infer_df = combined_df[combined_df["race_id"] == race_id].copy().reset_index(drop=True)

    # 全46特徴量リスト（main_phase2と完全一致）
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

    # 1. トリプルアンサンブル推論 (LGBM 40% + CatBoost 40% + Ranker 20%)
    lgbm_predictor = LGBMRacePredictor()
    lgbm_predictor.load(lgbm_model_path)
    lgbm_probs = lgbm_predictor.predict_proba(infer_df[feature_cols])

    if os.path.exists(catboost_model_path):
        cb_predictor = CatBoostRacePredictor()
        cb_predictor.load(catboost_model_path)
        cb_probs = cb_predictor.predict_proba(infer_df[feature_cols])
    else:
        cb_probs = lgbm_probs

    if os.path.exists(ranker_model_path):
        rank_predictor = LGBMRankPredictor()
        rank_predictor.load(ranker_model_path)
        rank_scores = rank_predictor.predict_score(infer_df)
        infer_df["_rank_score"] = rank_scores
        rank_norm_scores = infer_df["_rank_score"].rank(pct=True).values
    else:
        rank_norm_scores = lgbm_probs

    infer_df["pred_place_prob"] = (lgbm_probs * 0.40) + (cb_probs * 0.40) + (rank_norm_scores * 0.20)

    # 2. モンテカルロシミュレーション
    simulator = MonteCarloRaceSimulator(n_simulations=10000)
    result_df, wide_df, umaren_df = simulator.simulate_race(infer_df, bankroll=10000)

    result_df = result_df.sort_values("ensemble_place_prob", ascending=False).reset_index(drop=True)
    result_df["pred_rank"] = result_df.index + 1

    # 一覧テーブル表示
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
    table_view["odds"] = table_view["odds"].apply(lambda v: f"{v:.1f}" if pd.notnull(v) else "未定")
    table_view["place_odds_est"] = table_view["place_odds_est"].round(1)
    table_view["ev_place"] = table_view["ev_place"].round(2)
    table_view["kelly_bet_place"] = table_view["kelly_bet_place"].astype(str) + "円"
    table_view.columns = ["総合順位", "馬番", "枠", "馬名", "騎手", "単勝オッズ", "推定複勝", "AI複勝率", "シミュ勝率", "総合複勝率", "複勝EV", "ケリー推奨額"]

    print(tabulate(table_view, headers="keys", tablefmt="fancy_grid", showindex=False))
    if not odds_published:
        print("\n ⚠️  オッズ未公表のため、単勝オッズ・複勝EV・ケリー推奨額は参考値になりません。買い目判定は行いません。")

    # 買い目判定（いずれもrolling_walk_forward.pyによる3fold独立検証: 各foldでモデルを再学習し、
    # Validation期間のみで選定→一度も選定に使っていないTest期間に適用、の結果。詳細はREADME「6.1」）
    # 複勝: 3fold中2foldが単勝オッズ3.0〜5.0倍という同じオッズ帯を独立選定（残り1foldも2〜6倍と近い）。
    #       3fold合算326件でのプールROIは110.34%。
    # ワイド: 3fold中2foldがEV≥1.0/複勝率≥0.3/オッズ3〜10倍という完全に同一のルールを独立選定。
    #       3fold合算15,078件でのプールROIは120.00%と、この一連の検証の中で最も件数が多く安定した結果。
    # 単勝: 同様の3fold検証でオッズ帯・EVしきい値ともに一貫した優位性が確認できなかった
    # （3fold合算ROI 80.77%、選定ルールもfold間で安定しない）ため、根拠不足として無効化している。
    # オッズ未公表の場合、odds列がNaNとなり各フィルタは自然に全滅する（架空オッズで判定しない）が、
    # 「見送り」と「オッズ未公表」は意味が異なるため明示的に空DataFrameへ倒す。
    if odds_published:
        place_rec = result_df[
            (result_df["ev_place"] >= 1.4)
            & (result_df["ensemble_place_prob"] >= 0.45)
            & (result_df["pred_rank"] <= 3)
            & (result_df["odds"] >= 3.0)
            & (result_df["odds"] <= 5.0)
        ]

        # 単勝は根拠不足のため無効化（上記コメント参照）
        win_candidate = result_df.iloc[0:0]

        wide_rec = wide_df[
            (wide_df["ev"] >= 1.0)
            & (wide_df["prob"] >= 0.3)
            & (wide_df["est_odds"] >= 3.0)
            & (wide_df["est_odds"] <= 10.0)
        ].head(3) if not wide_df.empty else pd.DataFrame()
    else:
        place_rec = result_df.iloc[0:0]
        win_candidate = result_df.iloc[0:0]
        wide_rec = pd.DataFrame()

    print("\n" + "=" * 65)
    print(" 🎯 【トリプルAI × シミュレーション 厳選推奨買い目（資金傾斜付き）】")
    print("=" * 65)
    if not odds_published:
        print(" ⏸️ オッズ未公表のため買い目判定は保留です。発走が近づいてから再実行してください。")
    elif not win_candidate.empty:
        for _, row in win_candidate.iterrows():
            print(
                f" 🟠 単勝穴狙い: [{row['horse_num']}番] {row['horse_name']} "
                f"(単勝: {row['odds']:.1f}倍, シミュ勝率: {row['sim_win_prob']*100:.1f}%, 総合複勝率: {row['ensemble_place_prob']*100:.1f}%)"
            )

    if odds_published:
        if not place_rec.empty:
            for _, row in place_rec.iterrows():
                print(
                    f" 🟢 複勝推奨: [{row['horse_num']}番] {row['horse_name']} "
                    f"(総合複勝率: {row['ensemble_place_prob']*100:.1f}%, 想定: {row['place_odds_est']:.1f}倍, EV: {row['ev_place']:.2f}) "
                    f"👉 推奨賭け金: 【{row['kelly_bet_place']}円】"
                )
        else:
            print(" ⏸️ 複勝: 基準を満たす馬がいないため【見送り (KEN)】")

    if odds_published:
        print("-" * 65)
        if not wide_rec.empty:
            for _, row in wide_rec.iterrows():
                print(
                    f" 🔵 ワイド推奨: [{row['pair']}] {row['names']} "
                    f"(的中率: {row['prob']*100:.1f}%, 想定: {row['est_odds']}倍, EV: {row['ev']:.2f})"
                )
        else:
            print(" ⏸️ ワイド: 基準を満たす組み合わせがないため【見送り (KEN)】")
    print("=" * 65 + "\n")

    # 買い目記録（実運用の的中率・回収率トラックレコードを残すため、通知有無に関わらず必ず保存する）
    predicted_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_records = []
    for _, row in place_rec.iterrows():
        prediction_records.append({
            "bet_type": "place",
            "horse_num": row["horse_num"],
            "horse_name": row["horse_name"],
            "odds_at_predict": row["odds"],
            "pred_prob": row["ensemble_place_prob"],
            "ev": row["ev_place"],
            "bet_amount": int(row["kelly_bet_place"]) if row["kelly_bet_place"] else 100,
            "predicted_at": predicted_at,
        })
    for _, row in win_candidate.iterrows():
        prediction_records.append({
            "bet_type": "win",
            "horse_num": row["horse_num"],
            "horse_name": row["horse_name"],
            "odds_at_predict": row["odds"],
            "pred_prob": row["sim_win_prob"],
            "ev": row["ev_place"],
            "bet_amount": 100,
            "predicted_at": predicted_at,
        })
    for _, row in wide_rec.iterrows():
        prediction_records.append({
            "bet_type": "wide",
            "horse_num": row["pair"],
            "horse_name": row["names"],
            "odds_at_predict": row["est_odds"],
            "pred_prob": row["prob"],
            "ev": row["ev"],
            "bet_amount": 100,
            "predicted_at": predicted_at,
        })

    if prediction_records:
        with db_connector.get_session() as session:
            PredictionRepository(session).save_predictions(
                race_id=race_id,
                race_title=raw_card.get("race_title"),
                race_date=raw_card.get("race_date") or pd.Timestamp.now().strftime("%Y-%m-%d"),
                records=prediction_records,
            )

    # Discord通知
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