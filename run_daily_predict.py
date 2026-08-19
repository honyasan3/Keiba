"""当日・指定日のレース一括自動推論・Discord通知スクリプト"""
import argparse
import datetime
import time
from typing import List
from config.config_loader import ConfigLoader
from predict import predict_race
from src.common.logger import setup_logger
from src.crawler.race_schedule_scraper import RaceScheduleScraper

logger = setup_logger("run_daily_predict")


def run_daily_predictions(target_date: str, rounds: List[int], notify: bool = True) -> None:
    """指定日の対象レースを一括でスクレイピング・推論・Discord通知"""
    logger.info(f"=== {target_date} のレース一括自動予測を開始します (対象R: {rounds}) ===")

    date_str = target_date.replace("-", "").strip() if target_date else None

    # 1. 指定日・指定ラウンドのレース一覧を取得
    try:
        race_items = RaceScheduleScraper.get_race_ids_by_rounds(
            date_str=date_str, target_rounds=rounds
        )
    except Exception as e:
        logger.error(f"開催日程の取得に失敗しました: {e}")
        return

    if not race_items:
        logger.warning(f"{target_date} の対象レースが見つかりませんでした。")
        return

    logger.info(f"推論対象レース数: {len(race_items)} 件")

    # 2. 順次推論・通知を実行
    for idx, item in enumerate(race_items, 1):
        r_id = item["race_id"]
        r_name = item.get("race_name", "")
        logger.info(f"[{idx}/{len(race_items)}] Race ID: {r_id} ({r_name}) を処理中...")
        try:
            predict_race(race_id=r_id, notify=notify)
        except Exception as e:
            logger.error(f"Race ID: {r_id} の推論処理中にエラーが発生しました: {e}")

        # サーバー負荷軽減のためのインターバル
        if idx < len(race_items):
            time.sleep(2.0)

    logger.info("=== 全対象レースの一括自動予測が完了しました ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="当日・指定日のレース一括自動予測バッチ")
    parser.add_argument("--date", type=str, default="", help="対象日 (YYYY-MM-DD)。未指定の場合は当日")
    parser.add_argument("--rounds", type=str, default="11", help="対象ラウンド (カンマ区切り '9,10,11,12' または 'all')")
    parser.add_argument("--no-notify", action="store_true", help="Discord通知をスキップする場合に指定")
    args = parser.parse_args()

    # 日付設定
    t_date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d")

    # 対象ラウンド設定
    if args.rounds.lower() == "all":
        target_rounds = list(range(1, 13))
    else:
        target_rounds = [int(r.strip()) for r in args.rounds.split(",") if r.strip().isdigit()]

    run_daily_predictions(target_date=t_date, rounds=target_rounds, notify=not args.no_notify)