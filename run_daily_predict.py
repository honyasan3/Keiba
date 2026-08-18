"""指定日（または本日）のレース自動推論・一括通知スクリプト（全R・複数R指定対応）"""
import argparse
import time
from typing import List, Optional
from src.common.logger import setup_logger
from src.crawler.race_schedule_scraper import RaceScheduleScraper
from predict import predict_race

logger = setup_logger("daily_predict")


def run_daily_predictions(
    target_date: Optional[str] = None,
    target_rounds: Optional[List[int]] = None,
    notify: bool = True,
) -> None:
    round_desc = "全レース (1〜12R)" if target_rounds is None else f"{target_rounds}R"
    logger.info(f"=== 一括推論パイプラインを開始します (対象: {round_desc}) ===")

    # 1. 指定条件のレース一覧を取得
    target_races = RaceScheduleScraper.get_race_ids_by_rounds(
        date_str=target_date, target_rounds=target_rounds
    )
    if not target_races:
        logger.warning("対象のレースが見つかりませんでした。開催日・日程を確認してください。")
        return

    logger.info(f"推論対象レース数: {len(target_races)} 件")

    # 2. 各レースを順次推論・通知
    for idx, race in enumerate(target_races, start=1):
        race_id = race["race_id"]
        race_name = race.get("race_name", "")
        race_round = race.get("round", "")
        logger.info(f"[{idx}/{len(target_races)}] {race_round} (ID: {race_id} / {race_name}) の予想を開始")

        try:
            predict_race(race_id=race_id, notify=notify)
        except Exception as e:
            logger.error(f"Race ID {race_id} の推論中にエラーが発生しました: {e}")

        # サーバー負荷軽減の待機
        time.sleep(2.0)

    logger.info("=== 全対象レースの推論・通知が完了しました ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="レース一括推論・Discord通知")
    parser.add_argument("--date", type=str, default=None, help="対象日付 (YYYYMMDD形式 例: 20250223)。未指定時は本日")
    parser.add_argument(
        "--rounds",
        type=str,
        default="11",
        help="対象レース番号（カンマ区切り 例: '11', '10,11,12'）。'all' 指定で全レース(1〜12R)",
    )
    parser.add_argument("--no-notify", action="store_true", help="Discord通知を無効化する場合に指定")
    args = parser.parse_args()

    # ラウンド引数のパース
    if args.rounds.lower() == "all":
        rounds_list = None  # 全レース
    else:
        rounds_list = [int(r.strip()) for r in args.rounds.split(",") if r.strip().isdigit()]

    run_daily_predictions(target_date=args.date, target_rounds=rounds_list, notify=not args.no_notify)