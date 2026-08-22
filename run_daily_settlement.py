"""当日全レース結果の自動精算・Discord収支レポート送信エントリーポイント"""
import argparse
from datetime import datetime
from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.evaluation.settlement_reporter import SettlementReporter
from src.notification.discord_notifier import DiscordNotifier
from src.pipeline.repository import PredictionRepository

logger = setup_logger("run_settlement")

AUTO_SETTLE_BET_TYPES = ["place", "win", "wide"]


def run_today_settlement(target_date: str = None, notify: bool = True) -> None:
    date_str = target_date or datetime.now().strftime("%Y-%m-%d")
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)
    db_connector.create_tables()

    logger.info(f"=== {date_str} 以前の未精算買い目の結果精算および収支集計を開始します ===")

    reporter = SettlementReporter()

    with db_connector.get_session() as session:
        repo = PredictionRepository(session)
        unsettled = repo.get_unsettled(race_date_lte=date_str, bet_types=AUTO_SETTLE_BET_TYPES)

        if not unsettled:
            logger.info("精算対象の未精算買い目がありません。")
            print("\n精算対象の未精算買い目がありません（predict.py実行後にお試しください）。\n")
            return

        # horse_numはplace/winでは馬番("10")、wideでは"5-10"のようなペア表記。
        # 型変換はSettlementReporter.calculate_settlement側でbet_typeに応じて行うため、ここでは
        # DBに保存されている文字列のまま渡す。
        predicted_bets = [
            {
                "race_id": row.race_id,
                "race_title": row.race_title,
                "bet_type": row.bet_type,
                "horse_num": row.horse_num,
                "horse_name": row.horse_name,
                "bet_amount": row.bet_amount,
            }
            for row in unsettled
        ]

        settlement = reporter.calculate_settlement(predicted_bets)

        settled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row, detail in zip(unsettled, settlement["details"]):
            repo.mark_settled(
                prediction_id=row.id,
                is_hit=detail["is_hit"],
                payout_amount=detail["payout"],
                settled_at=settled_at,
            )

    print("\n" + "=" * 50)
    print(f" 📊 【{date_str} 確定収支レポート】")
    print(f" 対象買い目: {settlement['bet_count']}件 (的中: {settlement['hit_count']}件, 的中率: {settlement['hit_rate']:.1f}%)")
    print(f" 投資: {settlement['total_invest']}円 ｜ 払戻: {settlement['total_return']}円")
    print(f" 収支: {settlement['profit']}円 ｜ 回収率: {settlement['roi']:.1f}%")
    print("=" * 50 + "\n")

    notif_cfg = getattr(config, "notification", None)
    if notify and notif_cfg and getattr(notif_cfg, "enabled", False):
        notifier = DiscordNotifier(webhook_url=notif_cfg.discord_webhook_url, enabled=True)
        notifier.send_settlement_report(settlement, date_str=date_str)
        logger.info("Discordへ収支レポートを送信しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="当日以前の未精算買い目の自動精算バッチ")
    parser.add_argument("target_date", type=str, nargs="?", default=None, help="対象日 (YYYY-MM-DD)。未指定の場合は当日")
    parser.add_argument("--no-notify", action="store_true", help="Discord通知をスキップする場合に指定")
    args = parser.parse_args()

    run_today_settlement(target_date=args.target_date, notify=not args.no_notify)
