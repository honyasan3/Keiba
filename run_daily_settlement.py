"""当日全レース結果の自動精算・Discord収支レポート送信エントリーポイント"""
from datetime import datetime
import sys
from config.config_loader import ConfigLoader
from src.common.logger import setup_logger
from src.evaluation.settlement_reporter import SettlementReporter
from src.notification.discord_notifier import DiscordNotifier

logger = setup_logger("run_settlement")


def run_today_settlement(target_date: str = None) -> None:
    date_str = target_date or datetime.now().strftime("%Y-%m-%d")
    config = ConfigLoader.load_config("config/settings.yaml")

    logger.info(f"=== {date_str} のレース結果精算および収支集計を開始します ===")

    reporter = SettlementReporter()

    # 有馬記念でのテスト精算データ
    sample_bets = [
        {
            "race_id": "202506050811",
            "race_title": "有馬記念(GI)",
            "bet_type": "place",
            "horse_num": 13,
            "horse_name": "アドマイヤテラ",
            "bet_amount": 1000,
        }
    ]

    settlement = reporter.calculate_settlement(sample_bets)

    print("\n" + "=" * 50)
    print(f" 📊 【{date_str} 確定収支レポート】")
    print(f" 投資: {settlement['total_invest']}円 ｜ 払戻: {settlement['total_return']}円")
    print(f" 収支: {settlement['profit']}円 ｜ 回収率: {settlement['roi']:.1f}%")
    print("=" * 50 + "\n")

    notif_cfg = getattr(config, "notification", None)
    if notif_cfg and getattr(notif_cfg, "enabled", False):
        notifier = DiscordNotifier(webhook_url=notif_cfg.discord_webhook_url, enabled=True)
        notifier.send_settlement_report(settlement, date_str=date_str)
        logger.info("Discordへ収支レポートを送信しました。")


if __name__ == "__main__":
    t_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_today_settlement(t_date)