"""Discord Webhook 通知モジュール（高視認性UI版）"""
from typing import Any, Dict, List
import requests
from src.common.logger import setup_logger

logger = setup_logger("discord_notifier")


class DiscordNotifier:
    """Discord Webhook経由でレース推論結果をリッチ通知するクラス"""

    def __init__(self, webhook_url: str, enabled: bool = True) -> None:
        self.webhook_url = webhook_url
        self.enabled = enabled

    def send_prediction_report(
        self,
        race_info: Dict[str, Any],
        top_entries: List[Dict[str, Any]],
        place_recommendations: List[Dict[str, Any]],
        win_recommendations: List[Dict[str, Any]],
    ) -> bool:
        if not self.enabled or not self.webhook_url or "YOUR_DISCORD" in self.webhook_url:
            logger.info("Discord通知は無効化されているか、Webhook URLが未設定です。")
            return False

        race_id = race_info.get("race_id", "")
        race_title = race_info.get("race_title", "レース予想")
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

        # 買い目の有無でEmbedのアクセントカラーを変更 (推奨あり: 翠緑, 見送り: スレートグレー)
        has_bet = bool(place_recommendations or win_recommendations)
        embed_color = 0x2ECC71 if has_bet else 0x95A5A6

        # コース条件ヘッダー
        conditions = (
            f"**{race_info.get('course_type', '')} {race_info.get('distance', '')}m** ｜ "
            f"天候: `{race_info.get('weather', '-')}` ｜ 馬場: `{race_info.get('track_condition', '-')}`"
        )

        fields = []

        # 1. 厳選買い目セクション
        if place_recommendations or win_recommendations:
            bet_lines = []
            for r in place_recommendations:
                bet_lines.append(
                    f"🟢 **複勝** `[{r['horse_num']:>02}番]` **{r['horse_name']}** "
                    f"({r['pred_place_prob']*100:.1f}% / 想定 {r['place_odds_est']:.1f}倍 / **EV {r['ev_place']:.2f}**)"
                )
            for r in win_recommendations:
                bet_lines.append(
                    f"🟠 **単勝** `[{r['horse_num']:>02}番]` **{r['horse_name']}** "
                    f"(単勝 {r['odds']:.1f}倍 / 複勝圏 {r['pred_place_prob']*100:.1f}%)"
                )
            fields.append({
                "name": "🎯 AI厳選推奨買い目",
                "value": "\n".join(bet_lines),
                "inline": False,
            })
        else:
            fields.append({
                "name": "🎯 AI厳選推奨買い目",
                "value": "⏸️ **見送り**: 期待値条件（EV≧1.1, 確率≧35%）を満たす馬がいません",
                "inline": False,
            })

        # 2. AI予測 上位5頭データ（スマホでも折り返さないクリーン形式）
        top_lines = ["```text"]
        top_lines.append("順 枠 番 馬名               単勝  複勝率    EV")
        top_lines.append("────────────────────────────────────────")
        for r in top_entries[:5]:
            # 全角対応の簡易パディング
            name = r["horse_name"]
            padded_name = name[:8].ljust(9, " ") if len(name) <= 8 else name[:7] + "…"
            top_lines.append(
                f"{r['pred_rank']:>2}  {r['bracket_num']:>1} {r['horse_num']:>2} "
                f"{padded_name:<10} {r['odds']:>5.1f} {r['pred_place_prob']*100:>6.1f}% {r['ev_place']:>5.2f}"
            )
        top_lines.append("```")

        fields.append({
            "name": "📊 AI予測 上位5頭",
            "value": "\n".join(top_lines),
            "inline": False,
        })

        payload = {
            "username": "keiba_ai_predictor",
            "embeds": [
                {
                    "title": f"🏇 {race_title}",
                    "url": netkeiba_url,
                    "description": conditions,
                    "color": embed_color,
                    "fields": fields,
                    "footer": {
                        "text": f"Race ID: {race_id} ｜ netkeiba出馬表連携"
                    }
                }
            ]
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                logger.info(f"Discordへの推論レポート送信に成功しました (Race ID: {race_id})")
                return True
            else:
                logger.error(f"Discord通知失敗: HTTP {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Discord通知送信中に例外が発生しました: {e}")
            return False