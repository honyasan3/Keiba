"""Discord Webhook 通知モジュール（視認性特化・モダンダッシュボードUI版）"""
from typing import Any, Dict, List, Optional
import requests
from src.common.logger import setup_logger

logger = setup_logger("discord_notifier")


class DiscordNotifier:
    """Discord Webhook経由でレース推論結果をリッチ通知するクラス"""

    # JRA公式枠色アイコン
    BRACKET_ICONS = {
        1: "⬜", 2: "⬛", 3: "🟥", 4: "🟦",
        5: "🟨", 6: "🟩", 7: "🟧", 8: "🌸"
    }

    def __init__(self, webhook_url: str, enabled: bool = True) -> None:
        self.webhook_url = webhook_url
        self.enabled = enabled

    def send_prediction_report(
        self,
        race_info: Dict[str, Any],
        top_entries: List[Dict[str, Any]],
        place_recommendations: List[Dict[str, Any]],
        win_recommendations: List[Dict[str, Any]],
        wide_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if not self.enabled or not self.webhook_url or "YOUR_DISCORD" in self.webhook_url:
            logger.info("Discord通知は無効化されているか、Webhook URLが未設定です。")
            return False

        wide_recommendations = wide_recommendations or []
        race_id = race_info.get("race_id", "")
        race_title = race_info.get("race_title", "レース予想")
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

        has_bet = bool(place_recommendations or win_recommendations or wide_recommendations)
        embed_color = 0xF1C40F if win_recommendations else (0x2ECC71 if has_bet else 0x95A5A6)

        header_desc = (
            f"📍 **{race_info.get('course_type', '')} {race_info.get('distance', '')}m** ｜ "
            f"天候: `{race_info.get('weather', '-')}` ｜ 馬場: `{race_info.get('track_condition', '-')}`"
        )

        fields = []

        # 1. 厳選推奨買い目（引用ブロックでスマートに強調）
        if has_bet:
            bet_lines = []
            for r in win_recommendations:
                sim_win = r.get("sim_win_prob", 0) * 100
                bet_lines.append(
                    f"> 🟠 **単勝穴** `[{r['horse_num']:>02}]` **{r['horse_name']}**\n"
                    f">  └ 単 **{r['odds']:.1f}倍** ｜ 勝率 **{sim_win:.1f}%** ｜ 複勝率 {r.get('ensemble_place_prob', 0)*100:.1f}%"
                )
            for r in place_recommendations:
                kelly_str = f" ｜ 💰 **{r.get('kelly_bet_place', 0)}円**" if r.get("kelly_bet_place") else ""
                prob = r.get("ensemble_place_prob", r.get("pred_place_prob", 0)) * 100
                bet_lines.append(
                    f"> 🟢 **複勝** `[{r['horse_num']:>02}]` **{r['horse_name']}**\n"
                    f">  └ 複勝率 **{prob:.1f}%** ｜ 想定 **{r['place_odds_est']:.1f}倍** ｜ **EV {r['ev_place']:.2f}**{kelly_str}"
                )
            for w in wide_recommendations:
                bet_lines.append(
                    f"> 🔵 **ワイド** `[{w['pair']}]` **{w['names']}**\n"
                    f">  └ 的中 **{w['prob']*100:.1f}%** ｜ 想定 **{w['est_odds']:.1f}倍** ｜ **EV {w['ev']:.2f}**"
                )

            fields.append({
                "name": "🎯 厳選推奨買い目",
                "value": "\n".join(bet_lines),
                "inline": False,
            })
        else:
            fields.append({
                "name": "🎯 厳選推奨買い目",
                "value": "> ⏸️ **見送り (KEN)**: 基準を満たす妙味馬がいません",
                "inline": False,
            })

        # 2. AI予測 上位5頭（1行に凝縮してスキャンしやすく配置）
        rank_badges = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        top_lines = []
        for i, r in enumerate(top_entries[:5]):
            badge = rank_badges[i] if i < len(rank_badges) else f"`{i+1}`"
            waku_icon = self.BRACKET_ICONS.get(r.get("bracket_num"), "⬜")
            num = f"{r['horse_num']:>02}"
            name = r["horse_name"]
            odds = f"{r['odds']:.1f}倍"
            sim_w = f"{r.get('sim_win_prob', 0) * 100:.1f}%"
            ens_p = f"{r.get('ensemble_place_prob', r.get('pred_place_prob', 0)) * 100:.1f}%"
            ev = f"{r['ev_place']:.2f}"

            top_lines.append(
                f"{badge} {waku_icon}`{num}` **{name}**\n"
                f"  単 **{odds}** ｜ 勝 **{sim_w}** ｜ 複 **{ens_p}** ｜ EV `{ev}`"
            )

        fields.append({
            "name": "📊 AI総合予測 上位5頭",
            "value": "\n".join(top_lines),
            "inline": False,
        })

        payload = {
            "username": "keiba_ai_predictor",
            "embeds": [
                {
                    "title": f"🏇 {race_title}",
                    "url": netkeiba_url,
                    "description": header_desc,
                    "color": embed_color,
                    "fields": fields,
                    "footer": {
                        "text": f"Race ID: {race_id} ｜ 10,000回シミュレーション実施済"
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

    def send_settlement_report(self, settlement: Dict[str, Any], date_str: str) -> bool:
        """1日の確定収支レポートをDiscordへ通知"""
        if not self.enabled or not self.webhook_url or "YOUR_DISCORD" in self.webhook_url:
            return False

        profit = settlement["profit"]
        roi = settlement["roi"]
        embed_color = 0xF1C40F if profit >= 0 else 0xE74C3C

        lines = []
        for d in settlement["details"]:
            res_mark = "🎯 **的中**" if d["is_hit"] else "❌ はずれ"
            lines.append(
                f"> • `{d['race_title'][:8]}` ｜ **{d['bet_type']}** {d['horse']}\n"
                f">   投 `{d['invest']:,}円` → 戻 **`{d['payout']:,}円`** ｜ {res_mark}"
            )

        summary_field = (
            f"**総投資額**: `{settlement['total_invest']:,} 円`\n"
            f"**総払戻額**: `{settlement['total_return']:,} 円`\n"
            f"**本日収支**: `{'＋' if profit >= 0 else ''}{profit:,} 円`\n"
            f"**的中率**: `{settlement['hit_rate']:.1f}%` ({settlement['hit_count']}/{settlement['bet_count']})\n"
            f"**回収率 (ROI)**: **`{roi:.1f}%`**"
        )

        payload = {
            "username": "keiba_ai_predictor",
            "embeds": [
                {
                    "title": f"💰 【本日のAI運用 確定収支レポート】 ({date_str})",
                    "description": summary_field,
                    "color": embed_color,
                    "fields": [
                        {
                            "name": "📋 推奨買い目 精算明細",
                            "value": "\n".join(lines) if lines else "購入対象レースなし",
                            "inline": False,
                        }
                    ],
                    "footer": {"text": "Keiba AI 自動精算システム"},
                }
            ],
        }

        try:
            res = requests.post(self.webhook_url, json=payload, timeout=10)
            return res.status_code == 204
        except Exception as e:
            logger.error(f"収支レポート送信失敗: {e}")
            return False