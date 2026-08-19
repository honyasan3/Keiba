"""Discord Webhook 通知モジュール（ケリー推奨額・ワイド推奨・シミュ勝率対応版）"""
from typing import Any, Dict, List, Optional
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
        wide_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if not self.enabled or not self.webhook_url or "YOUR_DISCORD" in self.webhook_url:
            logger.info("Discord通知は無効化されているか、Webhook URLが未設定です。")
            return False

        wide_recommendations = wide_recommendations or []
        race_id = race_info.get("race_id", "")
        race_title = race_info.get("race_title", "レース予想")
        netkeiba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

        # 買い目の有無でアクセントカラーを変更 (推奨あり: 翠緑 0x2ECC71, 見送り: スレートグレー 0x95A5A6)
        has_bet = bool(place_recommendations or win_recommendations or wide_recommendations)
        embed_color = 0x2ECC71 if has_bet else 0x95A5A6

        # コース条件ヘッダー
        conditions = (
            f"**{race_info.get('course_type', '')} {race_info.get('distance', '')}m** ｜ "
            f"天候: `{race_info.get('weather', '-')}` ｜ 馬場: `{race_info.get('track_condition', '-')}`"
        )

        fields = []

        # 1. 厳選買い目セクション（ケリー賭け金・ワイド対応）
        if has_bet:
            bet_lines = []
            for r in place_recommendations:
                kelly_str = f" 👉 **賭金: {r.get('kelly_bet_place', 0)}円**" if r.get("kelly_bet_place") else ""
                prob = r.get("ensemble_place_prob", r.get("pred_place_prob", 0)) * 100
                bet_lines.append(
                    f"🟢 **複勝** `[{r['horse_num']:>02}番]` **{r['horse_name']}**\n"
                    f"  └ 複勝率 `{prob:.1f}%` / 想定 `{r['place_odds_est']:.1f}倍` / **EV {r['ev_place']:.2f}**{kelly_str}"
                )
            for r in win_recommendations:
                sim_win = r.get("sim_win_prob", 0) * 100
                prob = r.get("ensemble_place_prob", r.get("pred_place_prob", 0)) * 100
                bet_lines.append(
                    f"🟠 **単勝穴** `[{r['horse_num']:>02}番]` **{r['horse_name']}**\n"
                    f"  └ 単勝 `{r['odds']:.1f}倍` / 勝率 `{sim_win:.1f}%` / 複勝率 `{prob:.1f}%`"
                )
            for w in wide_recommendations:
                bet_lines.append(
                    f"🔵 **ワイド** `[{w['pair']}]` **{w['names']}**\n"
                    f"  └ 的中率 `{w['prob']*100:.1f}%` / 想定 `{w['est_odds']:.1f}倍` / **EV {w['ev']:.2f}**"
                )

            fields.append({
                "name": "🎯 AI厳選推奨買い目（資金傾斜付き）",
                "value": "\n".join(bet_lines),
                "inline": False,
            })
        else:
            fields.append({
                "name": "🎯 AI厳選推奨買い目",
                "value": "⏸️ **見送り (KEN)**: 基準（EV≧1.1, 複勝率≧40%）を満たす馬がいません",
                "inline": False,
            })

        # 2. AI予測 上位5頭サマリーテーブル
        top_lines = ["```text"]
        top_lines.append("順 枠 番 馬名            単勝  勝率  複勝率   EV")
        top_lines.append("─────────────────────────────────────────────")
        for r in top_entries[:5]:
            name = r["horse_name"]
            padded_name = name[:7].ljust(8, " ") if len(name) <= 7 else name[:6] + "…"
            sim_w = r.get("sim_win_prob", 0) * 100
            ens_p = r.get("ensemble_place_prob", r.get("pred_place_prob", 0)) * 100
            top_lines.append(
                f"{r['pred_rank']:>2}  {r['bracket_num']:>1} {r['horse_num']:>2} "
                f"{padded_name:<8} {r['odds']:>5.1f} {sim_w:>4.1f}% {ens_p:>5.1f}% {r['ev_place']:>5.2f}"
            )
        top_lines.append("```")

        fields.append({
            "name": "📊 AI × シミュレーション総合 上位5頭",
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
            # プラス収支: ゴールド 0xF1C40F / マイナス: 深紅 0xE74C3C
            embed_color = 0xF1C40F if profit >= 0 else 0xE74C3C

            lines = ["```text"]
            lines.append("レース             券種  対象馬          投資    払戻   結果")
            lines.append("──────────────────────────────────────────────────────────")
            for d in settlement["details"]:
                rtitle = d["race_title"][:8].ljust(9, " ")
                btype = d["bet_type"].ljust(3, " ")
                horse = d["horse"][:9].ljust(10, " ")
                res_mark = "🎯 的中" if d["is_hit"] else "✕ はずれ"
                lines.append(
                    f"{rtitle} {btype} {horse} {d['invest']:>5}円 {d['payout']:>6}円  {res_mark}"
                )
            lines.append("```")

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
                                "value": "\n".join(lines),
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