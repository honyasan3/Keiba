"""レース結果自動取得および収支集計・レポートモジュール"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.common.logger import setup_logger

logger = setup_logger("settlement_reporter")


class SettlementReporter:
    """確定レース結果のスクレイピングと推奨買い目の収支集計を行うクラス"""

    def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_race_payouts(self, race_id: str) -> Dict[str, Any]:
        """netkeibaのレース結果HTMLから確定着順と払戻金（単勝・複勝）を取得"""
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            res.encoding = "EUC-JP"
            soup = BeautifulSoup(res.text, "html.parser")

            # 確定着順 (1〜3着の馬番)
            top3_horses = []
            result_table = soup.find("table", class_="RaceTable01")
            if result_table:
                for row in result_table.find_all("tr")[1:4]:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        try:
                            h_num = int(cells[2].get_text(strip=True))
                            top3_horses.append(h_num)
                        except ValueError:
                            pass

            # 払戻金テーブル解析
            payouts = {"win": {}, "place": {}}
            payout_tables = soup.find_all("table", class_="Payout_Detail_Table")
            for table in payout_tables:
                for row in table.find_all("tr"):
                    header = row.find("th")
                    if not header:
                        continue
                    htext = header.get_text(strip=True)
                    cells = row.find_all("td")
                    if not cells:
                        continue

                    # 単勝
                    if "単勝" in htext:
                        horse_nums = [int(x) for x in cells[0].get_text(strip=True).split() if x.isdigit()]
                        pays = [int(x.replace(",", "")) for x in cells[1].get_text(strip=True).split() if x.replace(",", "").isdigit()]
                        for h, p in zip(horse_nums, pays):
                            payouts["win"][h] = p

                    # 複勝
                    elif "複勝" in htext:
                        horse_nums = [int(x) for x in cells[0].get_text(strip=True).split() if x.isdigit()]
                        pays = [int(x.replace(",", "")) for x in cells[1].get_text(strip=True).split() if x.replace(",", "").isdigit()]
                        for h, p in zip(horse_nums, pays):
                            payouts["place"][h] = p

            return {
                "race_id": race_id,
                "top3": top3_horses,
                "payouts": payouts,
            }
        except Exception as e:
            logger.error(f"払戻金データの取得に失敗しました (Race ID: {race_id}): {e}")
            return {"race_id": race_id, "top3": [], "payouts": {"win": {}, "place": {}}}

    def calculate_settlement(
        self,
        predicted_bets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        推奨買い目リストと確定結果を突き合わせて収支を集計
        predicted_bets: [{'race_id', 'race_title', 'bet_type', 'horse_num', 'horse_name', 'bet_amount'}]
        """
        total_invest = 0
        total_return = 0
        hit_count = 0
        bet_count = len(predicted_bets)
        details = []

        # レースごとに払戻データをキャッシュ
        payout_cache = {}

        for bet in predicted_bets:
            rid = bet["race_id"]
            if rid not in payout_cache:
                payout_cache[rid] = self.fetch_race_payouts(rid)
            res = payout_cache[rid]

            btype = bet.get("bet_type", "place")
            hnum = bet["horse_num"]
            invest = bet.get("bet_amount", 100)
            total_invest += invest

            is_hit = False
            payout_amount = 0

            if btype == "place" and hnum in res["payouts"]["place"]:
                # 複勝払戻金（100円あたりの払戻金額）
                unit_pay = res["payouts"]["place"][hnum]
                payout_amount = int(invest * (unit_pay / 100))
                is_hit = True
            elif btype == "win" and hnum in res["payouts"]["win"]:
                unit_pay = res["payouts"]["win"][hnum]
                payout_amount = int(invest * (unit_pay / 100))
                is_hit = True

            if is_hit:
                hit_count += 1
                total_return += payout_amount

            details.append({
                "race_title": bet.get("race_title", rid),
                "bet_type": "複勝" if btype == "place" else "単勝",
                "horse": f"[{hnum}番] {bet.get('horse_name', '')}",
                "invest": invest,
                "payout": payout_amount,
                "is_hit": is_hit,
            })

        roi = (total_return / total_invest * 100) if total_invest > 0 else 0.0
        profit = total_return - total_invest

        return {
            "bet_count": bet_count,
            "hit_count": hit_count,
            "hit_rate": (hit_count / bet_count * 100) if bet_count > 0 else 0.0,
            "total_invest": total_invest,
            "total_return": total_return,
            "profit": profit,
            "roi": roi,
            "details": details,
        }