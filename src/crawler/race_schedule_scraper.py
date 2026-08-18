"""当日・指定日の開催レース一覧抽出モジュール（全ラウンド・特定R指定対応版）"""
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import requests
from src.common.logger import setup_logger

logger = setup_logger("schedule_scraper")


class RaceScheduleScraper:
    """netkeibaの開催日程・レース一覧からレースIDを抽出するクラス"""

    @classmethod
    def get_race_ids_by_rounds(
        cls, date_str: Optional[str] = None, target_rounds: Optional[List[int]] = None
    ) -> List[Dict[str, str]]:
        """
        指定日・指定ラウンドのJRAレースID一覧を取得
        :param date_str: YYYYMMDD形式（例: 20250223）。Noneの場合は本日
        :param target_rounds: 対象とするレース番号のリスト（例: [11] や [9, 10, 11, 12]）。Noneの場合は全レース(1〜12R)
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        if date_str:
            target_url = f"https://db.netkeiba.com/race/list/{date_str}/"
        else:
            target_url = "https://race.netkeiba.com/top/"

        logger.info(f"開催スケジュールを取得中: {target_url}")
        try:
            res = requests.get(target_url, headers=headers, timeout=10)
            res.encoding = "euc-jp"
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            logger.error(f"スケジュール取得エラー: {e}")
            return []

        race_list = []
        seen_races = set()

        # ページ内のすべてのレースリンクから12桁のJRAレースIDを探索
        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = re.search(r"/(?:race|shutuba\.html\?race_id=)/?(\d{12})", href)
            if not match:
                match = re.search(r"race_id=(\d{12})", href)

            if match:
                race_id = match.group(1)
                venue_code = race_id[4:6]
                race_round_int = int(race_id[-2:])

                # JRA競馬場（01〜10）かつ 指定ラウンドに合致するか
                if venue_code in [f"{i:02d}" for i in range(1, 11)]:
                    if target_rounds is None or race_round_int in target_rounds:
                        if race_id not in seen_races:
                            seen_races.add(race_id)
                            race_name = link.get_text(strip=True) or f"{race_round_int}R"
                            race_list.append({
                                "race_id": race_id,
                                "race_name": race_name,
                                "round": f"{race_round_int}R",
                            })

        # レースID順（競馬場・レース番号順）にソート
        race_list.sort(key=lambda x: x["race_id"])

        logger.info(f"対象レースID抽出完了: {len(race_list)} 件検出")
        return race_list