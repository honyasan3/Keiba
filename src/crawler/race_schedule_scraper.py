"""当日・指定日の開催レース一覧抽出モジュール（全ラウンド・特定R指定対応版）"""
import datetime
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import requests
from src.common.logger import setup_logger

logger = setup_logger("schedule_scraper")


class RaceScheduleScraper:
    """netkeibaの開催日程・レース一覧からレースIDを抽出するクラス"""

    def __init__(self, config=None) -> None:
        self.config = config

    @classmethod
    def get_race_ids_by_rounds(
        cls, date_str: Optional[str] = None, target_rounds: Optional[List[int]] = None
    ) -> List[Dict[str, str]]:
        """
        指定日・指定ラウンドのJRAレースID一覧を取得
        :param date_str: YYYYMMDD形式（例: 20250223）。Noneの場合は本日
        :param target_rounds: 対象とするレース番号のリスト（例: [11] や [9, 10, 11, 12]）。Noneの場合は全レース(1〜12R)

        db.netkeiba.com/race/list/{日付}/ はレース結果確定後のアーカイブであり、未来（開催前）の
        日付には常に0件を返す。race.netkeiba.com/top/race_list.html の当日一覧はJSが非同期で
        race_list_sub.html を呼んで描画するため静的HTMLには出走馬・レースリンクが存在しない。
        そのため、そのJSが実際に叩いているエンドポイント race_list_sub.html を直接呼び出す
        （kaisai_date指定のみで動作し、開催前・開催後どちらの日付でも一覧を取得できることを確認済み）。
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        kaisai_date = date_str or datetime.datetime.now().strftime("%Y%m%d")
        target_url = "https://race.netkeiba.com/top/race_list_sub.html"

        logger.info(f"開催スケジュールを取得中: {target_url}?kaisai_date={kaisai_date}")
        try:
            res = requests.get(target_url, headers=headers, params={"kaisai_date": kaisai_date}, timeout=10)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception as e:
            logger.error(f"スケジュール取得エラー: {e}")
            return []

        race_list = []
        seen_races = set()

        for item in soup.find_all("li", class_=re.compile(r"RaceList_DataItem")):
            link = item.find("a", href=re.compile(r"race_id=\d{12}"))
            if not link:
                continue
            match = re.search(r"race_id=(\d{12})", link["href"])
            if not match:
                continue

            race_id = match.group(1)
            if race_id in seen_races:
                continue

            venue_code = race_id[4:6]
            race_round_int = int(race_id[-2:])

            # JRA競馬場（01〜10）かつ 指定ラウンドに合致するか
            if venue_code not in [f"{i:02d}" for i in range(1, 11)]:
                continue
            if target_rounds is not None and race_round_int not in target_rounds:
                continue

            seen_races.add(race_id)
            title_tag = item.find("span", class_="ItemTitle")
            race_name = title_tag.get_text(strip=True) if title_tag else f"{race_round_int}R"
            race_list.append({
                "race_id": race_id,
                "race_name": race_name,
                "round": f"{race_round_int}R",
            })

        # レースID順（競馬場・レース番号順）にソート
        race_list.sort(key=lambda x: x["race_id"])

        logger.info(f"対象レースID抽出完了: {len(race_list)} 件検出")
        return race_list

    def fetch_daily_race_ids(self, target_date: str) -> List[str]:
        """
        指定日 (YYYY-MM-DD) の全JRAレースIDリストを返す互換メソッド
        """
        date_str = target_date.replace("-", "").strip() if target_date else None
        race_dicts = self.get_race_ids_by_rounds(date_str=date_str, target_rounds=None)
        return [item["race_id"] for item in race_dicts]