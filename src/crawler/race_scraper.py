"""競馬データのURL構築および取得実行モジュール"""
import re
from typing import List
from bs4 import BeautifulSoup
from config.config_loader import CrawlerConfig
from src.crawler.base_scraper import BaseScraper
from src.common.logger import setup_logger

logger = setup_logger("race_scraper")


class RaceScraper(BaseScraper):
    """レース関連データの取得を専門に行うスクレイパークラス"""

    # JRA（中央競馬）の競馬場コード: 01(札幌) ~ 10(小倉)
    JRA_VENUE_CODES = {f"{i:02d}" for i in range(1, 11)}

    def __init__(self, config: CrawlerConfig, base_url: str = "https://db.netkeiba.com") -> None:
        super().__init__(config)
        self.base_url = base_url.rstrip("/")

    def fetch_race_ids_by_date(self, date_str: str, jra_only: bool = True) -> List[str]:
        """指定された日付（YYYYMMDD形式）の開催レース一覧ページから実在するレースIDをすべて抽出"""
        target_url = f"{self.base_url}/race/list/{date_str}/"
        logger.info(f"開催レース一覧の取得中: {date_str} ({target_url})")

        try:
            html = self.fetch_page(target_url, use_cache=True)
            soup = BeautifulSoup(html, "html.parser")

            race_ids = set()
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                match = re.search(r"/race/(\d{12})", href)
                if match:
                    r_id = match.group(1)
                    venue_code = r_id[4:6]
                    if jra_only and venue_code not in self.JRA_VENUE_CODES:
                        continue
                    race_ids.add(r_id)

            id_list = sorted(list(race_ids))
            logger.info(f"{date_str} のJRAレースID取得数: {len(id_list)} 件")
            return id_list
        except Exception as e:
            logger.warning(f"{date_str} のレース一覧取得中にエラーが発生しました: {e}")
            return []

    def fetch_race_result(self, race_id: str, use_cache: bool = True) -> str:
        """指定されたレースIDの結果ページHTMLを取得"""
        target_url = f"{self.base_url}/race/{race_id}/"
        logger.info(f"レース結果ページの取得開始 (Race ID: {race_id})")
        return self.fetch_page(target_url, use_cache=use_cache)

    def fetch_race_card(self, race_id: str, use_cache: bool = True) -> str:
            """指定されたレースIDの出馬情報HTMLを取得（過去レースはdb側、未来レースはrace側から取得）"""
            # db.netkeiba.com（過去・キャッシュ済みレース）を優先
            db_url = f"{self.base_url}/race/{race_id}/"
            try:
                return self.fetch_page(db_url, use_cache=use_cache)
            except Exception:
                # 取得できない場合は当日の出馬表ページを取得
                shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
                logger.info(f"当日の出馬表URLを取得中: {shutuba_url}")
                return self.fetch_page(shutuba_url, use_cache=use_cache)