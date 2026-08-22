"""競馬データのURL構築および取得実行モジュール"""
import random
import re
import time
from typing import Dict, List
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
            """指定されたレースIDの出馬情報HTMLを取得

            db.netkeiba.com の過去レースアーカイブは、未来（まだ結果が確定していない）レースIDに対して
            HTTPエラーを返さず出走馬データを含まない空ページを返すため、db側を優先すると本番の
            レース前予想が常に「出走馬0頭」で失敗する。race.netkeiba.com/race/shutuba.html は
            過去・未来どちらのレースIDでも出走馬データを正しく返すため、常にこちらを使用する。
            """
            shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            return self.fetch_page(shutuba_url, use_cache=use_cache)

    def fetch_live_odds(self, race_id: str) -> Dict[int, Dict[str, float]]:
        """単勝オッズ・複勝オッズ（実測レンジ）・人気順を取得する。

        出馬表(shutuba.html)には静的にオッズが埋め込まれておらず、JS側が
        race.netkeiba.com/api/api_get_jra_odds.html を非同期に呼んで反映している。
        JRAはレース直前までオッズを公表しないため、公表前は status != "result" となり、
        その場合は空dictを返す（呼び出し側は「オッズ未公表」として扱い、架空の値で
        代替してはならない）。

        レスポンスの data.odds には券種別のオッズが入っており、"1"=単勝、"2"=複勝。
        複勝は的中時の配当がレース結果確定まで確定しないため [最低倍率, 最高倍率, 人気順] の
        レンジで返る（例: ["1.2", "2.0", "2"]）。従来は単勝オッズからの近似式
        (odds ** 0.45) で複勝オッズを推定していたが、実測レンジが取得できる場合はそちらを
        優先する。
        """
        url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
        try:
            res = self.session.get(url, params={"race_id": race_id, "type": 1}, timeout=10)
            res.raise_for_status()
            payload = res.json()
        except Exception as e:
            logger.warning(f"オッズAPIの取得に失敗しました (Race ID: {race_id}): {e}")
            return {}
        finally:
            # fetch_page（HTML取得）と同じくランダム待機を挟む。run_daily_predict.py --rounds all
            # のように多数のレースを連続処理する際、このAPIだけ待機なしで連打しないようにするため。
            sleep_time = round(random.uniform(self.config.min_delay, self.config.max_delay), 2)
            logger.debug(f"オッズAPIアクセス間隔待機: {sleep_time} 秒")
            time.sleep(sleep_time)

        if payload.get("status") != "result":
            logger.info(f"オッズ未公表です (Race ID: {race_id}, status: {payload.get('status')})")
            return {}

        odds_data = payload.get("data", {}).get("odds", {})
        win_odds = odds_data.get("1", {})
        place_odds = odds_data.get("2", {})

        result: Dict[int, Dict[str, float]] = {}
        for horse_num_str, values in win_odds.items():
            try:
                horse_num = int(horse_num_str)
                entry = {
                    "odds": float(values[0]),
                    "popularity": int(values[2]),
                }
            except (ValueError, IndexError, TypeError):
                continue

            place_values = place_odds.get(horse_num_str)
            if place_values:
                try:
                    entry["place_odds_min"] = float(place_values[0])
                    entry["place_odds_max"] = float(place_values[1])
                except (ValueError, IndexError, TypeError):
                    pass

            result[horse_num] = entry
        logger.info(f"オッズ取得完了 (Race ID: {race_id}, {len(result)}頭)")
        return result