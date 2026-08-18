"""生HTMLからのデータ抽出・パース処理モジュール"""
import re
from typing import Any, Dict, List
from bs4 import BeautifulSoup
from src.common.exceptions import ParseError
from src.common.logger import setup_logger

logger = setup_logger("html_parser")


class RaceHtmlParser:
    """netkeibaのレース結果HTMLを詳細に解析するパーサークラス"""

    @staticmethod
    def parse_race_result(html_content: str, race_id: str) -> Dict[str, Any]:
        if not html_content:
            raise ParseError("解析対象のHTMLコンテンツが空です。")

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # 1. レース名・ラウンド・条件の取得
            intro_data = soup.find("div", class_="data_intro")
            race_title = ""
            race_round = None
            course_type, distance, weather, track_cond, race_date = None, None, None, None, None

            if intro_data:
                h1_tag = intro_data.find("h1")
                if h1_tag:
                    race_title = h1_tag.get_text(strip=True)

                dt_tag = intro_data.find("dt")
                if dt_tag:
                    r_match = re.search(r"(\d+)\s*R", dt_tag.get_text())
                    if r_match:
                        race_round = int(r_match.group(1))

                span_tag = intro_data.find("span")
                if span_tag:
                    span_text = span_tag.get_text(strip=True)
                    if "芝" in span_text:
                        course_type = "芝"
                    elif "ダ" in span_text:
                        course_type = "ダート"
                    elif "障" in span_text:
                        course_type = "障害"

                    dist_match = re.search(r"(\d{3,4})m", span_text)
                    if dist_match:
                        distance = int(dist_match.group(1))

                    w_match = re.search(r"天候\s*:\s*([^\s/]+)", span_text)
                    if w_match:
                        weather = w_match.group(1).strip()

                    cond_match = re.search(r"(?:芝|ダート|障)\s*:\s*([^\s/]+)", span_text)
                    if cond_match:
                        track_cond = cond_match.group(1).strip()

                small_txt = intro_data.find("p", class_="smalltxt")
                if small_txt:
                    d_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", small_txt.get_text())
                    if d_match:
                        race_date = f"{d_match.group(1)}-{int(d_match.group(2)):02d}-{int(d_match.group(3)):02d}"

            # 2. 着順テーブルのパース
            results: List[Dict[str, Any]] = []
            table = soup.find("table", class_="race_table_01")
            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:
                    cols = row.find_all(["td", "th"])
                    if len(cols) < 19:
                        continue

                    # 馬ID
                    h_link = cols[3].find("a")
                    h_id = ""
                    if h_link and "href" in h_link.attrs:
                        h_id_match = re.search(r"/horse/(\w+)", h_link["href"])
                        if h_id_match:
                            h_id = h_id_match.group(1)

                    result_record = {
                        "rank_raw": cols[0].get_text(strip=True),
                        "bracket_num_raw": cols[1].get_text(strip=True),
                        "horse_num_raw": cols[2].get_text(strip=True),
                        "horse_name": cols[3].get_text(strip=True),
                        "horse_id": h_id,
                        "sex_age_raw": cols[4].get_text(strip=True),
                        "jockey_weight_raw": cols[5].get_text(strip=True),
                        "jockey_name": cols[6].get_text(strip=True),
                        "finish_time_raw": cols[7].get_text(strip=True),
                        "margin_raw": cols[8].get_text(strip=True),
                        # 9~13列目はタイム指数等の有料情報（**）のためスキップ
                        "passage_order_raw": cols[14].get_text(strip=True),  # 14: 通過 (例: 11-8)
                        "last_3f_raw": cols[15].get_text(strip=True),        # 15: 上り (例: 36.4)
                        "odds_raw": cols[16].get_text(strip=True),           # 16: 単勝 (例: 2.4)
                        "popularity_raw": cols[17].get_text(strip=True),     # 17: 人気 (例: 1)
                        "horse_weight_raw": cols[18].get_text(strip=True),   # 18: 馬体重 (例: 534(+2))
                    }
                    results.append(result_record)

            return {
                "race_id": race_id,
                "race_title": race_title,
                "race_date": race_date,
                "race_round": race_round,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track_cond,
                "results": results,
            }

        except Exception as e:
            logger.error(f"パース失敗 (Race ID: {race_id}): {e}")
            raise ParseError(f"パース失敗: {e}") from e