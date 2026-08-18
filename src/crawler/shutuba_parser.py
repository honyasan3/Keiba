"""出馬表（発走前レース）および検証用レースのHTMLパースモジュール"""
import re
from typing import Any, Dict, List
from bs4 import BeautifulSoup
from src.common.exceptions import ParseError
from src.common.logger import setup_logger

logger = setup_logger("shutuba_parser")


class ShutubaHtmlParser:
    """出馬表およびレース結果HTMLから発走前情報を抽出するクラス"""

    @staticmethod
    def parse_shutuba_card(html_content: str, race_id: str) -> Dict[str, Any]:
        if not html_content:
            raise ParseError("解析対象のHTMLが空です。")

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # 1. レース基本情報の取得
            race_title = ""
            race_round = None
            course_type, distance, weather, track_cond, race_date = "芝", 1600, "晴", "良", None

            # パターンA: db.netkeiba.com 形式
            intro_data = soup.find("div", class_="data_intro")
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

            # パターンB: race.netkeiba.com 形式
            race_name_box = soup.find("div", class_="RaceName")
            if race_name_box:
                race_title = race_name_box.get_text(strip=True)
            r_num_box = soup.find("span", class_="RaceNum")
            if r_num_box:
                r_m = re.search(r"(\d+)", r_num_box.get_text())
                if r_m:
                    race_round = int(r_m.group(1))

            # 2. 出走馬一覧の取得
            entries: List[Dict[str, Any]] = []

            # 形式1: db.netkeiba.com の結果テーブル（着順テーブルから発走前情報のみ抽出）
            table_db = soup.find("table", class_="race_table_01")
            if table_db:
                rows = table_db.find_all("tr")
                for row in rows[1:]:
                    cols = row.find_all(["td", "th"])
                    if len(cols) < 19:
                        continue

                    h_link = cols[3].find("a")
                    h_id = ""
                    if h_link and "href" in h_link.attrs:
                        h_id_m = re.search(r"/horse/(\w+)", h_link["href"])
                        if h_id_m:
                            h_id = h_id_m.group(1)

                    entries.append({
                        "bracket_num_raw": cols[1].get_text(strip=True),
                        "horse_num_raw": cols[2].get_text(strip=True),
                        "horse_name": cols[3].get_text(strip=True),
                        "horse_id": h_id,
                        "sex_age_raw": cols[4].get_text(strip=True),
                        "jockey_weight_raw": cols[5].get_text(strip=True),
                        "jockey_name": cols[6].get_text(strip=True),
                        "odds_raw": cols[16].get_text(strip=True),
                        "popularity_raw": cols[17].get_text(strip=True),
                        "horse_weight_raw": cols[18].get_text(strip=True),
                    })

            # 形式2: race.netkeiba.com の出馬表テーブル (Shutuba_Table / HorseList)
            if not entries:
                shutuba_rows = soup.find_all("tr", class_=re.compile(r"HorseList"))
                for row in shutuba_rows:
                    b_num = row.find("td", class_=re.compile(r"Waku\d"))
                    h_num = row.find("td", class_=re.compile(r"Umaban\d"))
                    h_link = row.find("a", href=re.compile(r"/horse/"))

                    if not (h_num and h_link):
                        continue

                    h_id_m = re.search(r"/horse/(\w+)", h_link["href"])
                    h_id = h_id_m.group(1) if h_id_m else ""

                    # 性齢・斤量・騎手
                    sex_age_td = row.find("td", class_="Barei")
                    j_weight_td = row.find("td", class_="Txt_C")
                    jockey_td = row.find("td", class_="Jockey")
                    odds_td = row.find("span", class_="Popular")
                    weight_td = row.find("td", class_="Weight")

                    entries.append({
                        "bracket_num_raw": b_num.get_text(strip=True) if b_num else "",
                        "horse_num_raw": h_num.get_text(strip=True),
                        "horse_name": h_link.get_text(strip=True),
                        "horse_id": h_id,
                        "sex_age_raw": sex_age_td.get_text(strip=True) if sex_age_td else "",
                        "jockey_weight_raw": j_weight_td.get_text(strip=True) if j_weight_td else "",
                        "jockey_name": jockey_td.get_text(strip=True) if jockey_td else "",
                        "odds_raw": odds_td.get_text(strip=True) if odds_td else "10.0",
                        "popularity_raw": "1",
                        "horse_weight_raw": weight_td.get_text(strip=True) if weight_td else "",
                    })

            logger.info(f"出走馬情報の抽出完了: {len(entries)} 頭")
            return {
                "race_id": race_id,
                "race_title": race_title,
                "race_date": race_date,
                "race_round": race_round,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track_cond,
                "entries": entries,
            }

        except Exception as e:
            logger.error(f"出馬表パース失敗 (Race ID: {race_id}): {e}")
            raise ParseError(f"出馬表パース失敗: {e}") from e