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

            # パターンB: race.netkeiba.com 形式（RaceNameは<h1>、それ以外にタグ制約は付けない）
            race_name_box = soup.find(class_="RaceName")
            if race_name_box:
                race_title = race_name_box.get_text(strip=True)
            r_num_box = soup.find("span", class_="RaceNum")
            if r_num_box:
                r_m = re.search(r"(\d+)", r_num_box.get_text())
                if r_m:
                    race_round = int(r_m.group(1))

            race_data_box = soup.find(class_="RaceData01")
            if race_data_box:
                data_text = race_data_box.get_text(" ", strip=True)
                if "芝" in data_text:
                    course_type = "芝"
                elif "ダ" in data_text:
                    course_type = "ダート"
                elif "障" in data_text:
                    course_type = "障害"

                dist_match = re.search(r"(\d{3,4})m", data_text)
                if dist_match:
                    distance = int(dist_match.group(1))

                w_match = re.search(r"天候:?\s*([^\s/]+)", data_text)
                if w_match:
                    weather = w_match.group(1).strip()

                cond_match = re.search(r"馬場:?\s*([^\s/]+)", data_text)
                if cond_match:
                    track_cond = cond_match.group(1).strip()

            if race_date is None:
                # metaタグの説明文（例: "2026年8月16日 札幌1R ３歳未勝利の出馬表です。"）から開催日を取得
                meta_tag = soup.find("meta", attrs={"name": "description"})
                meta_text = meta_tag.get("content", "") if meta_tag else ""
                d_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", meta_text)
                if d_match:
                    race_date = f"{d_match.group(1)}-{int(d_match.group(2)):02d}-{int(d_match.group(3)):02d}"

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
            # 注意: 枠順抽選（通常レース数日前）が確定するまでは Waku/Umaban のclassに数字が付かず
            # （例: class="Umaban Txt_C"）、中身も空になる。<tr id="tr_N">のNは登録順の内部IDであり
            # 実際の馬番とは無関係なので、馬番の代わりに使ってはならない（誤った馬番で表示してしまう）。
            # 枠順確定後は class="Umaban3 Txt_C" のように数字が付き、セル内テキストにも実値が入るため、
            # そちらを正としてパースする。取得できない場合はまだ枠順未確定として当該馬をスキップする
            # （オッズも同様にレース直前まで存在しないため、これは実運用上正しい挙動）。
            if not entries:
                shutuba_rows = soup.find_all("tr", class_=re.compile(r"HorseList"))
                for row in shutuba_rows:
                    b_num_td = row.find("td", class_=re.compile(r"^Waku\d"))
                    h_num_td = row.find("td", class_=re.compile(r"^Umaban\d"))
                    h_link = row.find("a", href=re.compile(r"/horse/"))

                    if not (h_num_td and h_link):
                        continue

                    h_num_text = h_num_td.get_text(strip=True)
                    if not h_num_text.isdigit():
                        continue

                    h_id_m = re.search(r"/horse/(\w+)", h_link["href"])
                    h_id = h_id_m.group(1) if h_id_m else ""

                    # 性齢・斤量・騎手（これらは静的HTMLに実値が入っている）
                    sex_age_td = row.find("td", class_="Barei")
                    # class="Txt_C"単独のセルが斤量。Waku/Umaban等も"Txt_C"を含むため完全一致で絞り込む
                    j_weight_td = row.find("td", class_=lambda c: c == ["Txt_C"])
                    jockey_td = row.find("td", class_="Jockey")
                    # 馬体重もWaku/Umabanと同じ仕組み: JRA発表前は<td class="Weight"></td>が空で、
                    # 発表後（通常発走1時間前）に"452(+6)"のような実テキストが静的に入る。
                    weight_td = row.find("td", class_="Weight")

                    entries.append({
                        "bracket_num_raw": b_num_td.get_text(strip=True) if b_num_td else "",
                        "horse_num_raw": h_num_text,
                        "horse_name": h_link.get_text(strip=True),
                        "horse_id": h_id,
                        "sex_age_raw": sex_age_td.get_text(strip=True) if sex_age_td else "",
                        "jockey_weight_raw": j_weight_td.get_text(strip=True) if j_weight_td else "",
                        "jockey_name": jockey_td.get_text(strip=True) if jockey_td else "",
                        "odds_raw": "",  # predict.py側でオッズAPI(fetch_live_odds)の結果を突き合わせて埋める
                        "popularity_raw": "",
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