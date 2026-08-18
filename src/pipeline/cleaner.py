"""データクレンジング・前処理モジュール"""
import re
from typing import Any, Dict, List, Optional
from src.common.exceptions import ParseError
from src.common.logger import setup_logger

logger = setup_logger("cleaner")


class DataCleaner:
    @staticmethod
    def clean_race_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        if not raw_data:
            raise ParseError("クレンジング対象のデータが空です。")

        cleaned_results: List[Dict[str, Any]] = []
        for item in raw_data.get("results", []):
            try:
                # 性別・年齢の分離 (例: "牡3", "牝4", "セ5")
                gender, age = None, None
                sex_age = item.get("sex_age_raw", "")
                if sex_age:
                    gender = sex_age[0]
                    age_match = re.search(r"\d+", sex_age)
                    age = int(age_match.group()) if age_match else None

                # 馬体重と増減 (例: "480(+2)", "500(-4)", "470(0)")
                hw, hw_diff = None, None
                hw_raw = item.get("horse_weight_raw", "")
                hw_match = re.search(r"(\d+)(?:\(([-+]?\d+)\))?", hw_raw)
                if hw_match:
                    hw = int(hw_match.group(1))
                    if hw_match.group(2):
                        hw_diff = int(hw_match.group(2))

                cleaned_results.append({
                    "rank": DataCleaner._extract_int(item.get("rank_raw")),
                    "bracket_num": DataCleaner._extract_int(item.get("bracket_num_raw")),
                    "horse_num": DataCleaner._extract_int(item.get("horse_num_raw")),
                    "horse_name": item.get("horse_name", "").strip(),
                    "horse_id": item.get("horse_id", ""),
                    "gender": gender,
                    "age": age,
                    "jockey_weight": DataCleaner._extract_float(item.get("jockey_weight_raw")),
                    "jockey_name": item.get("jockey_name", "").strip(),
                    "finish_time_sec": DataCleaner._parse_time_to_seconds(item.get("finish_time_raw")),
                    "margin": item.get("margin_raw", "").strip(),
                    "passage_order": item.get("passage_order_raw", "").strip(),
                    "last_3f_time": DataCleaner._extract_float(item.get("last_3f_raw")),
                    "odds": DataCleaner._extract_float(item.get("odds_raw")),
                    "popularity": DataCleaner._extract_int(item.get("popularity_raw")),
                    "horse_weight": hw,
                    "horse_weight_diff": hw_diff,
                })
            except Exception as e:
                logger.warning(f"行データのクレンジングスキップ: {e}")
                continue

        return {
            "race_id": str(raw_data.get("race_id")),
            "race_title": str(raw_data.get("race_title", "")).strip(),
            "race_date": raw_data.get("race_date"),
            "race_round": raw_data.get("race_round"),
            "course_type": raw_data.get("course_type"),
            "distance": raw_data.get("distance"),
            "weather": raw_data.get("weather"),
            "track_condition": raw_data.get("track_condition"),
            "results": cleaned_results,
        }

    @staticmethod
    def _extract_int(val: Optional[str]) -> Optional[int]:
        if not val:
            return None
        match = re.search(r"\d+", str(val))
        return int(match.group()) if match else None

    @staticmethod
    def _extract_float(val: Optional[str]) -> Optional[float]:
        if not val:
            return None
        match = re.search(r"\d+\.?\d*", str(val))
        return float(match.group()) if match else None

    @staticmethod
    def _parse_time_to_seconds(time_str: Optional[str]) -> Optional[float]:
        if not time_str:
            return None
        try:
            parts = time_str.strip().split(":")
            if len(parts) == 2:
                return round(float(parts[0]) * 60.0 + float(parts[1]), 2)
            elif len(parts) == 1:
                return round(float(parts[0]), 2)
        except ValueError:
            pass
        return None