"""レース基本属性およびレース内相対特徴量モジュール"""
import pandas as pd
from src.features.base_feature import BaseFeatureExtractor
from src.common.logger import setup_logger

logger = setup_logger("race_features")


class RaceFeatureExtractor(BaseFeatureExtractor):
    """レース属性とレース内相対特徴量の抽出クラス"""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("レース属性・相対特徴量の生成を開始します。")
        df = df.copy()

        # 1. 競馬場コードの抽出 (race_id の 5~6桁目)
        df["venue_code"] = df["race_id"].astype(str).str[4:6].astype(int)

        # 2. カテゴリ変数の数値変換 (欠損値は 'Unknown' として扱う)
        category_mappings = {
            "course_type": {"芝": 0, "ダート": 1, "障害": 2},
            "weather": {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5},
            "track_condition": {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不良": 3},
            "gender": {"牡": 0, "牝": 1, "セ": 2},
        }

        for col, mapping in category_mappings.items():
            if col in df.columns:
                df[f"{col}_cat"] = df[col].map(mapping).fillna(-1).astype(int)

        # 3. 季節・月特徴量
        if "race_date" in df.columns:
            df["race_date_dt"] = pd.to_datetime(df["race_date"])
            df["month"] = df["race_date_dt"].dt.month

        # 4. レース内での相対特徴量 (レース平均斤量との差など)
        if "race_id" in df.columns and "jockey_weight" in df.columns:
            race_weight_mean = df.groupby("race_id")["jockey_weight"].transform("mean")
            df["jockey_weight_diff_from_race_mean"] = df["jockey_weight"] - race_weight_mean

        # 5. 頭数（レースごとの出走数）
        if "race_id" in df.columns:
            df["race_horse_count"] = df.groupby("race_id")["horse_num"].transform("count")

        logger.info("レース属性・相対特徴量の生成が完了しました。")
        return df