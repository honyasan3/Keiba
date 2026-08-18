"""時系列データおよび特徴量の未来情報漏洩（データリーク）検証モジュール"""
from typing import List
from src.common.logger import setup_logger

logger = setup_logger("leak_validator")


class DataLeakageValidator:
    """発走前の予測に使用してはならない未来情報の混入を検知するクラス"""

    # 発走後にしか判明しない禁止カラム一覧
    LEAK_FORBIDDEN_COLUMNS = {
        "rank",
        "finish_time_sec",
        "margin",
        "last_3f_time",
        "passage_order",
        "target_win",
        "target_place",
        "is_win",
        "is_place",
    }

    @classmethod
    def validate_features(cls, feature_cols: List[str]) -> bool:
        """
        指定された特徴量リストに未来情報が含まれていないかを検証。
        """
        leak_detected = False
        forbidden_found = set(feature_cols) & cls.LEAK_FORBIDDEN_COLUMNS

        if forbidden_found:
            logger.error(
                f"【重大なデータリーク検知】発走前予測に使用できないカラムが含まれています: {forbidden_found}"
            )
            leak_detected = True

        if not leak_detected:
            logger.info(f"リーク検証をクリアしました (使用特徴量数: {len(feature_cols)})")
            return True

        return False