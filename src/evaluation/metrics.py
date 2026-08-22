"""モデル評価指標（AUC, LogLoss, Accuracy, Precision, Recall, 確率較正）算出モジュール"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from src.common.logger import setup_logger

logger = setup_logger("metrics")


class MetricsEvaluator:
    """分類モデルの各種評価指標を算出するクラス"""

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """AUC, LogLoss, Accuracy, Precision, Recallを一括計算"""
        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        y_pred_bin = (y_pred_prob >= threshold).astype(int)

        try:
            auc = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
        except Exception:
            auc = 0.0

        try:
            loss = round(float(log_loss(y_true, y_pred_prob)), 4)
        except Exception:
            loss = 0.0

        accuracy = round(float(accuracy_score(y_true, y_pred_bin)), 4)
        precision = round(float(precision_score(y_true, y_pred_bin, zero_division=0)), 4)
        recall = round(float(recall_score(y_true, y_pred_bin, zero_division=0)), 4)

        return {
            "auc": auc,
            "logloss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """ブライアスコア（予測確率と実結果の二乗誤差平均。較正・識別力の両方を反映する単一指標）"""
        y_true = np.array(y_true, dtype=float)
        y_pred_prob = np.array(y_pred_prob, dtype=float)
        return round(float(np.mean((y_pred_prob - y_true) ** 2)), 5)

    @staticmethod
    def calibration_report(
        y_true: np.ndarray, y_pred_prob: np.ndarray, bin_edges: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """予測確率が実際の的中率と一致しているか（確率較正）をビン別に集計する。

        ベッティング判断（EV = pred_prob × odds）は pred_prob をそのまま信頼して成立しているため、
        「予測45%のグループが実際には何%的中しているか」のズレが、EVしきい値戦略の損益に直結する。
        AUCはランキング能力（高い予測ほど実際に上位か）しか見ないため、較正のズレはAUCだけでは
        検出できないことに注意。デフォルトのbin_edgesは0〜1を10等分するが、意思決定に近い高確率帯
        （0.35〜0.50付近）を細かく見たい場合はbin_edgesを指定する。
        """
        if bin_edges is None:
            bin_edges = [round(i * 0.1, 2) for i in range(11)]

        df = pd.DataFrame({
            "y_true": np.array(y_true, dtype=float),
            "y_pred": np.array(y_pred_prob, dtype=float),
        })
        df["bin"] = pd.cut(df["y_pred"], bins=bin_edges, include_lowest=True)

        report = df.groupby("bin", observed=True).agg(
            count=("y_true", "size"),
            mean_pred=("y_pred", "mean"),
            actual_rate=("y_true", "mean"),
        ).reset_index()
        report["gap"] = report["mean_pred"] - report["actual_rate"]
        report["mean_pred"] = report["mean_pred"].round(4)
        report["actual_rate"] = report["actual_rate"].round(4)
        report["gap"] = report["gap"].round(4)
        return report