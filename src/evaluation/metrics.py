"""モデル評価指標（AUC, LogLoss, Accuracy, Precision, Recall）算出モジュール"""
from typing import Any, Dict
import numpy as np
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