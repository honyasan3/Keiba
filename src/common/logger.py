"""統一ログ出力管理モジュール"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "keiba_ai",
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """指定された名前のLoggerを構築して返します。"""
    logger = logging.getLogger(name)

    # 既にハンドラが設定されている場合は再設定を防止
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ログフォーマット設定
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # コンソール出力ハンドラ
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ファイル出力ハンドラ
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path / "app.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger