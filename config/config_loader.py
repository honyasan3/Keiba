"""設定ファイルの読み込みとバリデーションモジュール"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml
from src.common.exceptions import ConfigurationError


@dataclass(frozen=True)
class DBConfig:
    connection_string: str


@dataclass(frozen=True)
class CrawlerConfig:
    min_delay: float
    max_delay: float
    max_retries: int
    cache_dir: Path
    user_agent: str


@dataclass(frozen=True)
class DataConfig:
    start_year: int
    end_year: int


@dataclass(frozen=True)
class NotificationConfig:
    discord_webhook_url: str = ""
    enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    db: DBConfig
    crawler: CrawlerConfig
    data: DataConfig
    notification: NotificationConfig  # 追加


class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config/settings.yaml") -> AppConfig:
        path = Path(config_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / config_path

        if not path.exists():
            raise ConfigurationError(f"設定ファイルが見つかりません: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f)

            db_conf = DBConfig(
                connection_string=data["db"]["connection_string"]
            )
            crawler_conf = CrawlerConfig(
                min_delay=float(data["crawler"].get("min_delay", 1.5)),
                max_delay=float(data["crawler"].get("max_delay", 3.0)),
                max_retries=int(data["crawler"]["max_retries"]),
                cache_dir=Path(data["crawler"]["cache_dir"]),
                user_agent=str(data["crawler"]["user_agent"]),
            )
            data_conf = DataConfig(
                start_year=int(data["data"]["start_year"]),
                end_year=int(data["data"]["end_year"]),
            )

            # notification の読み込み (設定がない場合はデフォルト値)
            notif_raw = data.get("notification", {})
            notif_conf = NotificationConfig(
                discord_webhook_url=str(notif_raw.get("discord_webhook_url", "")),
                enabled=bool(notif_raw.get("enabled", False)),
            )

            return AppConfig(
                db=db_conf,
                crawler=crawler_conf,
                data=data_conf,
                notification=notif_conf,
            )
        except KeyError as e:
            raise ConfigurationError(f"設定ファイルの必須キーが不足しています: {e}")
        except Exception as e:
            raise ConfigurationError(f"設定ファイルの読み込みに失敗しました: {e}")