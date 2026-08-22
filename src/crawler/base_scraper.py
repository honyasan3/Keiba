"""通信・キャッシュ・遅延処理を司るクローラー基底モジュール"""
import hashlib
import random
import time
from pathlib import Path
from typing import Optional
import requests
from config.config_loader import CrawlerConfig
from src.common.exceptions import ScraperError
from src.common.logger import setup_logger

logger = setup_logger("base_scraper")


class BaseScraper:
    """HTTP通信とローカルキャッシュ処理を行うクローラー基底クラス"""

    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self.session = requests.Session()
        
        # ブラウザと同等の標準ヘッダーを設定
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        })

        # キャッシュディレクトリの作成
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, url: str) -> Path:
        """URLからハッシュ値を生成し、キャッシュファイルパスを作成"""
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.config.cache_dir / f"{url_hash}.html"

    def fetch_page(self, url: str, use_cache: bool = True) -> str:
        """
        指定されたURLのHTMLを取得。
        キャッシュが存在する場合は通信を行わずにローカルから即座に返却。
        """
        cache_path = self._get_cache_path(url)

        # 1. キャッシュの読み込み（Webアクセスなし）
        if use_cache and cache_path.exists():
            logger.debug(f"キャッシュから読み込みます: {url}")
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"キャッシュ読み込みに失敗したため再取得します: {e}")

        # 2. Webサーバーからの取得（再試行ロジック付き）
        html_content = self._fetch_with_retry(url)

        # 3. キャッシュへ保存
        try:
            cache_path.write_text(html_content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"キャッシュ保存に失敗しました: {e}")

        # 4. ランダムウェイト（1.5秒〜3.0秒のランダム待機で機械的アクセスを回避）
        sleep_time = round(random.uniform(self.config.min_delay, self.config.max_delay), 2)
        logger.debug(f"アクセス間隔待機: {sleep_time} 秒")
        time.sleep(sleep_time)

        return html_content

    def _fetch_with_retry(self, url: str) -> str:
        """指数バックオフを用いてHTTPリクエストを実行"""
        retries = 0
        backoff_time = 2.0
        while retries <= self.config.max_retries:
            try:
                logger.info(f"ページを取得中: {url}")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                # db.netkeiba.com(EUC-JP)とrace.netkeiba.com(UTF-8)でエンコーディングが異なるため自動判定する
                response.encoding = response.apparent_encoding
                return response.text
            except requests.RequestException as e:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"最大再試行回数を超過しました ({url}): {e}")
                    raise ScraperError(f"HTTPリクエスト失敗: {url}") from e

                logger.warning(
                    f"リクエスト失敗 ({e})。{backoff_time}秒後に再試行します... ({retries}/{self.config.max_retries})"
                )
                time.sleep(backoff_time)
                backoff_time *= 2.0

        raise ScraperError(f"ページ取得不能: {url}")