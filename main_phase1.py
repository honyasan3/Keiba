"""フェーズ1 パイプライン一括実行エントリーポイント"""
import sys
from datetime import datetime, timedelta
import pandas as pd
from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.crawler.race_scraper import RaceScraper
from src.crawler.html_parser import RaceHtmlParser
from src.pipeline.cleaner import DataCleaner
from src.pipeline.repository import RaceRepository
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.dataset.leak_validator import DataLeakageValidator

logger = setup_logger("main_phase1")


def generate_date_range(start_date_str: str, end_date_str: str):
    """開始日と終了日（YYYYMMDD）から日付リストを生成"""
    start = datetime.strptime(start_date_str, "%Y%m%d")
    end = datetime.strptime(end_date_str, "%Y%m%d")
    curr = start
    while curr <= end:
        yield curr.strftime("%Y%m%d")
        curr += timedelta(days=1)


def main() -> None:
    logger.info("=== フェーズ1 パイプライン処理を開始します ===")
    try:
        # 1. 設定ファイルの読み込み
        config = ConfigLoader.load_config("config/settings.yaml")
        logger.info(f"設定ファイルを読み込みました (DB: {config.db.connection_string})")

        # 2. データベース接続およびテーブル作成
        db_connector = DatabaseConnector(config.db.connection_string)
        db_connector.create_tables()

        scraper = RaceScraper(config.crawler)

        # 3. 取得対象期間の設定
        start_date = "20260101"
        end_date = "20260819"

        all_race_ids = []
        logger.info(f"期間内のレースIDを検索します: {start_date} ~ {end_date}")
        for date_str in generate_date_range(start_date, end_date):
            ids = scraper.fetch_race_ids_by_date(date_str)
            all_race_ids.extend(ids)

        logger.info(f"合計 {len(all_race_ids)} 件のレースが見つかりました。詳細データの取得を開始します。")

        # 4. レース結果の取得・パース・DB保存
        with db_connector.get_session() as session:
            repository = RaceRepository(session)
            
            # テスト用に最初の数レース（例: 3レース）を処理する場合は [:3] などを適用
            for race_id in all_race_ids:
                logger.info(f"--- 処理対象 Race ID: {race_id} ---")
                try:
                    html_content = scraper.fetch_race_result(race_id, use_cache=True)
                    parsed_raw = RaceHtmlParser.parse_race_result(html_content, race_id)
                    
                    if not parsed_raw.get("results"):
                        logger.warning(f"Race ID {race_id}: 結果データが空のためスキップします。")
                        continue

                    cleaned_data = DataCleaner.clean_race_data(parsed_raw)
                    repository.save_race_data(cleaned_data)
                except Exception as e:
                    logger.warning(f"Race ID {race_id} の処理中にエラーが発生しました（スキップ）: {e}")

        logger.info("=== フェーズ1 パイプライン処理が正常に完了しました ===")

    except Exception as e:
        logger.critical(f"パイプライン処理が異常終了しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()