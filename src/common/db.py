"""データベース接続およびセッション管理モジュール"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.common.exceptions import DatabaseError
from src.common.logger import setup_logger

logger = setup_logger("db")
Base = declarative_base()


class DatabaseConnector:
    """データベース接続管理およびセッション提供を行うクラス"""

    def __init__(self, connection_string: str) -> None:
        try:
            # SQLiteの場合、親ディレクトリ（data/）が存在しない場合は自動作成
            if connection_string.startswith("sqlite:///"):
                db_path = connection_string.replace("sqlite:///", "")
                from pathlib import Path
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            self.engine = create_engine(
                connection_string,
                echo=False,
                future=True
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        except Exception as e:
            logger.error(f"データベースエンジンの作成に失敗しました: {e}")
            raise DatabaseError(f"DB初期化エラー: {e}")

    def create_tables(self) -> None:
        """定義されているテーブル（Baseのサブクラス）をDB上に作成します。"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("データベーステーブルの作成・確認が完了しました。")
        except Exception as e:
            logger.error(f"テーブル作成に失敗しました: {e}")
            raise DatabaseError(f"テーブル作成エラー: {e}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """安全なトランザクション管理を提供するセッション・コンテキストマネージャ"""
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"データベース処理中にエラーが発生したためロールバックしました: {e}")
            raise DatabaseError(f"セッション処理エラー: {e}")
        finally:
            session.close()