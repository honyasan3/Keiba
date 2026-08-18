"""データベーステーブル定義およびCRUD操作モジュール"""
from typing import Any, Dict
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Date, UniqueConstraint
from sqlalchemy.orm import Session, relationship
from src.common.db import Base
from src.common.exceptions import DatabaseError
from src.common.logger import setup_logger

logger = setup_logger("repository")


class RaceModel(Base):
    """レース基本情報テーブル"""
    __tablename__ = "races"

    race_id = Column(String(32), primary_key=True)
    race_title = Column(String(128), nullable=True)
    race_date = Column(String(32), nullable=True)        # 開催日 (YYYY-MM-DD)
    race_round = Column(Integer, nullable=True)          # R数 (1~12)
    course_type = Column(String(16), nullable=True)      # 芝 / ダ / 障
    distance = Column(Integer, nullable=True)            # 距離 (m)
    weather = Column(String(16), nullable=True)          # 天候
    track_condition = Column(String(16), nullable=True)  # 馬場状態

    results = relationship("RaceResultModel", back_populates="race", cascade="all, delete-orphan")


class RaceResultModel(Base):
    """出走馬・レース成績テーブル"""
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(32), ForeignKey("races.race_id"), nullable=False)
    rank = Column(Integer, nullable=True)                # 着順
    bracket_num = Column(Integer, nullable=True)         # 枠番
    horse_num = Column(Integer, nullable=True)           # 馬番
    horse_name = Column(String(64), nullable=False)      # 馬名
    horse_id = Column(String(32), nullable=True)         # 馬ID
    gender = Column(String(8), nullable=True)            # 牡 / 牝 / セ
    age = Column(Integer, nullable=True)                 # 年齢
    jockey_weight = Column(Float, nullable=True)         # 斤量
    jockey_name = Column(String(64), nullable=True)      # 騎手名
    finish_time_sec = Column(Float, nullable=True)       # タイム（秒）
    margin = Column(String(32), nullable=True)           # 着差
    passage_order = Column(String(32), nullable=True)    # 通過順位 (例: 3-3)
    last_3f_time = Column(Float, nullable=True)          # 上がり3F
    odds = Column(Float, nullable=True)                  # 単勝オッズ
    popularity = Column(Integer, nullable=True)          # 人気順
    horse_weight = Column(Integer, nullable=True)        # 馬体重
    horse_weight_diff = Column(Integer, nullable=True)   # 馬体重増減

    __table_args__ = (
        UniqueConstraint("race_id", "horse_num", name="uq_race_horse"),
    )

    race = relationship("RaceModel", back_populates="results")


class RaceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save_race_data(self, cleaned_data: Dict[str, Any]) -> None:
        race_id = cleaned_data["race_id"]
        try:
            race = self.session.query(RaceModel).filter_by(race_id=race_id).first()
            if not race:
                race = RaceModel(
                    race_id=race_id,
                    race_title=cleaned_data.get("race_title"),
                    race_date=cleaned_data.get("race_date"),
                    race_round=cleaned_data.get("race_round"),
                    course_type=cleaned_data.get("course_type"),
                    distance=cleaned_data.get("distance"),
                    weather=cleaned_data.get("weather"),
                    track_condition=cleaned_data.get("track_condition"),
                )
                self.session.add(race)
            else:
                race.race_title = cleaned_data.get("race_title")
                race.race_date = cleaned_data.get("race_date")
                race.race_round = cleaned_data.get("race_round")
                race.course_type = cleaned_data.get("course_type")
                race.distance = cleaned_data.get("distance")
                race.weather = cleaned_data.get("weather")
                race.track_condition = cleaned_data.get("track_condition")

            for res in cleaned_data.get("results", []):
                h_num = res.get("horse_num")
                if h_num is None:
                    continue

                existing = self.session.query(RaceResultModel).filter_by(
                    race_id=race_id, horse_num=h_num
                ).first()

                data_dict = {
                    "rank": res.get("rank"),
                    "bracket_num": res.get("bracket_num"),
                    "horse_name": res.get("horse_name"),
                    "horse_id": res.get("horse_id"),
                    "gender": res.get("gender"),
                    "age": res.get("age"),
                    "jockey_weight": res.get("jockey_weight"),
                    "jockey_name": res.get("jockey_name"),
                    "finish_time_sec": res.get("finish_time_sec"),
                    "margin": res.get("margin"),
                    "passage_order": res.get("passage_order"),
                    "last_3f_time": res.get("last_3f_time"),
                    "odds": res.get("odds"),
                    "popularity": res.get("popularity"),
                    "horse_weight": res.get("horse_weight"),
                    "horse_weight_diff": res.get("horse_weight_diff"),
                }

                if existing:
                    for k, v in data_dict.items():
                        setattr(existing, k, v)
                else:
                    new_item = RaceResultModel(race_id=race_id, horse_num=h_num, **data_dict)
                    self.session.add(new_item)

            self.session.commit()
            logger.info(f"DB保存完了 (Race ID: {race_id})")
        except Exception as e:
            self.session.rollback()
            logger.error(f"DB保存処理中にエラーが発生しました (Race ID: {race_id}): {e}")
            raise DatabaseError(f"保存失敗: {e}") from e