"""データベーステーブル定義およびCRUD操作モジュール"""
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Date, UniqueConstraint
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


class PredictionModel(Base):
    """AI推奨買い目の記録・精算結果テーブル（実運用の的中率・回収率トラックレコード用）"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(32), nullable=False)
    race_title = Column(String(128), nullable=True)
    race_date = Column(String(32), nullable=True)          # 開催日 (YYYY-MM-DD)
    bet_type = Column(String(16), nullable=False)           # place / win / wide
    horse_num = Column(String(16), nullable=False)          # place,win: 馬番 / wide: "3-7"のペア表記
    horse_name = Column(String(128), nullable=True)
    odds_at_predict = Column(Float, nullable=True)          # 推論時点の単勝オッズ
    pred_prob = Column(Float, nullable=True)                # 推論時点の的中確率
    ev = Column(Float, nullable=True)                       # 推論時点の期待値
    bet_amount = Column(Integer, nullable=False, default=100)
    predicted_at = Column(String(32), nullable=True)        # 推論実行日時 (YYYY-MM-DD HH:MM:SS)

    settled = Column(Boolean, nullable=False, default=False)
    is_hit = Column(Boolean, nullable=True)
    payout_amount = Column(Integer, nullable=True)
    settled_at = Column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("race_id", "bet_type", "horse_num", name="uq_prediction_bet"),
    )


class PredictionRepository:
    """推奨買い目の保存・精算結果反映を担うリポジトリ（predict.py / run_daily_settlement.py から利用）"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save_predictions(
        self, race_id: str, race_title: Optional[str], race_date: Optional[str], records: List[Dict[str, Any]]
    ) -> None:
        """1レース分の推奨買い目を保存。同一(race_id, bet_type, horse_num)は再推論内容で上書き（再実行しても重複しない）"""
        try:
            for rec in records:
                horse_num = str(rec["horse_num"])
                bet_type = rec["bet_type"]
                existing = (
                    self.session.query(PredictionModel)
                    .filter_by(race_id=race_id, bet_type=bet_type, horse_num=horse_num)
                    .first()
                )
                if existing:
                    existing.race_title = race_title
                    existing.race_date = race_date
                    existing.horse_name = rec.get("horse_name")
                    existing.odds_at_predict = rec.get("odds_at_predict")
                    existing.pred_prob = rec.get("pred_prob")
                    existing.ev = rec.get("ev")
                    existing.bet_amount = rec.get("bet_amount", 100)
                    existing.predicted_at = rec.get("predicted_at")
                else:
                    self.session.add(PredictionModel(
                        race_id=race_id,
                        race_title=race_title,
                        race_date=race_date,
                        bet_type=bet_type,
                        horse_num=horse_num,
                        horse_name=rec.get("horse_name"),
                        odds_at_predict=rec.get("odds_at_predict"),
                        pred_prob=rec.get("pred_prob"),
                        ev=rec.get("ev"),
                        bet_amount=rec.get("bet_amount", 100),
                        predicted_at=rec.get("predicted_at"),
                    ))
            self.session.commit()
            logger.info(f"買い目記録を保存しました (Race ID: {race_id}, {len(records)}件)")
        except Exception as e:
            self.session.rollback()
            logger.error(f"買い目記録の保存に失敗しました (Race ID: {race_id}): {e}")
            raise DatabaseError(f"予測保存失敗: {e}") from e

    def get_unsettled(self, race_date_lte: Optional[str] = None, bet_types: Optional[List[str]] = None) -> List["PredictionModel"]:
        """未精算の買い目を取得（race_date_lte指定時はその日付以前のみ）"""
        query = self.session.query(PredictionModel).filter_by(settled=False)
        if race_date_lte:
            query = query.filter(PredictionModel.race_date <= race_date_lte)
        if bet_types:
            query = query.filter(PredictionModel.bet_type.in_(bet_types))
        return query.all()

    def mark_settled(self, prediction_id: int, is_hit: bool, payout_amount: int, settled_at: str) -> None:
        row = self.session.query(PredictionModel).filter_by(id=prediction_id).first()
        if row:
            row.settled = True
            row.is_hit = is_hit
            row.payout_amount = payout_amount
            row.settled_at = settled_at
            self.session.commit()


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