"""実運用フォワードテスト実績レポートスクリプト

predict.py が predictions テーブルへ記録した推奨買い目と、run_daily_settlement.py による精算結果を
集計し、実際の的中率・回収率の推移を表示する。README「6.1」で正直に書いた通り、現時点のβ運用は
まだ実績が薄いため、このスクリプトで運用結果を継続的に可視化し、回収率を訴求文言に使わずとも
自分たちで実態を追えるようにすることを目的とする。

使い方:
    python track_record_report.py                 # 全期間・全券種の実績を表示
    python track_record_report.py --bet-type place # 券種を絞る
    python track_record_report.py --since 2026-08-01
"""
import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.pipeline.repository import PredictionModel

logger = setup_logger("track_record_report")


def load_predictions(db_connector: DatabaseConnector, since: Optional[str] = None) -> pd.DataFrame:
    with db_connector.get_session() as session:
        query = session.query(PredictionModel)
        if since:
            query = query.filter(PredictionModel.race_date >= since)
        rows = query.all()
        records = [
            {
                "race_id": r.race_id,
                "race_title": r.race_title,
                "race_date": r.race_date,
                "bet_type": r.bet_type,
                "horse_num": r.horse_num,
                "horse_name": r.horse_name,
                "odds_at_predict": r.odds_at_predict,
                "pred_prob": r.pred_prob,
                "ev": r.ev,
                "bet_amount": r.bet_amount,
                "predicted_at": r.predicted_at,
                "settled": r.settled,
                "is_hit": r.is_hit,
                "payout_amount": r.payout_amount,
                "settled_at": r.settled_at,
            }
            for r in rows
        ]
        return pd.DataFrame(records)


def _summarize(df: pd.DataFrame) -> dict:
    bet_count = len(df)
    hit_count = int(df["is_hit"].sum())
    invest = int(df["bet_amount"].sum())
    payout = int(df["payout_amount"].fillna(0).sum())
    profit = payout - invest
    roi = round(payout / invest * 100, 2) if invest > 0 else 0.0
    hit_rate = round(hit_count / bet_count * 100, 2) if bet_count > 0 else 0.0
    return {
        "件数": bet_count,
        "的中": hit_count,
        "的中率(%)": hit_rate,
        "投資額": invest,
        "払戻額": payout,
        "収支": profit,
        "回収率(%)": roi,
    }


def run_report(bet_types: Optional[List[str]] = None, since: Optional[str] = None) -> None:
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    df = load_predictions(db_connector, since=since)
    if df.empty:
        print("\npredictionsテーブルに記録がありません。predict.py を実行すると記録が蓄積されます。\n")
        return

    if bet_types:
        df = df[df["bet_type"].isin(bet_types)]
        if df.empty:
            print(f"\n指定した券種({bet_types})の記録がありません。\n")
            return

    settled = df[df["settled"] == True].copy()  # noqa: E712
    unsettled = df[df["settled"] == False].copy()  # noqa: E712

    print("\n" + "=" * 70)
    print(f" 【フォワードテスト実績レポート】 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f" 総記録件数: {len(df)}件（精算済み: {len(settled)}件 / 未精算: {len(unsettled)}件)")

    if not unsettled.empty:
        pending_by_type = unsettled.groupby("bet_type").size().to_dict()
        print(f" 未精算内訳: {pending_by_type}（run_daily_settlement.pyの実行で精算されます）")

    if settled.empty:
        print("\n精算済みの記録がまだありません。run_daily_settlement.py を実行してください。\n")
        return

    print("\n[全体サマリー（精算済みのみ）]")
    print(tabulate([_summarize(settled)], headers="keys", tablefmt="github", showindex=False))

    print("\n[券種別]")
    by_type_rows = []
    for bt, g in settled.groupby("bet_type"):
        row = {"bet_type": bt}
        row.update(_summarize(g))
        by_type_rows.append(row)
    print(tabulate(by_type_rows, headers="keys", tablefmt="github", showindex=False))

    print("\n[週別推移]")
    settled["race_date_dt"] = pd.to_datetime(settled["race_date"])
    settled["week"] = settled["race_date_dt"].dt.to_period("W").astype(str)
    weekly_rows = []
    for wk, g in sorted(settled.groupby("week"), key=lambda x: x[0]):
        row = {"週": wk}
        row.update(_summarize(g))
        weekly_rows.append(row)
    print(tabulate(weekly_rows, headers="keys", tablefmt="github", showindex=False))

    print("\n[オッズ帯別（複勝ルールの想定オッズ帯3〜5倍で実際に的中傾向が出ているか確認用）]")
    settled_odds = settled.dropna(subset=["odds_at_predict"]).copy()
    if not settled_odds.empty:
        settled_odds["odds_band"] = pd.cut(
            settled_odds["odds_at_predict"],
            bins=[0, 1.5, 3.0, 5.0, 10.0, 9999],
            labels=["~1.5倍", "1.5~3倍", "3~5倍", "5~10倍", "10倍~"],
        )
        band_rows = []
        for band, g in settled_odds.groupby("odds_band", observed=True):
            if len(g) == 0:
                continue
            row = {"オッズ帯": band}
            row.update(_summarize(g))
            band_rows.append(row)
        print(tabulate(band_rows, headers="keys", tablefmt="github", showindex=False))

    print(
        "\n※ このレポートはβ運用の実績を正直に追跡するためのものです。件数がまだ少ないうちは"
        "回収率のブレが大きく出ます。README「6.1」に記載の通り、これを保証や訴求文言として"
        "使わないでください。\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="フォワードテスト実績レポート")
    parser.add_argument("--bet-type", type=str, default=None, help="place,win,wide をカンマ区切りで指定（省略時は全券種）")
    parser.add_argument("--since", type=str, default=None, help="この日付(YYYY-MM-DD)以降のレースのみ集計")
    args = parser.parse_args()

    bt_list = [b.strip() for b in args.bet_type.split(",")] if args.bet_type else None
    run_report(bet_types=bt_list, since=args.since)
