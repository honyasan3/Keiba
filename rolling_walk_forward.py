"""ローリングwalk-forward検証スクリプト

optimize_betting.py の単一Val/Test窓によるwalk-forward検証（複勝Test ROI 88.51%、単勝85.30%。
確率較正やオッズ帯制限を試しても改善せず）が、たまたまこの1つの窓に固有の結果なのか、複数の
独立した期間で一貫した傾向なのかを確認する。

3つの独立したfoldそれぞれについて、
  1. train窓（拡張窓方式。foldが進むほどtrain期間が長くなる）でLGBM/CatBoost/LambdaMARTを
     ゼロから再学習する（models_saved/ の本番モデルは一切上書きしない。学習結果はメモリ上のみ）
  2. val窓のみでベッティング閾値をグリッドサーチ選定する（複勝・単勝・ワイド）
  3. 選定した閾値を、選定に一切使っていないtest窓に一度だけ適用して評価する
という、optimize_betting.py と同一の手続きを繰り返し、3つの独立したtest窓の結果を集計する。
ワイドはpredict.py本番と同じくMonteCarloRaceSimulatorでレースごとにシミュレーションするため、
複勝・単勝より時間がかかる（1レースあたり数ms〜十数ms程度）。

各fold単体の学習・推論には数分〜十数分かかるため、全体の実行には相応の時間を要する。
"""
import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.ensemble_runner import run_ensemble_inference, train_ensemble_models
from src.evaluation.strategy_optimizer import BettingStrategyOptimizer
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("rolling_walk_forward")

# (train_end_frac, val_end_frac, test_end_frac) の3fold。train窓は毎回広がる拡張窓方式。
# fold3は現行のoptimize_betting.pyと同一の窓（70/15/15）。
FOLD_FRACTIONS = [
    (0.40, 0.55, 0.70),
    (0.55, 0.70, 0.85),
    (0.70, 0.85, 1.00),
]


def get_all_race_data(db_connector: DatabaseConnector) -> pd.DataFrame:
    with db_connector.get_session() as session:
        query = (
            session.query(
                RaceModel.race_id,
                RaceModel.race_title,
                RaceModel.race_date,
                RaceModel.race_round,
                RaceModel.course_type,
                RaceModel.distance,
                RaceModel.weather,
                RaceModel.track_condition,
                RaceResultModel.rank,
                RaceResultModel.bracket_num,
                RaceResultModel.horse_num,
                RaceResultModel.horse_name,
                RaceResultModel.horse_id,
                RaceResultModel.gender,
                RaceResultModel.age,
                RaceResultModel.jockey_weight,
                RaceResultModel.jockey_name,
                RaceResultModel.finish_time_sec,
                RaceResultModel.margin,
                RaceResultModel.passage_order,
                RaceResultModel.last_3f_time,
                RaceResultModel.odds,
                RaceResultModel.popularity,
                RaceResultModel.horse_weight,
                RaceResultModel.horse_weight_diff,
            )
            .join(RaceResultModel, RaceModel.race_id == RaceResultModel.race_id)
        )
        return pd.DataFrame(query.all())


def run_fold(
    fold_idx: int,
    featured_df: pd.DataFrame,
    unique_dates: list,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict:
    n = len(unique_dates)
    train_end = unique_dates[int(n * train_frac)]
    val_end = unique_dates[int(n * val_frac)]
    test_end_idx = int(n * test_frac)
    test_end = unique_dates[test_end_idx - 1] if test_end_idx < n else unique_dates[-1]

    splitter = TimeSeriesDataSplitter()
    train_df, val_df, test_df = splitter.split_by_date(featured_df, train_end=train_end, val_end=val_end)
    # split_by_dateはval_end以降を無条件にtest_dfへ入れるため、fold用のtest_endで切り詰める
    test_df = test_df[test_df["race_date"] <= pd.to_datetime(test_end)]

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    logger.info(
        f"[Fold {fold_idx}] Train: ~{train_end} ({len(train_df)}件) / "
        f"Val: {train_end}~{val_end} ({len(val_df)}件) / "
        f"Test: {val_end}~{test_end} ({len(test_df)}件)"
    )

    logger.info(f"[Fold {fold_idx}] トリプルアンサンブルを再学習中...")
    lgbm_p, cb_p, rank_p = train_ensemble_models(train_df, val_df)

    val_df = run_ensemble_inference(val_df, lgbm_predictor=lgbm_p, cb_predictor=cb_p, rank_predictor=rank_p)
    test_df = run_ensemble_inference(test_df, lgbm_predictor=lgbm_p, cb_predictor=cb_p, rank_predictor=rank_p)

    optimizer = BettingStrategyOptimizer()
    result = {
        "fold": fold_idx,
        "test_period": f"{test_df['race_date'].min()} ~ {test_df['race_date'].max()}",
    }

    place_val = optimizer.optimize_place_strategy(val_df)
    if not place_val.empty:
        best = place_val.iloc[0]
        test_eval = BettingStrategyOptimizer.prep_place_eval_df(test_df)
        result["place"] = BettingStrategyOptimizer.evaluate_place_strategy(
            test_eval,
            ev_th=best["ev_threshold"],
            min_p=best["min_prob"],
            max_r=int(best["max_rank"]),
            odds_range=(best["min_odds"], best["max_odds"]),
        )
        result["place_rule"] = (
            f"EV>={best['ev_threshold']}, prob>={best['min_prob']}, "
            f"rank<={int(best['max_rank'])}, odds {best['odds_range']}"
        )
    else:
        result["place"] = None
        result["place_rule"] = "該当なし"

    win_val = optimizer.optimize_win_strategy(val_df)
    if not win_val.empty:
        best = win_val.iloc[0]
        test_eval = BettingStrategyOptimizer.prep_win_eval_df(test_df)
        result["win"] = BettingStrategyOptimizer.evaluate_win_strategy(
            test_eval,
            ev_th=best["ev_threshold"],
            min_p=best["min_prob"],
            max_r=int(best["max_rank"]),
            odds_range=(best["min_odds"], best["max_odds"]),
        )
        result["win_rule"] = (
            f"EV>={best['ev_threshold']}, prob>={best['min_prob']}, "
            f"rank<={int(best['max_rank'])}, odds {best['odds_range']}"
        )
    else:
        result["win"] = None
        result["win_rule"] = "該当なし"

    logger.info(f"[Fold {fold_idx}] ワイド候補ペアをシミュレーション中（Val）...")
    val_wide_eval = BettingStrategyOptimizer.prep_wide_eval_df(val_df)
    wide_val = optimizer.optimize_wide_strategy(val_wide_eval)
    if not wide_val.empty:
        best = wide_val.iloc[0]
        logger.info(f"[Fold {fold_idx}] ワイド候補ペアをシミュレーション中（Test）...")
        test_wide_eval = BettingStrategyOptimizer.prep_wide_eval_df(test_df)
        result["wide"] = BettingStrategyOptimizer.evaluate_wide_strategy(
            test_wide_eval,
            ev_th=best["ev_threshold"],
            min_p=best["min_prob"],
            odds_range=(best["min_odds"], best["max_odds"]),
        )
        result["wide_rule"] = f"EV>={best['ev_threshold']}, prob>={best['min_prob']}, odds {best['odds_range']}"
    else:
        result["wide"] = None
        result["wide_rule"] = "該当なし"

    return result


def _pooled_roi(results: list, key: str) -> dict:
    """各foldのtest結果を、賭け金ベースでプールした合算ROIとして集計する（単純平均ではない）"""
    rows = [r[key] for r in results if r.get(key)]
    if not rows:
        return {}
    total_bets = sum(r["bet_count"] for r in rows)
    total_hits = sum(r["hit_count"] for r in rows)
    # roi(%) * bet_count*100 / 100 = roi(%) * bet_count が「払戻総額の100円単位換算」に相当
    total_return_units = sum(r["roi"] / 100.0 * r["bet_count"] for r in rows)
    pooled_roi = (total_return_units / total_bets * 100) if total_bets > 0 else 0.0
    return {
        "fold数": len(rows),
        "合計購入件数": total_bets,
        "合計的中件数": total_hits,
        "合計的中率(%)": round(total_hits / total_bets * 100, 2) if total_bets > 0 else 0.0,
        "プール合算ROI(%)": round(pooled_roi, 2),
    }


def run_rolling_walk_forward() -> None:
    logger.info("=== ローリングwalk-forward検証を開始します（複数fold・再学習あり） ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    raw_df = get_all_race_data(db_connector)
    if raw_df.empty:
        logger.error("データベースからデータを取得できませんでした。")
        return

    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3, elo_k_factor=16.0)
    bias_fe = TrackBiasFeatureExtractor()

    featured_df = race_fe.transform(raw_df)
    featured_df = horse_fe.transform(featured_df)
    featured_df = bias_fe.transform(featured_df)

    unique_dates = sorted(featured_df["race_date"].unique())

    all_results = []
    for i, (tf, vf, tef) in enumerate(FOLD_FRACTIONS, 1):
        result = run_fold(i, featured_df, unique_dates, tf, vf, tef)
        all_results.append(result)

    print("\n" + "=" * 90)
    print(" 【ローリングwalk-forward検証: fold別結果】")
    print("=" * 90)

    place_rows = []
    win_rows = []
    wide_rows = []
    for r in all_results:
        row = {"fold": r["fold"], "test_period": r["test_period"], "rule": r["place_rule"]}
        if r["place"]:
            row.update(r["place"])
        place_rows.append(row)

        wrow = {"fold": r["fold"], "test_period": r["test_period"], "rule": r["win_rule"]}
        if r["win"]:
            wrow.update(r["win"])
        win_rows.append(wrow)

        wide_row = {"fold": r["fold"], "test_period": r["test_period"], "rule": r["wide_rule"]}
        if r["wide"]:
            wide_row.update(r["wide"])
        wide_rows.append(wide_row)

    print("\n[複勝: fold別のTest期間 out-of-sample結果]")
    print(tabulate(pd.DataFrame(place_rows), headers="keys", tablefmt="github", showindex=False))
    print("\n[複勝: 3fold合算（賭け金ベースでプール）]")
    print(tabulate([_pooled_roi(all_results, "place")], headers="keys", tablefmt="github", showindex=False))

    print("\n[単勝: fold別のTest期間 out-of-sample結果]")
    print(tabulate(pd.DataFrame(win_rows), headers="keys", tablefmt="github", showindex=False))
    print("\n[単勝: 3fold合算（賭け金ベースでプール）]")
    print(tabulate([_pooled_roi(all_results, "win")], headers="keys", tablefmt="github", showindex=False))

    print("\n[ワイド: fold別のTest期間 out-of-sample結果]")
    print(tabulate(pd.DataFrame(wide_rows), headers="keys", tablefmt="github", showindex=False))
    print("\n[ワイド: 3fold合算（賭け金ベースでプール）]")
    print(tabulate([_pooled_roi(all_results, "wide")], headers="keys", tablefmt="github", showindex=False))

    print(
        "\n※ 各foldは独立して train を拡張窓で再学習し、val窓のみで閾値選定、test窓には一度だけ"
        "適用している。fold間でルール（rule列）が変わるのは、各foldのvalデータで再選定している"
        "ため（同一モデル・同一ルールを固定して複数期間評価しているわけではない点に注意）。"
    )


if __name__ == "__main__":
    run_rolling_walk_forward()
