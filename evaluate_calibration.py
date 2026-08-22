"""モデル確率較正の診断スクリプト（ベッティング判断層とモデル品質の切り分け）

複勝/単勝の推奨買い目は EV = pred_place_prob × odds で判定しており、この判断が成立するためには
「モデルが45%と言ったら実際に約45%的中する」という確率較正が前提になる。walk-forward検証で
EVしきい値戦略が損失（複勝ROI 86.00%、単勝ROI 85.30%）だったことを受け、これが
(a) モデルの確率推定自体のズレ（較正不良）によるものか
(b) 確率は概ね正しいがしきい値・戦略設計側の問題か
を切り分けるために、Validation/Test期間それぞれで確率較正（reliability diagram相当）とAUC/LogLoss/
Brierスコアを計測する。AUCはランキング能力のみを見る指標で較正のズレを検出できないため、
較正表と併読すること。
"""
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.ensemble_runner import run_ensemble_inference
from src.evaluation.metrics import MetricsEvaluator
from src.evaluation.strategy_optimizer import BettingStrategyOptimizer
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("evaluate_calibration")

# 現行の複勝/単勝しきい値戦略が実際に判断に使っている確率帯（walk-forward検証で選定された値）
DECISION_RELEVANT_BINS = [0.0, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 1.0]


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


def _report_period(name: str, df: pd.DataFrame) -> None:
    df_eval = df.dropna(subset=["rank", "pred_place_prob"]).copy()
    df_eval["target_place"] = (df_eval["rank"] <= 3).astype(int)

    metrics = MetricsEvaluator.calculate_all_metrics(df_eval["target_place"], df_eval["pred_place_prob"])
    brier = MetricsEvaluator.brier_score(df_eval["target_place"], df_eval["pred_place_prob"])

    print(f"\n{'=' * 70}")
    print(f" 【{name}】 n={len(df_eval)}  AUC={metrics['auc']}  LogLoss={metrics['logloss']}  Brier={brier}")
    print("=" * 70)

    print("\n[全体較正（10等分ビン）: mean_pred(モデルの平均予測) vs actual_rate(実際の的中率)]")
    report_10 = MetricsEvaluator.calibration_report(df_eval["target_place"], df_eval["pred_place_prob"])
    print(tabulate(report_10, headers="keys", tablefmt="github", showindex=False))

    print("\n[判断関連ビン: 現行しきい値(0.38/0.42/0.45等)付近を細かく分割]")
    report_fine = MetricsEvaluator.calibration_report(
        df_eval["target_place"], df_eval["pred_place_prob"], bin_edges=DECISION_RELEVANT_BINS
    )
    print(tabulate(report_fine, headers="keys", tablefmt="github", showindex=False))

    # オッズ帯別の較正（本命 vs 穴馬でズレ方が違うか）
    df_odds = df_eval.dropna(subset=["odds"]).copy()
    if not df_odds.empty:
        df_odds["odds_band"] = pd.cut(
            df_odds["odds"],
            bins=[0, 1.5, 3.0, 5.0, 10.0, 9999],
            labels=["~1.5倍", "1.5~3倍", "3~5倍", "5~10倍", "10倍~"],
        )
        odds_report = df_odds.groupby("odds_band", observed=True).apply(
            lambda g: pd.Series({
                "count": len(g),
                "mean_pred": round(g["pred_place_prob"].mean(), 4),
                "actual_rate": round(g["target_place"].mean(), 4),
                "gap": round(g["pred_place_prob"].mean() - g["target_place"].mean(), 4),
            }),
            include_groups=False,
        ).reset_index()
        print("\n[オッズ帯別較正: 本命(低オッズ)と穴馬(高オッズ)でズレ方が異なるか]")
        print(tabulate(odds_report, headers="keys", tablefmt="github", showindex=False))


def fit_isotonic_calibrator(val_df: pd.DataFrame) -> IsotonicRegression:
    """Validation期間のみを使い、pred_place_prob -> 実際の的中確率 のIsotonic回帰較正器を学習する"""
    val_eval = val_df.dropna(subset=["rank", "pred_place_prob"]).copy()
    val_eval["target_place"] = (val_eval["rank"] <= 3).astype(int)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_eval["pred_place_prob"], val_eval["target_place"])
    return calibrator


def run_calibrated_walk_forward(val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validationで学習した較正器をpred_place_probに適用した上で、walk-forwardの買い目戦略検証をやり直す

    Isotonic回帰は単調変換のため、レース内の予測順位(pred_rank)は較正前後で変化しない。
    """
    calibrator = fit_isotonic_calibrator(val_df)

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["pred_place_prob"] = calibrator.predict(val_df["pred_place_prob"])
    test_df["pred_place_prob"] = calibrator.predict(test_df["pred_place_prob"])

    print("\n" + "#" * 70)
    print("# 較正後のTest期間 較正表（改善したか確認）")
    print("#" * 70)
    _report_period("Test（較正後）", test_df)

    optimizer = BettingStrategyOptimizer()

    print("\n" + "#" * 70)
    print("# 較正後の確率でwalk-forwardベッティング戦略を再検証")
    print("#" * 70)

    place_val_results = optimizer.optimize_place_strategy(val_df)
    print("\n【複勝 最適戦略 Top 5（Validation期間・較正後で選定）】")
    if not place_val_results.empty:
        print(tabulate(place_val_results.head(5), headers="keys", tablefmt="github", showindex=False))

        best_place = place_val_results.iloc[0]
        test_place_eval = BettingStrategyOptimizer.prep_place_eval_df(test_df)
        place_test_result = BettingStrategyOptimizer.evaluate_place_strategy(
            test_place_eval,
            ev_th=best_place["ev_threshold"],
            min_p=best_place["min_prob"],
            max_r=int(best_place["max_rank"]),
            odds_range=(best_place["min_odds"], best_place["max_odds"]),
        )
        print("\n【複勝 較正後ルールをTest期間に適用した結果（out-of-sample）】")
        print(tabulate([place_test_result], headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")

    win_val_results = optimizer.optimize_win_strategy(val_df)
    print("\n【単勝 最適戦略 Top 5（Validation期間・較正後で選定）】")
    if not win_val_results.empty:
        print(tabulate(win_val_results.head(5), headers="keys", tablefmt="github", showindex=False))

        best_win = win_val_results.iloc[0]
        test_win_eval = BettingStrategyOptimizer.prep_win_eval_df(test_df)
        win_test_result = BettingStrategyOptimizer.evaluate_win_strategy(
            test_win_eval,
            ev_th=best_win["ev_threshold"],
            min_p=best_win["min_prob"],
            max_r=int(best_win["max_rank"]),
            odds_range=(best_win["min_odds"], best_win["max_odds"]),
        )
        print("\n【単勝 較正後ルールをTest期間に適用した結果（out-of-sample）】")
        print(tabulate([win_test_result], headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")


def run_calibration_check() -> None:
    logger.info("=== 確率較正チェックを開始します ===")
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
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)
    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    splitter = TimeSeriesDataSplitter()
    train_df, val_df, test_df = splitter.split_by_date(featured_df, train_end=train_end, val_end=val_end)
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df = run_ensemble_inference(train_df)
    val_df = run_ensemble_inference(val_df)
    test_df = run_ensemble_inference(test_df)

    _report_period(f"Train ({train_df['race_date'].min()} ~ {train_df['race_date'].max()})", train_df)
    _report_period(f"Validation ({val_df['race_date'].min()} ~ {val_df['race_date'].max()})", val_df)
    _report_period(f"Test ({test_df['race_date'].min()} ~ {test_df['race_date'].max()})", test_df)

    run_calibrated_walk_forward(val_df, test_df)


if __name__ == "__main__":
    run_calibration_check()
