"""ベッティング戦略最適化スクリプト（全46特徴量: トラックバイアス & Elo & トリプルアンサンブル統合版）

【walk-forward検証】
以前の実装は、グリッドサーチによる戦略選定と、その戦略の性能評価を同一のテスト期間データに対して
行っており、報告されるROIが選定に使ったデータへの評価という点で厳密なout-of-sampleになっていなかった
（このスクリプトのグリッドサーチ結果を backtest_simulation.py がそのまま同じテスト期間に対して
再評価していたため、事実上「一番儲かるルールを探して、それを同じデータで採点する」形になっていた）。

本スクリプトでは、閾値の選定を Validation 期間のみで行い、選ばれた戦略を一度も選定に使っていない
Test 期間に適用して初めて成績を報告する、時系列を跨いだ walk-forward 検証に改めている。
"""
import pandas as pd
from tabulate import tabulate

from config.config_loader import ConfigLoader
from src.common.db import DatabaseConnector
from src.common.logger import setup_logger
from src.dataset.time_splitter import TimeSeriesDataSplitter
from src.evaluation.ensemble_runner import run_ensemble_inference
from src.evaluation.strategy_optimizer import BettingStrategyOptimizer
from src.features.horse_features import PastPerformanceExtractor
from src.features.race_features import RaceFeatureExtractor
from src.features.track_bias_features import TrackBiasFeatureExtractor
from src.pipeline.repository import RaceModel, RaceResultModel

logger = setup_logger("optimize_betting")


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


def run_optimization() -> None:
    logger.info("=== 買い目戦略 walk-forward 最適化（トリプルアンサンブル & 46特徴量版）を開始します ===")
    config = ConfigLoader.load_config("config/settings.yaml")
    db_connector = DatabaseConnector(config.db.connection_string)

    raw_df = get_all_race_data(db_connector)
    if raw_df.empty:
        logger.error("データベースからデータを取得できませんでした。")
        return

    # 1. 特徴量生成（ピュア能力・Eloレーティング・当日トラックバイアス）
    race_fe = RaceFeatureExtractor()
    horse_fe = PastPerformanceExtractor(recent_runs=3, elo_k_factor=16.0)
    bias_fe = TrackBiasFeatureExtractor()

    featured_df = race_fe.transform(raw_df)
    featured_df = horse_fe.transform(featured_df)
    featured_df = bias_fe.transform(featured_df)

    # 2. 時系列分割 (Train: 70% / Val: 15% / Test: 15%)
    unique_dates = sorted(featured_df["race_date"].unique())
    train_idx = int(len(unique_dates) * 0.70)
    val_idx = int(len(unique_dates) * 0.85)

    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    splitter = TimeSeriesDataSplitter()
    _, val_df, test_df = splitter.split_by_date(
        featured_df, train_end=train_end, val_end=val_end
    )
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    logger.info(f"Validation期間: {val_df['race_date'].min()} ~ {val_df['race_date'].max()}（戦略選定用）")
    logger.info(f"Test期間: {test_df['race_date'].min()} ~ {test_df['race_date'].max()}（out-of-sample評価用、選定には一切使用しない）")

    # 3. トリプルアンサンブル推論（Val/Testそれぞれ独立に実行）
    val_df = run_ensemble_inference(val_df)
    test_df = run_ensemble_inference(test_df)

    optimizer = BettingStrategyOptimizer()

    # === 複勝戦略 ===
    logger.info("=== 複勝ベッティングルールの最適化中（Validation期間のみを使用） ===")
    place_val_results = optimizer.optimize_place_strategy(val_df)
    print("\n【複勝 最適戦略 Top 5（Validation期間で選定）】")
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
        print("\n【複勝 Validation選定ルールをTest期間に適用した結果（out-of-sample）】")
        print(tabulate([place_test_result], headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")

    # === 単勝戦略 ===
    logger.info("=== 単勝ベッティングルールの最適化中（Validation期間のみを使用） ===")
    win_val_results = optimizer.optimize_win_strategy(val_df)
    print("\n【単勝 最適戦略 Top 5（Validation期間で選定）】")
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
        print("\n【単勝 Validation選定ルールをTest期間に適用した結果（out-of-sample）】")
        print(tabulate([win_test_result], headers="keys", tablefmt="github", showindex=False))
    else:
        print("条件を満たす戦略が見つかりませんでした。")

    print(
        "\n※ 上記「Validation選定ルールをTest期間に適用した結果」が、閾値選定に使っていない"
        "データに対する唯一の公正な回収率評価です。Validation期間のTop5表はあくまで選定過程の参考値であり、"
        "そのROIをそのまま実運用の期待値として扱わないでください。"
    )


if __name__ == "__main__":
    run_optimization()
