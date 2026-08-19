import numpy as np
import pandas as pd


class TrackBiasFeatureExtractor:
    """
    開催日・競馬場・トラック種別ごとの当日トラックバイアス（内外差・脚質有利不利）を
    直前レースまでの結果から累積計算する特徴量抽出クラス（リーク完全防止）
    """

    def __init__(self):
        pass

    def _extract_last_corner_order(self, passage_order_str: str) -> float:
        """通過順文字列（例: '2-2-3-4'）から最終コーナー（4角）の順位を抽出"""
        if pd.isnull(passage_order_str) or not str(passage_order_str).strip():
            return np.nan
        parts = str(passage_order_str).split("-")
        try:
            return float(parts[-1])
        except (ValueError, IndexError):
            return np.nan

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "venue_code" not in df.columns:
            df["venue_code"] = df["race_id"].astype(str).str[4:6]

        # 4角通過順の抽出とレース内パーセンタイル
        df["_last_corner_pos"] = df["passage_order"].apply(self._extract_last_corner_order)
        # レース内の出走頭数で正規化 (0.0 = 先頭, 1.0 = 最後方)
        horse_counts = df.groupby("race_id")["horse_num"].transform("count")
        df["_last_corner_pct"] = df["_last_corner_pos"] / horse_counts.replace(0, np.nan)

        df["_is_top3"] = df["rank"].apply(lambda r: 1 if pd.notnull(r) and 1 <= r <= 3 else 0)
        df["_is_inner"] = df["bracket_num"].apply(lambda b: 1 if pd.notnull(b) and b <= 4 else 0)

        # レース単位で集計するためのキー
        # (race_date, venue_code, course_type, race_round, race_id)
        race_summary = df.groupby(
            ["race_date", "venue_code", "course_type", "race_round", "race_id"],
            as_index=False
        ).agg(
            inner_top3_count=pd.NamedAgg(column="_is_inner", aggfunc=lambda s: ((s == 1) & (df.loc[s.index, "_is_top3"] == 1)).sum()),
            inner_total_count=pd.NamedAgg(column="_is_inner", aggfunc=lambda s: (s == 1).sum()),
            outer_top3_count=pd.NamedAgg(column="_is_inner", aggfunc=lambda s: ((s == 0) & (df.loc[s.index, "_is_top3"] == 1)).sum()),
            outer_total_count=pd.NamedAgg(column="_is_inner", aggfunc=lambda s: (s == 0).sum()),
            top3_avg_corner_pct=pd.NamedAgg(
                column="_last_corner_pct",
                aggfunc=lambda s: s[df.loc[s.index, "_is_top3"] == 1].mean()
            )
        ).sort_values(["race_date", "venue_code", "course_type", "race_round"])

        # 同日・同競馬場・同コース種別内で、直前レースまでの累積値を算出 (shift して当日レースを除外)
        group_keys = ["race_date", "venue_code", "course_type"]
        
        # 内枠・外枠の累積好走数と出走数
        race_summary["cum_inner_top3"] = race_summary.groupby(group_keys)["inner_top3_count"].cumsum() - race_summary["inner_top3_count"]
        race_summary["cum_inner_total"] = race_summary.groupby(group_keys)["inner_total_count"].cumsum() - race_summary["inner_total_count"]
        race_summary["cum_outer_top3"] = race_summary.groupby(group_keys)["outer_top3_count"].cumsum() - race_summary["outer_top3_count"]
        race_summary["cum_outer_total"] = race_summary.groupby(group_keys)["outer_total_count"].cumsum() - race_summary["outer_total_count"]

        # 4角通過順の累積平均
        race_summary["cum_corner_sum"] = (
            race_summary.groupby(group_keys)["top3_avg_corner_pct"].cumsum() - race_summary["top3_avg_corner_pct"].fillna(0)
        )
        race_summary["cum_race_count"] = race_summary.groupby(group_keys).cumcount()

        # 1. 内枠有利度スコア（累積内枠複勝率 - 累積外枠複勝率）
        # 事前分布（初期値0.0）へのベイズ平滑化
        inner_rate = (race_summary["cum_inner_top3"] + 3) / (race_summary["cum_inner_total"] + 10)
        outer_rate = (race_summary["cum_outer_top3"] + 3) / (race_summary["cum_outer_total"] + 10)
        race_summary["bias_inner_bracket_advantage"] = inner_rate - outer_rate

        # 2. 前残り有利度スコア（直前レースまでの3着以内馬の平均4角位置。中央値0.35程度が標準）
        # サンプルがない場合は中央値 0.35 で補完
        race_summary["bias_front_runner_advantage"] = np.where(
            race_summary["cum_race_count"] > 0,
            race_summary["cum_corner_sum"] / race_summary["cum_race_count"],
            0.35
        )
        race_summary["bias_front_runner_advantage"] = race_summary["bias_front_runner_advantage"].fillna(0.35)

        # 特徴量を元データにマージ
        bias_cols = ["race_id", "bias_inner_bracket_advantage", "bias_front_runner_advantage"]
        df = df.merge(race_summary[bias_cols], on="race_id", how="left")

        # 3. 各馬のトラックバイアス適合度スコア
        # 内枠馬 × 内枠有利度 + 先行馬（過去平均通過順が前）× 前残り有利度
        is_inner_sign = df["bracket_num"].apply(lambda b: 1.0 if pd.notnull(b) and b <= 4 else -1.0)
        
        # 過去の平均通過順（存在すれば利用、なければ中央値 0.5）
        avg_pass = df.get("horse_avg_passage_rate", 0.5).fillna(0.5)
        # avg_pass は小さいほど逃げ先行。bias_front_runner_advantage が小さい（前残り）ほど適合度プラス
        front_match = (0.5 - avg_pass) * (0.35 - df["bias_front_runner_advantage"]) * 4.0

        df["bias_horse_match_score"] = (is_inner_sign * df["bias_inner_bracket_advantage"]) + front_match

        # 一時カラムの削除
        drop_cols = ["_last_corner_pos", "_last_corner_pct", "_is_top3", "_is_inner"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df