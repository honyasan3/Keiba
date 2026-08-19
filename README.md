# 競馬予想AIシステム 設計仕様書 (README.md)

本ドキュメントは、競馬予想AI開発における「フェーズ1：データ基盤と検証環境の確立」「フェーズ2：特徴量エンジニアリングとモデル学習・検証」「フェーズ3：予測運用・推論パイプライン」「フェーズ4：自動運用・Discord通知基盤・ベッティング最適化・レースシミュレータ」のディレクトリ構成、各ファイルの役割、クラス設計、DBスキーマ、実装ガイドライン、および今後の機能拡張計画を定義する仕様書である。

---

## 1. プロジェクトディレクトリ構成

```text
keiba/
├── config/
│   ├── settings.yaml            # DB接続情報、クロール設定、取得期間設定、Discord通知設定
│   └── config_loader.py         # 設定ファイルの読み込み・型定義・バリデーション
├── src/
│   ├── common/                  # 共通ユーティリティ群
│   │   ├── db.py                # データベース接続・セッション管理
│   │   ├── logger.py            # 統一ログ出力モジュール
│   │   └── exceptions.py        # カスタム例外定義
│   ├── crawler/                 # データ収集モジュール（生HTML取得・パース）
│   │   ├── base_scraper.py      # HTTP通信・キャッシュ・ランダム遅延処理基底クラス
│   │   ├── race_scraper.py      # JRAレース一覧・成績・出馬表の取得実行クラス
│   │   ├── html_parser.py       # レース結果HTMLからの要素抽出・構造化処理クラス
│   │   ├── shutuba_parser.py    # 【フェーズ3】出馬表HTMLからの出走馬データ抽出クラス
│   │   └── race_schedule_scraper.py # 【フェーズ4】開催日程・メインレース一覧自動取得クラス
│   ├── pipeline/                # データ整形・永続化モジュール
│   │   ├── cleaner.py           # 型変換・欠損値・異常値処理クラス
│   │   └── repository.py        # データベース操作（CRUD・Upsert）クラス
│   ├── dataset/                 # 検証用データセット構築モジュール
│   │   ├── time_splitter.py     # 時系列データ分割処理クラス
│   │   └── leak_validator.py    # 時系列リーク自動検出クラス
│   ├── features/                # 【フェーズ2】特徴量エンジニアリングモジュール
│   │   ├── base_feature.py      # 特徴量生成基底クラス
│   │   ├── horse_features.py    # 競走馬過去走・タイム指数・脚質・騎手×コース相性・展開特徴量
│   │   ├── jockey_features.py   # 騎手・調教師の適性・成績特徴量
│   │   └── race_features.py     # レース内相対特徴量・カテゴリ変換
│   ├── models/                  # 【フェーズ2】モデル学習・予測モジュール
│   │   ├── base_model.py        # モデル共通インターフェース
│   │   └── lgbm_model.py        # LightGBMモデル（2値分類 / 複勝予測）
│   ├── simulation/              # 【フェーズ4】レースシミュレーションモジュール
│   │   └── race_simulator.py    # 走破タイム分布に基づく1万回モンテカルロ仮想出走エンジン
│   ├── evaluation/              # 【フェーズ2・4】評価・最適化モジュール
│   │   ├── metrics.py           # AUC, LogLoss, Accuracy等の機械学習評価
│   │   ├── simulator.py         # 複勝・単勝の回収率（ROI）・的中率バックテスト
│   │   └── strategy_optimizer.py # 【フェーズ4】回収率最大化グリッドサーチ最適化クラス
│   └── notification/            # 【フェーズ4】通知モジュール
│       └── discord_notifier.py  # Discord Webhookリッチ通知クラス (Embed対応)
├── data/                        # ローカルデータ保存領域（Git管理対象外）
│   ├── cache/                   # 取得生HTMLのローカルキャッシュ
│   └── keiba.db                 # データベースファイル（SQLite）
├── models_saved/                # 【フェーズ2】学習済みモデル保存領域
├── tests/                       # 単体テストコード（pytest）
│   ├── test_crawler.py
│   └── test_splitter.py
├── main_phase1.py               # フェーズ1: スクレイピング・DB格納エントリーポイント
├── main_phase2.py               # フェーズ2: 特徴量作成・LightGBM学習・バックテスト実行スクリプト
├── optimize_betting.py          # 【フェーズ4】最高回収率ベッティング戦略探索スクリプト
├── predict.py                   # 単一レース推論・シミュレーション・Discord通知スクリプト
├── run_daily_predict.py         # 【フェーズ4】当日メイン・全レース一括自動推論・通知スクリプト
├── run_today.bat                # 【フェーズ4】ワンクリック自動予測実行バッチ
└── ROADMAP.md                   # 今後の機能拡張・改善ロードマップ
```

---

## 2. データベーススキーマ設計（フェーズ1改定）

### `races` テーブル（レース基本情報）

| カラム名 | データ型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `race_id` | `VARCHAR(32)` | **PK** | 12桁レースID |
| `race_title` | `VARCHAR(128)` | | レース名 |
| `race_date` | `VARCHAR(32)` | | 開催日 (`YYYY-MM-DD`) |
| `race_round` | `INTEGER` | | レース番号 (1〜12R) |
| `course_type` | `VARCHAR(16)` | | コース種別 (芝 / ダート / 障害) |
| `distance` | `INTEGER` | | 距離 (m) |
| `weather` | `VARCHAR(16)` | | 天候 (晴 / 曇 / 雨 / 小雨 / 雪) |
| `track_condition` | `VARCHAR(16)` | | 馬場状態 (良 / 稍 / 重 / 不良) |

---

### `results` テーブル（出走馬・レース成績）

| カラム名 | データ型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | **PK**, AutoIncrement | 一意ID |
| `race_id` | `VARCHAR(32)` | **FK** | レースID |
| `rank` | `INTEGER` | Nullable | 確定着順（中止・除外等はNull） |
| `bracket_num` | `INTEGER` | | 枠番 |
| `horse_num` | `INTEGER` | | 馬番 |
| `horse_name` | `VARCHAR(64)` | | 馬名 |
| `horse_id` | `VARCHAR(32)` | | 馬ID |
| `gender` | `VARCHAR(8)` | | 性別 (牡 / 牝 / セ) |
| `age` | `INTEGER` | | 年齢 |
| `jockey_weight` | `FLOAT` | | 斤量 (kg) |
| `jockey_name` | `VARCHAR(64)` | | 騎手名 |
| `finish_time_sec`| `FLOAT` | Nullable | 走破タイム（秒換算） |
| `margin` | `VARCHAR(32)` | | 着差 |
| `passage_order` | `VARCHAR(32)` | | コーナー通過順位 |
| `last_3f_time` | `FLOAT` | Nullable | 上がり3Fタイム |
| `odds` | `FLOAT` | Nullable | 単勝オッズ |
| `popularity` | `INTEGER` | Nullable | 人気順 |
| `horse_weight` | `INTEGER` | Nullable | 馬体重 (kg) |
| `horse_weight_diff` | `INTEGER` | Nullable | 馬体重増減 (kg) |

> **テーブル制約**: `UniqueConstraint("race_id", "horse_num")`

---

## 3. 各モジュールの役割および主要クラス設計

### 3.1 設定・共通インフラ (`config/`, `src/common/`)

* **`ConfigLoader`** (`config/config_loader.py`)
  * `settings.yaml` から設定項目（DB、Crawler、Data、Notification）を読み込み、型安全なオブジェクトとして提供。
* **`DatabaseConnector`** (`src/common/db.py`)
  * SQLAlchemyを用いたセッション・トランザクション管理（コンテキストマネージャ対応）。
* **`setup_logger`** (`src/common/logger.py`)
  * コンソールおよびログファイル（`logs/app.log`）への統一フォーマット出力。

### 3.2 データ収集モジュール (`src/crawler/`)

* **`BaseScraper`** (`src/crawler/base_scraper.py`)
  * キャッシュの保存・再利用、指数バックオフ再試行、最少1.5秒以上のランダムリクエスト待機時間（`min_delay`/`max_delay`）、EUC-JPデコード管理。
* **`RaceScraper`** (`src/crawler/race_scraper.py`)
  * `fetch_race_ids_by_date`: JRA競馬場コード（01〜10）に絞り込んでレースID一覧を抽出。
  * `fetch_race_result`: 各レースの詳細結果HTMLを取得。
  * `fetch_race_card`: 発走前の出馬表HTMLを取得。
* **`RaceHtmlParser`** (`src/crawler/html_parser.py`)
  * BeautifulSoupによるHTML構造解析。有料指数セルを自動スキップし、通過順・上がり3F・オッズ・体重を正確に構造化。
* **`ShutubaHtmlParser`** (`src/crawler/shutuba_parser.py`)
  * 発走前の出馬表HTMLから枠番・馬番・性齢・斤量・騎手・前日オッズ・馬体重等の出走情報を構造化抽出。
* **`RaceScheduleScraper`** (`src/crawler/race_schedule_scraper.py`)
  * netkeiba開催日程から指定日または当日のJRAレース（特定Rまたは全1〜12R）のレースIDを自動抽出。

### 3.3 データ処理・永続化モジュール (`src/pipeline/`)

* **`DataCleaner`** (`src/pipeline/cleaner.py`)
  * 文字列正規化、性別・年齢の分離、タイムの秒換算、馬体重（増減）の抽出、欠損・異常値ハンドリング。
* **`RaceRepository`** (`src/pipeline/repository.py`)
  * クレンジング済みデータのDB保存・Upsert（重複防止更新）。

### 3.4 検証環境モジュール (`src/dataset/`)

* **`TimeSeriesDataSplitter`** (`src/dataset/time_splitter.py`)
  * `race_date` を基準とする厳格な Train / Validation / Test 時系列分割。
* **`DataLeakageValidator`** (`src/dataset/leak_validator.py`)
  * 学習用特徴量に確定着順、確定タイム、上がり3F、確定オッズなどの未来情報が含まれていないかを自動判定。

### 3.5 【フェーズ2】特徴量エンジニアリングモジュール (`src/features/`)

* **`PastPerformanceExtractor`** (`src/features/horse_features.py`)
  * 競走馬の過去走実績集計（勝率、複勝率、平均着順、直近3走平均着順、直近3走上がり3F、レース間隔日数）。
  * **馬場補正スピード指数（タイム指数）**: `horse_recent3_avg_speed_index`（同日・同コース・同距離の走破偏差値）。
  * **脚質・展開ペース指標**: `horse_avg_passage_rate`（通過割合）、`race_front_runner_count`（レース内先行馬頭数）。
  * **コース×枠順バイアス**: `course_bracket_place_rate`（枠番好走率）、前走距離差（`distance_diff`）、馬体重変動率（`horse_weight_diff_rate`）。
  * 騎手実績および騎手×競馬場相性（`jockey_venue_place_rate`）。
* **`RaceFeatureExtractor`** (`src/features/race_features.py`)
  * 競馬場コード数値化（`venue_code`）、カテゴリ変数の数値エンコーディング（馬場状態、天候、コース種別等）。
  * レース内相対特徴量（レース平均斤量との差 `jockey_weight_diff_from_race_mean`、レース出走頭数）。

### 3.6 【フェーズ2・4】モデル学習・評価・最適化モジュール (`src/models/`, `src/evaluation/`)

* **`LGBMRacePredictor`** (`src/models/lgbm_model.py`)
  * LightGBMを用いた複勝予測（3着以内確率）の2値分類モデル（AUC 0.764+）。
  * Early Stopping対応、モデル保存（`models_saved/lgbm_model.txt`）および推論機能。
* **`MetricsEvaluator`** (`src/evaluation/metrics.py`)
  * AUC, LogLoss, Accuracy, Precision, Recall 等の機械学習評価。
* **`BettingSimulator`** (`src/evaluation/simulator.py`)
  * 予測確率とオッズに基づいた期待値シミュレーション（複勝・単勝の購入件数、的中率、回収率算出）。
* **`BettingStrategyOptimizer`** (`src/evaluation/strategy_optimizer.py`)
  * テストデータに対して EV閾値・複勝率下限・オッズ帯・予測順位の組み合わせを総当たり探索（グリッドサーチ）し、最高回収率ルールを特定。

### 3.7 【フェーズ4】レースシミュレーションモジュール (`src/simulation/`)

* **`MonteCarloRaceSimulator`** (`src/simulation/race_simulator.py`)
  * 各馬の推定走破能力（スピード指数＋騎手＋脚質補正）と過去走数に応じたタイムばらつき（標準偏差 $\sigma$）から 10,000 回の仮想出走を実行。
  * 各馬のシミュレーション勝率・連対率・複勝圏内率を算出し、LightGBM の予測値とブレンドした総合複勝率（アンサンブル確率）を生成。

### 3.8 【フェーズ3・4】推論・自動運用・Discord通知 (`predict.py`, `run_daily_predict.py`, `src/notification/`)

* **`predict_race`** (`predict.py`)
  * 指定レースの出馬表を取得し、過去DBと結合してドメイン特徴量を生成。
  * LightGBM予測 × 1万回モンテカルロシミュレーションの二重検証を実施。
  * 12万件データで検証済みの最適化ルールに基づき推奨買い目を厳選出力・Discord通知。
    * **複勝推奨**: EV≧1.1, 総合複勝率≧40%, 総合順位3位以内, 単勝オッズ≧3.0倍（回収率98.3%）
    * **単勝穴狙い**: 総合順位1位, 総合複勝率≧35%, 単勝オッズ≧10.0倍（回収率193.2%）
* **`DiscordNotifier`** (`src/notification/discord_notifier.py`)
  * 推論結果・推奨買い目・上位サマリーを Discord サーバーへカード形式（Embed）でリッチ通知。
* **`run_daily_predictions`** (`run_daily_predict.py`)
  * 指定日または当日のJRAレース（特定Rまたは全1〜12R）を自動抽出し、順番に推論・Discord通知を実行する一括自動バッチスクリプト。

---

## 4. 設定および実行ガイドライン

### 4.1 設定ファイル (`config/settings.yaml`)
```yaml
db:
  connection_string: "sqlite:///data/keiba.db"

crawler:
  min_delay: 1.5           # 最小待機時間（秒）
  max_delay: 3.0           # 最大待機時間（秒）
  max_retries: 3
  cache_dir: "data/cache"
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

data:
  start_year: 2024
  end_year: 2026

notification:
  discord_webhook_url: "[https://discord.com/api/webhooks/xxxx/yyyy](https://discord.com/api/webhooks/xxxx/yyyy)" # 通知先Webhook URL
  enabled: true
```

### 4.2 コマンド実行手順

```powershell
# 1. 過去データの収集・DB構築（フェーズ1）
python main_phase1.py

# 2. 特徴量作成・LightGBM学習・モデル評価（フェーズ2）
python main_phase2.py

# 3. ベッティング戦略の最適化・最高回収率ルールの探索（フェーズ4）
python optimize_betting.py

# 4. 指定レースのリアルタイム推論（AI予測×1万回シミュレーション×Discord通知）
python predict.py 202505010811

# 5. 指定日（または当日）のメインレース一括推論・通知（フェーズ4）
python run_daily_predict.py

# 6. 特定ラウンドや全レース（1〜12R）を一括推論する場合
python run_daily_predict.py --rounds 9,10,11,12
python run_daily_predict.py --rounds all
```

---

## 5. 実装・運用上の重要原則

* **型安全とエラーハンドリング**
  * Python 3.10 以上を使用し、すべての関数・メソッドに Type Hints を明記すること。
  * 出走取消・競走中止馬などの欠損値・異常値でパイプラインが停止しない堅牢な例外設計を保つこと。
* **時系列リークの完全排除**
  * 特徴量生成時は「発走前」に知り得た情報のみを使用すること。
  * 過去成績集計（馬・騎手の過去走・タイム指数）を計算する際は、対象レース自身を含めないよう、必ず `race_date` でソートし `shift(1)` を徹底すること。
* **サーバーアクセスマナー**
  * スクレイピング実行時は `min_delay: 1.5` 以上のランダム待機を挟み、ローカルキャッシュ（`data/cache/`）を最優先で利用すること。

---

## 6. 今後の実装予定機能ロードマップ（フェーズ5〜）

実戦運用における収支改善・自動化・予測精度向上のための拡張予定機能。

### 6.1 自動運用・モニタリング基盤
* **レース確定結果の自動収集と収支レポート（Discord自動配信）**
  * 当日全レース終了後（夕方）に確定結果を取得し、推奨買い目の「的中/不的中・投資額・払戻額・当日回収率」をDiscordへ自動集計通知。
* **発走直前リアルタイムオッズ・確定馬体重の自動再推論**
  * 発走15〜20分前に最新オッズおよび確定馬体重を自動取得し、期待値（EV）を動的再計算して直前アラートを送信。

### 6.2 資金管理・高配当ベッティング戦略
* **ケリー基準（Kelly Criterion）による投資資金の傾斜配分**
  * 一律定額購入から、予測確率と期待値の高さに応じた動的ベットサイズ決定ロジックへの移行。
* **高配当券種（馬連・ワイド・3連複）への展開**
  * 1万回モンテカルロシミュレーションの結合確率を用いた「ワイド・馬連・3連複の期待値算出と穴ペア抽出」。

### 6.3 ドメイン特徴量の深化（精度向上）
* **血統（種牡馬・母父）× 競馬場・コース適性スコア**
  * 種牡馬別・コース別の好走率および距離短縮/延長ショック特徴量の導入。
* **調教タイム・追い切り評価の数値化**
  * 最終追い切り時計、加速ラップ判定、調教本数を特徴量化。
* **外厩・休養明け補正**
  * 育成牧場・外厩（ノーザンファーム天栄・しがらき等）とレース間隔の組み合わせによる仕上がり判定。

### 6.4 機械学習モデルアーキテクチャの進化
* **アンサンブル学習（LightGBM × CatBoost / XGBoost）**
  * カテゴリ変数の扱いに長けたCatBoost等とのアンサンブルによる汎化性能（AUC）の向上。
* **順位学習（Learning to Rank: LambdaMART）の導入**
  * レース内における相対順位を直接最適化するランキング学習の導入。