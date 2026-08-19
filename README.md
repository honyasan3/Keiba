# 競馬予想AIシステム 設計仕様書 (README.md)

本ドキュメントは、競馬予想AI開発における「フェーズ1：データ基盤と検証環境の確立」「フェーズ2：展開負荷・Elo対戦レーティング・順位学習（LambdaMART）を含むトリプルアンサンブル学習」「フェーズ3：予測運用・推論パイプライン」「フェーズ4：自動運用・Discord通知基盤・ベッティング最適化・レースシミュレータ」「フェーズ5・6：高度ドメイン特徴量・ケリー資金管理・自動収支精算」のディレクトリ構成、各ファイルの役割、クラス設計、DBスキーマ、実装ガイドライン、および今後の機能拡張計画を定義する仕様書である。

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
│   │   ├── shutuba_parser.py    # 出馬表HTMLからの出走馬データ抽出クラス
│   │   └── race_schedule_scraper.py # 開催日程・メインレース一覧自動取得クラス
│   ├── pipeline/                # データ整形・永続化モジュール
│   │   ├── cleaner.py           # 型変換・欠損値・異常値処理クラス
│   │   └── repository.py        # データベース操作（CRUD・Upsert）クラス
│   ├── dataset/                 # 検証用データセット構築モジュール
│   │   ├── time_splitter.py     # 時系列データ分割処理クラス
│   │   └── leak_validator.py    # 時系列リーク自動検出クラス
│   ├── features/                # 特徴量エンジニアリングモジュール
│   │   ├── base_feature.py      # 特徴量生成基底クラス
│   │   ├── horse_features.py    # 過去走・タイム指数・PCI・展開不利・休養・Elo対戦レーティング特徴量
│   │   ├── jockey_features.py   # 騎手・調教師の適性・成績特徴量
│   │   └── race_features.py     # レース内相対特徴量・カテゴリ変換・想定ペース判定
│   ├── models/                  # モデル学習・予測モジュール
│   │   ├── base_model.py        # モデル共通インターフェース
│   │   ├── lgbm_model.py        # LightGBMモデル（2値分類 / 複勝予測）
│   │   ├── catboost_model.py    # CatBoostモデル（カテゴリ変数ネイティブ対応）
│   │   └── ranker_model.py      # LambdaMARTモデル（順位学習 / Learning to Rank）
│   ├── simulation/              # レースシミュレーションモジュール
│   │   └── race_simulator.py    # 1万回モンテカルロ仮想出走・ワイド/馬連結合確率・ケリー資金配分エンジン
│   ├── evaluation/              # 評価・最適化・自動精算モジュール
│   │   ├── metrics.py           # AUC, LogLoss, Accuracy等の機械学習評価
│   │   ├── simulator.py         # 複勝・単勝の回収率（ROI）・的中率バックテスト
│   │   ├── strategy_optimizer.py # 回収率最大化グリッドサーチ最適化クラス
│   │   └── settlement_reporter.py # 確定レース結果・払戻金の自動取得および収支集計クラス
│   └── notification/            # 通知モジュール
│       └── discord_notifier.py  # Discord Webhookリッチ通知クラス (予想Embed・収支Embed対応)
├── data/                        # ローカルデータ保存領域（Git管理対象外）
│   ├── cache/                   # 取得生HTMLのローカルキャッシュ
│   └── keiba.db                 # データベースファイル（SQLite）
├── models_saved/                # 学習済みモデル保存領域
│   ├── lgbm_model.txt           # LightGBM学習済みモデル重み
│   ├── catboost_model.cbm       # CatBoost学習済みモデル重み
│   └── lambdarank_model.txt     # LambdaMART順位学習済みモデル重み
├── tests/                       # 単体テストコード（pytest）
│   ├── test_crawler.py
│   └── test_splitter.py
├── main_phase1.py               # フェーズ1: スクレイピング・DB格納エントリーポイント
├── main_phase2.py               # フェーズ2: 特徴量作成・トリプルアンサンブル学習・検証スクリプト
├── backtest_simulation.py       # 長期運用バックテスト（ケリー基準資金管理・月別収支検証）
├── optimize_betting.py          # 最高回収率ベッティング戦略探索スクリプト
├── predict.py                   # レース推論（トリプルアンサンブル × シミュレーション × Discord通知）
├── run_daily_predict.py         # 当日メイン・全レース一括自動推論・通知スクリプト
├── run_daily_settlement.py      # 当日レース結果の自動精算・Discord収支レポート送信スクリプト
├── run_today.bat                # ワンクリック自動予測実行バッチ
└── ROADMAP.md                   # 今後の機能拡張・改善ロードマップ
```

## 2. データベーススキーマ設計

### `races` テーブル（レース基本情報）

| **カラム名**          | **データ型**       | **制約** | **説明**                  |
| ----------------- | -------------- | ------ | ----------------------- |
| `race_id`         | `VARCHAR(32)`  | **PK** | 12桁レースID                |
| `race_title`      | `VARCHAR(128)` |        | レース名                    |
| `race_date`       | `VARCHAR(32)`  |        | 開催日 (`YYYY-MM-DD`)      |
| `race_round`      | `INTEGER`      |        | レース番号 (1〜12R)           |
| `course_type`     | `VARCHAR(16)`  |        | コース種別 (芝 / ダート / 障害)    |
| `distance`        | `INTEGER`      |        | 距離 (m)                  |
| `weather`         | `VARCHAR(16)`  |        | 天候 (晴 / 曇 / 雨 / 小雨 / 雪) |
| `track_condition` | `VARCHAR(16)`  |        | 馬場状態 (良 / 稍 / 重 / 不良)   |

### `results` テーブル（出走馬・レース成績）

| **カラム名**            | **データ型**      | **制約**                | **説明**            |
| ------------------- | ------------- | --------------------- | ----------------- |
| `id`                | `INTEGER`     | **PK**, AutoIncrement | 一意ID              |
| `race_id`           | `VARCHAR(32)` | **FK**                | レースID             |
| `rank`              | `INTEGER`     | Nullable              | 確定着順（中止・除外等はNull） |
| `bracket_num`       | `INTEGER`     |                       | 枠番                |
| `horse_num`         | `INTEGER`     |                       | 馬番                |
| `horse_name`        | `VARCHAR(64)` |                       | 馬名                |
| `horse_id`          | `VARCHAR(32)` |                       | 馬ID               |
| `gender`            | `VARCHAR(8)`  |                       | 性別 (牡 / 牝 / セ)    |
| `age`               | `INTEGER`     |                       | 年齢                |
| `jockey_weight`     | `FLOAT`       |                       | 斤量 (kg)           |
| `jockey_name`       | `VARCHAR(64)` |                       | 騎手名               |
| `finish_time_sec`   | `FLOAT`       | Nullable              | 走破タイム（秒換算）        |
| `margin`            | `VARCHAR(32)` |                       | 着差                |
| `passage_order`     | `VARCHAR(32)` |                       | コーナー通過順位          |
| `last_3f_time`      | `FLOAT`       | Nullable              | 上がり3Fタイム           |
| `odds`              | `FLOAT`       | Nullable              | 単勝オッズ             |
| `popularity`        | `INTEGER`     | Nullable              | 人気順               |
| `horse_weight`      | `INTEGER`     | Nullable              | 馬体重 (kg)          |
| `horse_weight_diff` | `INTEGER`     | Nullable              | 馬体重増減 (kg)        |

> **テーブル制約**: `UniqueConstraint("race_id", "horse_num")`

## 3. 各モジュールの役割および主要クラス設計

### 3.1 設定・共通インフラ (`config/`, `src/common/`)

- **`ConfigLoader`** (`config/config_loader.py`): `settings.yaml` から設定項目を読み込み、型安全なオブジェクトとして提供。
- **`DatabaseConnector`** (`src/common/db.py`): SQLAlchemyを用いたセッション・トランザクション管理。
- **`setup_logger`** (`src/common/logger.py`): コンソールおよびログファイル（`logs/app.log`）への統一フォーマット出力。

### 3.2 データ収集モジュール (`src/crawler/`)

- **`BaseScraper`** (`src/crawler/base_scraper.py`): キャッシュ保存・再利用、指数バックオフ再試行、最少1.5秒以上のランダム遅延待機。
- **`RaceScraper`** (`src/crawler/race_scraper.py`): レースID一覧、詳細結果HTML、発走前出馬表HTMLを取得。
- **`RaceHtmlParser`** (`src/crawler/html_parser.py`): 結果HTMLから通過順・上がり3F・オッズ・馬体重を構造化抽出。
- **`ShutubaHtmlParser`** (`src/crawler/shutuba_parser.py`): 出馬表HTMLから出走馬データを抽出。
- **`RaceScheduleScraper`** (`src/crawler/race_schedule_scraper.py`): 開催日程からレースID一覧を自動取得。

### 3.3 データ処理・検証モジュール (`src/pipeline/`, `src/dataset/`)

- **`DataCleaner`** (`src/pipeline/cleaner.py`): 文字列正規化、性齢分離、秒換算、馬体重増減処理。
- **`RaceRepository`** (`src/pipeline/repository.py`): クレンジング済みデータのDB保存・Upsert（重複防止更新）。
- **`TimeSeriesDataSplitter`** (`src/dataset/time_splitter.py`): `race_date` を基準とする厳格な Train / Validation / Test 時系列分割。
- **`DataLeakageValidator`** (`src/dataset/leak_validator.py`): 特徴量に未来情報が含まれていないかを自動判定。

### 3.4 特徴量エンジニアリングモジュール (`src/features/`)

- **`PastPerformanceExtractor`** (`src/features/horse_features.py`):
  - **過去実績集計**: 勝率、複勝率、平均着順、直近3走平均着順、直近3走上がり3F、レース間隔日数（すべて `shift(1)` でリーク完全排除）。
  - **馬場補正スピード指数**: `horse_recent3_avg_speed_index`（同日・同コース・同距離の走破偏差値）。
  - **【フェーズA】展開負荷・ラップペース指数 (PCI)**:
    - `horse_recent3_avg_pci`: 馬自身の前後半スピード比率（$PCI = \frac{v_{first}}{v_{last}} \times 100$）。
    - `prev_pace_disadvantage_front`: 前走ハイペース先行で潰れた馬の巻き返し検知フラグ。
    - `prev_pace_disadvantage_back`: 前走スロー後方で脚を余した馬の検知フラグ。
  - **【フェーズD】競走馬多頭数Eloレーティングエンジン**:
    - `horse_elo_rating`: 過去全レースの直接対決結果から時系列更新された競走馬の真の実力レート。
    - `race_elo_diff_from_mean`: 今回出走メンバー内での平均Eloレートとの差分（突出度）。
  - **休養・ローテーション・距離ショック**:
    - `distance_shock_cat`: 距離短縮(-1)・同距離(0)・延長(+1)。
    - `rest_category_cat` / `is_second_run_after_rest`: 連闘・適度・外厩・長期休養および叩き2戦目フラグ。
    - `is_jockey_changed`: 騎手乗り替わり判定。
    - `age_gender_cat`: 年齢×性別クロス適性。
  - **騎手・コース枠バイアス**: `jockey_past_place_rate`, `jockey_venue_place_rate`, `course_bracket_place_rate`, `horse_weight_diff_rate`。
- **`RaceFeatureExtractor`** (`src/features/race_features.py`):
  - `venue_code`、コース・天候・馬場状態のカテゴリエンコーディング、`jockey_weight_diff_from_race_mean`、`race_horse_count`。
  - `race_expected_pace_cat`（先行馬比率による想定ペース区分: ハイ/ミドル/スロー）、`pace_match_score`（想定ペースと脚質の適合度）。

### 3.5 機械学習モデルアーキテクチャ (`src/models/`)

- **`LGBMRacePredictor`** (`src/models/lgbm_model.py`): LightGBMを用いた2値分類（3着以内好走確率）モデル。
- **`CatBoostRacePredictor`** (`src/models/catboost_model.py`): カテゴリ変数をネイティブ処理する2値分類モデル。
- **`LGBMRankPredictor`** (`src/models/ranker_model.py`):
  - **【フェーズB】順位学習（Learning to Rank / LambdaMART）**:
  - 目的関数: `objective="lambdarank"`（評価指標: NDCG@1, @3, @5）。
  - 同一レース内での相対着順序列を直接最適化。
- **★ トリプルアンサンブル構成 (40:40:20 Blend)**:
  - $\text{Ensemble\_Prob} = (0.40 \times \text{LGBM}) + (0.40 \times \text{CatBoost}) + (0.20 \times \text{LambdaRank\_Percentile})$
  - ピュア能力43特徴量でテストセット **AUC: 0.7734** を達成。

### 3.6 シミュレーション・資金管理・バックテスト (`src/simulation/`, `src/evaluation/`)

- **`MonteCarloRaceSimulator`** (`src/simulation/race_simulator.py`):
  - 10,000回仮想出走による各馬の勝率・複勝率およびワイド結合確率算出。
- **フラクショナル・ケリー基準（Fractional Kelly Criterion）**:
  - $f^* = \frac{b \cdot p - q}{b} \times 0.15$ （破産確率を抑制した最適賭け金傾斜配分）。
- **`backtest_simulation.py`**:
  - 2026年3月〜8月のテスト期間（1,033レース）における長期運用検証。
  - **購入件数: 1,033件 / 的中率: 37.9% / 総投資額: 400,800円 / 総払戻額: 473,174円 / 純利益: ＋72,374円 / 最終回収率 (ROI): 【118.06%】**。

### 3.7 自動運用・収支精算・Discord通知 (`predict.py`, `run_daily_predict.py`, `run_daily_settlement.py`, `src/notification/`, `src/evaluation/`)

- **`predict.py`**: 出馬表取得 → 43特徴量生成（Elo算出含む） → トリプルアンサンブル推論 → 1万回シミュレーション → ケリー推奨額付き買い目抽出 → Discord通知。
- **`SettlementReporter`** (`src/evaluation/settlement_reporter.py`): レース確定結果・公式払戻金の自動取得および収支突合精算。
- **`DiscordNotifier`** (`src/notification/discord_notifier.py`): 予想レポートおよび確定収支レポートのリッチEmbed通知。

## 4. 設定および実行ガイドライン

### 4.1 設定ファイル (`config/settings.yaml`)

YAML

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
  discord_webhook_url: "https://discord.com/api/webhooks/xxxx/yyyy"
  enabled: true
```

### 4.2 コマンド実行手順

PowerShell

```powershell
# 1. 過去データの収集・DB構築（フェーズ1）
python main_phase1.py

# 2. 43特徴量作成・トリプルアンサンブル学習
#    (LGBM × CatBoost × LambdaMART)・モデル保存（フェーズ2）
python main_phase2.py

# 3. 長期運用バックテスト
#    （Eloレーティング＆ケリー基準収支シミュレーション）
python backtest_simulation.py

# 4. 指定レースのリアルタイム推論
#    （トリプルアンサンブル×1万回シミュレーション×ケリー資金配分×Discord通知）
python predict.py 202605020811

# 5. 指定日（または当日）のメインレース一括推論・通知
python run_daily_predict.py

# 6. 特定ラウンドや全レース（1〜12R）を一括推論する場合
python run_daily_predict.py --rounds 9,10,11,12
python run_daily_predict.py --rounds all

# 7. 夕方の確定レース結果自動取得・収支精算・Discordレポート送信
python run_daily_settlement.py
```

## 5. 実装・運用上の重要原則

- **オッズ非依存のピュア能力予測原則**
  - モデルの学習特徴量には「市場オッズ・人気順」を直接含めない（大衆の集合知に引きずられるカンニングと回収率の希釈を防止）。
  - オッズは「推論後の期待値（EV）計算およびケリー資金配分」でのみ利用すること。

- **時系列リークの完全排除**
  - 特徴量生成（過去走・タイム指数・PCI・Eloレート等）はすべて `shift(1)` または時系列順の累積更新（過去レースのみ）を徹底すること。

- **型安全とエラーハンドリング**
  - Python 3.10 以上を使用し、すべての関数・メソッドに Type Hints を明記すること。
  - 出走取消・競走中止馬などの欠損値・異常値でパイプラインが停止しない堅牢な例外設計を保つこと。

## 6. 今後の実装予定機能ロードマップ

1. **血統知識グラフ埋め込み（Embedding）**
   - 種牡馬・母父・系統のネットワーク表現学習による血統適性のベクトル化。

2. **発走直前リアルタイムオッズ・確定馬体重の自動再推論**
   - 発走15〜20分前の最新オッズと馬体重による期待値（EV）動的再計算と直前アラート配信。

3. **調教時計・追い切り評価の数値化**
   - 最終追い切り時計、加速ラップ判定、調教本数の特徴量化。

4. **3連複・3連単へのフォーメーション展開**
   - モンテカルロシミュレーションの上位馬群を用いた3連系券種の期待値算出と買い目絞り込み。