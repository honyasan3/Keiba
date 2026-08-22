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
│   │   ├── horse_features.py    # 過去走・タイム指数・PCI・展開不利・休養・Elo対戦レーティング・騎手実績特徴量
│   │   ├── race_features.py     # レース内相対特徴量・カテゴリ変換・想定ペース判定
│   │   └── track_bias_features.py # 当日トラックバイアス特徴量（内外枠有利度・前残り有利度、直前レースまでの累積値でリーク防止）
│   ├── models/                  # モデル学習・予測モジュール
│   │   ├── base_model.py        # モデル共通インターフェース
│   │   ├── lgbm_model.py        # LightGBMモデル（2値分類 / 複勝予測）
│   │   ├── catboost_model.py    # CatBoostモデル（カテゴリ変数ネイティブ対応）
│   │   └── ranker_model.py      # LambdaMARTモデル（順位学習 / Learning to Rank）
│   ├── simulation/              # レースシミュレーションモジュール
│   │   └── race_simulator.py    # 1万回モンテカルロ仮想出走・ワイド/馬連結合確率・ケリー資金配分エンジン
│   ├── evaluation/              # 評価・最適化・自動精算モジュール
│   │   ├── metrics.py           # AUC, LogLoss, Accuracy, Brierスコア, 確率較正表（reliability diagram）算出
│   │   ├── ensemble_runner.py   # optimize_betting.py/evaluate_calibration.py共通のトリプルアンサンブル一括推論ヘルパー
│   │   ├── simulator.py         # 複勝・単勝の回収率（ROI）・的中率バックテスト
│   │   ├── strategy_optimizer.py # 回収率最大化グリッドサーチ最適化クラス（複勝/単勝/ワイド対応、1ルール単体評価用のevaluate_*メソッドも提供）
│   │   └── settlement_reporter.py # 確定レース結果・払戻金の自動取得およびDB記録済み買い目との突合精算クラス
│   └── notification/            # 通知モジュール
│       └── discord_notifier.py  # Discord Webhookリッチ通知クラス (予想Embed・収支Embed対応)
├── data/                        # ローカルデータ保存領域（Git管理対象外）
│   ├── cache/                   # 取得生HTMLのローカルキャッシュ
│   └── keiba.db                 # データベースファイル（SQLite: races / results / predictions）
├── models_saved/                # 学習済みモデル保存領域
│   ├── lgbm_model.txt           # LightGBM学習済みモデル重み
│   ├── catboost_model.cbm       # CatBoost学習済みモデル重み
│   └── lambdarank_model.txt     # LambdaMART順位学習済みモデル重み
├── tests/                       # 単体テストコード（pytest）
│   ├── test_crawler.py
│   └── test_splitter.py
├── main_phase1.py               # フェーズ1: スクレイピング・DB格納エントリーポイント
├── main_phase2.py               # フェーズ2: 特徴量作成・トリプルアンサンブル学習・検証スクリプト
├── backtest_simulation.py       # 長期運用バックテスト（複勝: ケリー基準資金管理、ワイド: フラットステーク）
├── optimize_betting.py          # ベッティング戦略walk-forward探索スクリプト（Validation選定→Test評価）
├── evaluate_calibration.py      # 確率較正診断＋Isotonic回帰較正のwalk-forward再検証スクリプト
├── rolling_walk_forward.py      # 複数fold（train拡張窓で毎回モデル再学習）によるローリングwalk-forward検証スクリプト
├── track_record_report.py       # predictionsテーブルの実績（券種別・週別・オッズ帯別）集計レポートスクリプト
├── predict.py                   # レース推論（トリプルアンサンブル × シミュレーション × 買い目のDB記録 × Discord通知）
├── run_daily_predict.py         # 当日メイン・全レース一括自動推論・通知スクリプト
├── run_daily_settlement.py      # DB記録済み買い目を実際の払戻データと自動精算・Discord収支レポート送信スクリプト
├── run_today.bat                # ワンクリック自動予測実行バッチ
└── requirements.txt              # 依存パッケージ一覧（pip install -r requirements.txt）
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

### `predictions` テーブル（AI推奨買い目の記録・精算結果）

実運用の的中率・回収率トラックレコードを残すためのテーブル。`predict.py`が買い目を出すたびに保存し、`run_daily_settlement.py`が実際の払戻データと突合して結果を書き戻す。

| **カラム名**          | **データ型**       | **制約** | **説明**                  |
| ----------------- | -------------- | ------ | ----------------------- |
| `id`               | `INTEGER`      | **PK**, AutoIncrement | 一意ID |
| `race_id`          | `VARCHAR(32)`  |        | レースID |
| `race_title`       | `VARCHAR(128)` | Nullable | レース名 |
| `race_date`        | `VARCHAR(32)`  | Nullable | 開催日 (`YYYY-MM-DD`) |
| `bet_type`         | `VARCHAR(16)`  |        | 券種 (`place` / `win` / `wide`) |
| `horse_num`        | `VARCHAR(16)`  |        | `place`/`win`は馬番、`wide`は`"3-7"`のペア表記 |
| `horse_name`       | `VARCHAR(128)` | Nullable | 馬名（ペア表記含む） |
| `odds_at_predict`  | `FLOAT`        | Nullable | 推論時点の単勝オッズ（`wide`は想定オッズ） |
| `pred_prob`        | `FLOAT`        | Nullable | 推論時点の的中確率 |
| `ev`               | `FLOAT`        | Nullable | 推論時点の期待値 |
| `bet_amount`       | `INTEGER`      | Default 100 | 推奨賭け金 |
| `predicted_at`     | `VARCHAR(32)`  | Nullable | 推論実行日時 |
| `settled`          | `BOOLEAN`      | Default False | 精算済みフラグ |
| `is_hit`           | `BOOLEAN`      | Nullable | 的中したか |
| `payout_amount`    | `INTEGER`      | Nullable | 払戻金額 |
| `settled_at`       | `VARCHAR(32)`  | Nullable | 精算実行日時 |

> **テーブル制約**: `UniqueConstraint("race_id", "bet_type", "horse_num")` — 同一レースを再推論しても重複登録されず、最新の推論内容で上書きされる。`run_daily_settlement.py`は単勝・複勝・ワイドすべてを自動精算する。

## 3. 各モジュールの役割および主要クラス設計

### 3.1 設定・共通インフラ (`config/`, `src/common/`)

- **`ConfigLoader`** (`config/config_loader.py`): `settings.yaml` から設定項目を読み込み、型安全なオブジェクトとして提供。
- **`DatabaseConnector`** (`src/common/db.py`): SQLAlchemyを用いたセッション・トランザクション管理。
- **`setup_logger`** (`src/common/logger.py`): コンソールおよびログファイル（`logs/app.log`）への統一フォーマット出力。

### 3.2 データ収集モジュール (`src/crawler/`)

- **`BaseScraper`** (`src/crawler/base_scraper.py`): キャッシュ保存・再利用、指数バックオフ再試行、最少1.5秒以上のランダム遅延待機。文字コードは`db.netkeiba.com`(EUC-JP)と`race.netkeiba.com`(UTF-8)で異なるため、`response.apparent_encoding`による自動判定を行う（固定値指定は文字化けの原因になるため使用しない）。
- **`RaceScraper`** (`src/crawler/race_scraper.py`): レースID一覧・詳細結果HTMLを取得。
  - `fetch_race_card()`: 出馬表取得は常に`race.netkeiba.com/race/shutuba.html`を使用する。`db.netkeiba.com`側の過去アーカイブは未来（未確定）レースIDに対してHTTPエラーを返さず出走馬データを含まない空ページを返すため、db側を優先すると発走前予想が常に失敗する。
  - `fetch_live_odds()`: 単勝オッズ・人気順の取得。出馬表HTMLには静的にオッズが埋め込まれておらず、ページ内JSが`race.netkeiba.com/api/api_get_jra_odds.html`を非同期に呼んで反映している。JRAは枠順抽選・オッズ公表前の段階では該当データを一切持たないため、未公表時は空dictを返す（呼び出し側は架空の値で代替してはならない）。`fetch_page`（HTML取得）と同じランダム遅延を`finally`節で必ず挟む（成功・失敗・未公表いずれの場合も待機する）。追加前は待機なしで即時応答していたため、`run_daily_predict.py --rounds all`のように1日36レース分を連続処理すると、このAPIだけ配慮のない連打になっていた。
- **`RaceHtmlParser`** (`src/crawler/html_parser.py`): 結果HTMLから通過順・上がり3F・オッズ・馬体重を構造化抽出。
- **`ShutubaHtmlParser`** (`src/crawler/shutuba_parser.py`): 出馬表HTMLから出走馬データを抽出。
  - 枠番・馬番は、枠順抽選が確定するまで`class="Umaban Txt_C"`のようにクラス名に数字が付与されず、セルも空になる（JS側で後から埋め込む仕様）。抽選確定後は`class="Umaban3 Txt_C"`のように数字が付くため、これを正としてパースする。`<tr id="tr_N">`のNは出走馬登録順の内部IDであり実際の馬番とは無関係なので、馬番の代用として使用しない。
  - 馬体重は`<td class="Weight">`から取得する。`Umaban`/`Waku`と同様、JRA発表前（通常発走1時間前まで）は空セルのため、その間は空文字を返す。
  - オッズは本パーサーでは取得しない（`odds_raw`は空文字のまま返す）。`predict.py`側で`RaceScraper.fetch_live_odds()`の結果と馬番で突き合わせて補完する。
- **`RaceScheduleScraper`** (`src/crawler/race_schedule_scraper.py`): 開催日程からレースID一覧を自動取得。
  - `race.netkeiba.com/top/race_list.html`の当日一覧はJSが`race_list_sub.html?kaisai_date=YYYYMMDD`を非同期に呼んで描画しており、静的HTMLにはレースリンクが存在しない。本クラスはそのエンドポイントを直接叩くことで、開催前・開催後どちらの日付でも一覧を取得できる（`db.netkeiba.com/race/list/{日付}/`は結果確定後のアーカイブのため未来日には使えない）。

### 3.3 データ処理・検証モジュール (`src/pipeline/`, `src/dataset/`)

- **`DataCleaner`** (`src/pipeline/cleaner.py`): 文字列正規化、性齢分離、秒換算、馬体重増減処理。
- **`RaceRepository`** (`src/pipeline/repository.py`): クレンジング済みデータのDB保存・Upsert（重複防止更新）。
- **`PredictionRepository`** (`src/pipeline/repository.py`): `predict.py`が出した推奨買い目の保存（`(race_id, bet_type, horse_num)`単位でUpsert）、未精算買い目の取得、精算結果の書き戻しを行う。
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
- **`TrackBiasFeatureExtractor`** (`src/features/track_bias_features.py`)【フェーズ4追加】:
  - 当日・同競馬場・同コース種別内で、直前レースまでの結果を累積集計した「当日トラックバイアス」特徴量（ベイズ平滑化込み、`shift`相当で当日の自レース結果はリーク排除）。
  - `bias_inner_bracket_advantage`: 累積内枠複勝率 − 累積外枠複勝率。
  - `bias_front_runner_advantage`: 直前レースまでの3着以内馬の平均4角通過位置（前残り傾向の強さ）。
  - `bias_horse_match_score`: 出走馬の枠番・脚質と当日バイアスとの適合度スコア。
  - 注: `src/features/base_feature.py`の`BaseFeatureExtractor`は継承していない（他の抽出クラスと実装形式が異なる）。

### 3.5 機械学習モデルアーキテクチャ (`src/models/`)

- **`LGBMRacePredictor`** (`src/models/lgbm_model.py`): LightGBMを用いた2値分類（3着以内好走確率）モデル。
- **`CatBoostRacePredictor`** (`src/models/catboost_model.py`): カテゴリ変数をネイティブ処理する2値分類モデル。
- **`LGBMRankPredictor`** (`src/models/ranker_model.py`):
  - **【フェーズB】順位学習（Learning to Rank / LambdaMART）**:
  - 目的関数: `objective="lambdarank"`（評価指標: NDCG@1, @3, @5）。
  - 同一レース内での相対着順序列を直接最適化。
- **★ トリプルアンサンブル構成 (40:40:20 Blend)**:
  - $\text{Ensemble\_Prob} = (0.40 \times \text{LGBM}) + (0.40 \times \text{CatBoost}) + (0.20 \times \text{LambdaRank\_Percentile})$
  - ピュア能力43特徴量でテストセット **AUC: 0.7734** を達成（この数値は43特徴量版での計測であり、後続で追加された当日トラックバイアス特徴量込みの現行46特徴量版では未再計測）。

### 3.6 シミュレーション・資金管理・バックテスト (`src/simulation/`, `src/evaluation/`)

- **`MonteCarloRaceSimulator`** (`src/simulation/race_simulator.py`):
  - 10,000回仮想出走による各馬の勝率・複勝率およびワイド結合確率算出。
- **フラクショナル・ケリー基準（Fractional Kelly Criterion）**:
  - $f^* = \frac{b \cdot p - q}{b} \times 0.15$ （破産確率を抑制した最適賭け金傾斜配分）。
- **`backtest_simulation.py`**:
  - 2025年12月〜2026年8月のテスト期間（2,418レース、現行デプロイ済みモデルの学習カットオフ以降）における長期運用検証。複勝・ワイドをそれぞれ独立したシミュレーションとして評価する（合算すると片方の勝敗がもう片方の資金効率に影響してしまうため）。
  - **複勝**（ケリー資金配分）: 購入基準は`rolling_walk_forward.py`のローリングwalk-forward検証（3fold・各foldでモデル再学習）で一貫して選ばれたルール（EV≥1.4, 複勝率≥0.45, 予測3位以内, 単勝オッズ3.0〜5.0倍）。**購入件数: 44件 / 的中率: 47.73% / 総投資額: 175,000円 / 総払戻額: 159,276円 / 純利益: −15,724円 / 回収率: 【91.02%】 / 最大ドローダウン: 24.89%**。
  - **ワイド**（1点100円のフラットステーク）: 購入基準は同じくローリングwalk-forward検証で選ばれたルール（EV≥1.0, 的中確率≥0.3, 想定オッズ3.0〜10.0倍）。**購入件数: 5,369件 / 的中率: 21.98% / 総投資額: 536,900円 / 総払戻額: 653,130円 / 純利益: ＋116,230円 / 回収率: 【121.65%】**。
  - ⚠️ **ワイドがケリー資金配分ではなくフラットステークである理由**: ワイドは複勝よりベット頻度が桁違いに高く（この期間だけで5,369件）、しかも同じ馬を含むペア同士は結果が強く連動する。bankroll比例のケリー複利をそのまま適用すると、たった8ヶ月のシミュレーションでも理論上は数百億〜数兆円規模まで際限なく複利成長してしまい、現実の馬券市場が持つ資金吸収力の限界（賭け金が大きいほどオッズが不利に動く）を無視した非現実的な数字になることが実装時に判明した（開発時に実際に発生し、原因を特定して意図的にフラットステークへ切り替えた）。複勝は購入頻度が低くこの問題が表面化しないためケリー資金配分のまま残している。
  - ⚠️ **回収率の位置づけ**: 複勝はこの単一Test期間単体では依然わずかに赤字（3fold中もっとも成績が弱かった窓と現行モデルの学習カットオフが一致しているため）。一方、`rolling_walk_forward.py`による3つの独立した期間の合算では複勝プールROI 110.34%（326件）・ワイドプールROI 120.00%（15,078件）といずれも黒字で、特にワイドは件数・安定性ともにこの一連の検証で最も裏付けが強い（詳細は「6.1」）。ただしいずれもサンプルサイズ・観測期間が長期的な保証を出せるほど十分とは言えないため、回収率を保証や訴求文言として使うべきではない。以前記載していた「購入件数1,033件・ROI 118.06%」は、閾値選定と評価に同一テスト期間データを使う循環検証によるものだったため撤回済み。

### 3.7 自動運用・収支精算・Discord通知 (`predict.py`, `run_daily_predict.py`, `run_daily_settlement.py`, `src/notification/`, `src/evaluation/`)

- **`predict.py`**: 出馬表取得 → ライブオッズ取得(`fetch_live_odds`) → 46特徴量生成（Elo・トラックバイアス算出含む） → トリプルアンサンブル推論 → 1万回シミュレーション → ケリー推奨額付き買い目抽出 → **推奨買い目をDBへ記録**(`PredictionRepository`、通知有無に関わらず必ず保存) → Discord通知。
  - **買い目判定基準**（いずれも`rolling_walk_forward.py`によるローリングwalk-forward検証で選定、詳細は「6.1」）: 複勝はEV≥1.4・複勝率≥0.45・予測3位以内・単勝オッズ3.0〜5.0倍。ワイドはEV≥1.0・的中確率≥0.3・想定オッズ3.0〜10.0倍（この一連の検証の中で最も件数が多く安定した結果）。単勝(win)推奨は同検証で優位性が確認できなかったため無効化している（常に空リスト）。
  - **オッズ未公表ガード**: 枠順抽選・オッズ公表前のレースは`odds`が全てNoneになる。この場合、EV・複勝オッズ推定・ケリー推奨額は計算せず（架空オッズでの買い目提示を防止するため）、買い目判定そのものを保留する。AI予測の確率ランキング自体はオッズ非依存のため通常通り表示される。
  - `--no-notify`オプションでDiscord通知をスキップ可能（DB記録は行われる）。
- **`SettlementReporter`** (`src/evaluation/settlement_reporter.py`): レース確定結果・公式払戻金(`race.netkeiba.com/race/result.html`、UTF-8)の自動取得、および買い目リストとの突合精算。単勝・複勝・ワイドに対応（ワイドは的中3組の払戻を`min-max`馬番キーに正規化して照合するため、買い目記録側の馬番順序に依存しない）。
- **`run_daily_settlement.py`**: `PredictionRepository`から未精算買い目（指定日以前・単勝/複勝/ワイド）を取得 → `SettlementReporter`で実際の払戻データと突合 → 結果(`is_hit`/`payout_amount`)をDBへ書き戻し → Discord収支レポート送信。`--no-notify`オプションあり。日付を省略した場合、指定日以前の未精算分をまとめて精算する。
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

# 2. 46特徴量作成・トリプルアンサンブル学習
#    (LGBM × CatBoost × LambdaMART)・モデル保存（フェーズ2）
python main_phase2.py

# 3. ベッティング閾値のwalk-forward選定
#    （Validation期間のみでグリッドサーチ→Test期間に一度だけ適用して評価。手順6.1参照）
python optimize_betting.py

# 4. 長期運用バックテスト
#    （Eloレーティング＆ケリー基準収支シミュレーション。閾値は3.のValidation選定結果を反映済み）
python backtest_simulation.py

# 5. 指定レースのリアルタイム推論
#    （トリプルアンサンブル×1万回シミュレーション×ケリー資金配分×買い目DB記録×Discord通知）
#    枠順抽選・オッズ公表前は買い目判定が保留になるため、発走が近づいてから実行すること
python predict.py 202605020811
python predict.py 202605020811 --no-notify   # Discord通知を出さずDB記録のみ行う場合

# 6. 指定日（または当日）のメインレース一括推論・通知
python run_daily_predict.py

# 7. 特定ラウンドや全レース（1〜12R）を一括推論する場合
python run_daily_predict.py --rounds 9,10,11,12
python run_daily_predict.py --rounds all

# 8. 確定レース結果自動取得・DB記録済み買い目との精算・Discordレポート送信
#    日付省略時は「指定日以前の未精算買い目」をまとめて精算する
python run_daily_settlement.py
python run_daily_settlement.py 2026-08-16
python run_daily_settlement.py --no-notify
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

## 6. 既知の制限事項・今後の実装課題

実運用検証の過程で判明した、現時点で未解決の課題を優先度順に記載する。

### 6.0 対応済み（旧・優先度高の項目）

以下は当初「優先度高」として記載していたが、既に対応済み。

- **`run_daily_predict.py`の日程自動検出**: `race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD`（ページ内JSが実際に叩いているエンドポイント）を直接呼び出す方式に修正済み。開催前・開催後どちらの日付でも正しくレース一覧を取得できることを確認済み（`RaceScheduleScraper.get_race_ids_by_rounds`）。
- **複勝オッズの実測値化**: `RaceScraper.fetch_live_odds`がオッズAPIの複勝レンジ(`data.odds."2"`、最低〜最高)を取得するようになり、`MonteCarloRaceSimulator`は実測値（保守的に下限値を採用）を優先し、取得できない場合のみ従来の近似式(`odds ** 0.45`)にフォールバックする。ただし過去データ（`results`テーブル）には複勝オッズが保存されていないため、`strategy_optimizer.py`/`backtest_simulation.py`の過去データに対する評価は引き続き近似式(`odds ** 0.45`)に依存している（`backtest_simulation.py`は当初`odds / 3.5`という別式になっていたが、以下のベッティング閾値見直しの過程で発見・統一済み）。
- **ベッティング閾値の再検証と本番反映**: 一連の検証を経て、複勝は根拠のある閾値へ更新・本番反映済み、単勝は根拠不足のため無効化済み。経緯と最終結論は下記「6.1」参照。
- **馬体重のライブ取得**: `Waku`/`Umaban`（枠番・馬番）と同じ仕組みで、`<td class="Weight">`もJRAが正式発表（通常発走1時間前）するまでは空、発表後は`"452(+6)"`のような実テキストが静的HTMLに入ることを確認。新規API不要で`ShutubaHtmlParser`側の抽出のみ修正済み。未発表時は仮値(470kg・増減0)で補完しつつ、`predict.py`が警告ログを出すようにした。
- **フォワードテスト実績のレポート機能**: `track_record_report.py`を新設。`predictions`テーブルの記録を券種別・週別・オッズ帯別に集計し、的中率・回収率の推移を表示する。
- **ワイド推奨の自動精算**: `SettlementReporter.fetch_race_payouts`がワイドの払戻表（的中3組の払戻を`min-max`馬番キーで正規化）にも対応し、`run_daily_settlement.py`の自動精算対象に追加済み。馬番の順序が逆（例: 実際の払戻表記が"3-10"で買い目記録が"10-3"）でも正しく突合できることを確認済み。
- **`jockey_features.py`の整理**: 空ファイルでどこからもimportされていないことを確認し削除済み（騎手関連特徴量は実際には`horse_features.py`の`PastPerformanceExtractor`に統合されている）。
- **`requirements.txt`の整備**: 実際にimportされているパッケージを洗い出し、動作確認済みのバージョンで固定して追加済み（Python 3.11.1環境）。
- **オッズAPIのレート制限**: `RaceScraper.fetch_live_odds`に`fetch_page`と同じランダム遅延（`min_delay`〜`max_delay`）を追加済み。実測で1回あたり約3〜3.7秒の待機が正しく効くことを確認。
- **キャッシュ整合性**: `data/cache/`（約17,824ファイル）のタイムスタンプを調査した結果、文字コード自動判定修正（`base_scraper.py`更新: 2026-08-20 23:53）より前に書き込まれたファイルが17,813件と大半を占めており、`race.netkeiba.com`系ページの誤ったデコード結果が残っている可能性を排除できなかったため、2026-08-22に`data/cache/`を丸ごと削除して対応済み。`fetch_page`はキャッシュ不在時に自動で再取得するため実害はない。
- **レース発走直前の自動実行（`schedule_today_predictions.py`）**: 当日の全JRAレースを`RaceScheduleScraper.get_race_ids_by_rounds`で取得し、各レースの出馬表(`RaceData01`)から発走時刻を専用の軽量フェッチ（`BaseScraper`のキャッシュは経由しない — キャッシュしてしまうと発走直前の馬体重確定・出走除外・騎手変更を取りこぼす古いスナップショットが残るため）で取得、発走時刻の指定分前（既定15分、`--margin`で変更可）に`predict.py <race_id>`を1回だけ実行するWindowsタスクスケジューラのタスクを自動登録する。`schtasks /TR`の261文字制限（日本語を含む長いプロジェクトパスを毎回埋め込むと超過する）を避けるため、実行内容は`run_scheduled_predict.bat`（純ASCII、`%~dp0`で自身の場所を解決）にまとめ、`/TR`にはそのbatパス＋race_idのみを渡す方式にしている。ログの文字化け（cmd.exeのデフォルトコードページ由来）は`set PYTHONIOENCODING=utf-8`で解消済み。`--dry-run`（登録せず一覧確認のみ）・`--cleanup`（登録済み`KeibaPredict_*`タスクを一括削除）にも対応。ログ出力先は`logs/scheduled/predict_<race_id>.log`。PCがスリープ・シャットダウンした時間帯はタスクが発火しないため、対象時間帯はPCを起動・ログイン状態に保つ必要がある。2026-08-22に本番実行し、36件全て登録完了（スキップ0件）を確認済み。

### 6.1 ベッティング閾値検証の経緯と結論（対応済み）

**結論（現在の本番設定）**: `predict.py` / `backtest_simulation.py` は複勝（EV≥1.4・複勝率≥0.45・予測3位以内・単勝オッズ3.0〜5.0倍）とワイド（EV≥1.0・的中確率≥0.3・想定オッズ3.0〜10.0倍）のルールで買い目判定を行う。単勝(win)の推奨は根拠不足のため無効化した（`predict.py`は常に空リストを返す）。3fold合算の証拠量では**ワイド（15,078件・プールROI 120.00%）が複勝（326件・プールROI 110.34%）より件数・安定性ともに強い**。

**経緯（何を検証し、何が分かったか）**

1. **単純なwalk-forward検証（Validation選定→Test適用）**: 旧ルール（複勝EV≥1.2/prob≥0.45/1位/オッズ≥1.5、単勝EV≥1.8/prob≥0.35/2位以内/5〜30倍）をTest期間（2025-12〜2026-08、選定に未使用）に適用した結果、複勝ROI 88.51%（531件）、単勝ROI 85.30%（1367件）、ケリー資金配分の長期バックテストでも複勝ROI 86.00%（純利益−59,500円、最大DD 77.42%）と、いずれも100%割れ。以前報告していた「ROI 118.06%」が同一テスト期間に対する循環検証（選定と評価が同一データ）による過大評価だったことが確定した。

2. **確率較正チェック（`evaluate_calibration.py`）**: AUCはTrain 0.79/Val 0.77/Test 0.78と安定しておりランキング能力自体は健全。一方、判断関連の確率帯（0.30〜0.55）で3期間とも一貫して+5〜+9pt程度の自信過剰があり、さらにオッズ帯によって歪みの方向が逆転（本命は最大-21pt過小評価、穴馬は+9pt過大評価）していることが判明。EV=予測確率×オッズで判断するため、モデルが最も過信する穴馬ゾーンほど見かけのEVが高く出やすく、これが損失の一因と推定された。Validation期間でIsotonic回帰較正を学習しTest期間に適用したところ較正表の乖離は大幅縮小したが、**肝心の収益性は複勝88.51%→91.51%とほぼ変わらず、単勝は85.30%→77.71%とむしろ悪化**。1次元の確率較正だけでは、オッズと連動した2次元的な歪みまでは直せなかった。

3. **オッズ帯（3〜5倍）に絞った戦略の単一Test再検証**: 較正チェックで歪みが最小だったオッズ帯を候補に加えて再度Validation選定→Test適用したところ、複勝Test ROIは91.43%（44件）とわずかに改善したが、**Validation側の該当件数がグリッドサーチの最低件数条件（50件）ちょうど**というごく小さいサンプルで、統計的には誤差の範囲内。同一Test期間に対する選定→確認をこれ以上繰り返すのは循環検証の再発になるためここで打ち切った。

4. **ローリングwalk-forward検証（`rolling_walk_forward.py`、再学習あり・3fold）**: 上記3の結果が単一Test窓固有のものか確認するため、train窓を広げながら3つの独立したfoldでモデルをゼロから再学習し、各foldでValidation選定→Test適用を行った。
   - 複勝: Fold1 (Test 2024-07〜2025-03) ROI 105.47%（142件）、Fold2 (Test 2025-04〜2025-12) ROI 121.22%（140件）、Fold3 (Test 2025-12〜2026-08) ROI 91.43%（44件）。**3fold中2foldが単勝オッズ3.0〜5.0倍という同一オッズ帯を独立に選定**（残り1foldも2.0〜6.0倍と近い）。3fold合算（賭け金ベースでプール）326件・的中率56.44%・**プールROI 110.34%**。
   - 単勝: 3fold合算1948件・**プールROI 80.77%**、かつ選定ルールがfold間で安定しない（Fold1は的中4/121件という極端な longshot 限定ルールを選定）。優位性は確認できなかった。
   - **複数の独立した時代・独立に再学習したモデルで、複勝×3〜5倍オッズ帯という同じパターンが繰り返し選ばれたことは、単一テスト期間の結果よりも強い証拠になる。**
   - この複勝ルール（EV≥1.4/prob≥0.45/3位以内/3〜5倍）を`backtest_simulation.py`（現行デプロイ済みモデル・Test期間2025-12〜2026-08のみ）でケリー資金配分再評価したところ、**44件・的中21件・ROI 91.02%・純利益−15,724円・最大DD 24.89%**。3fold中もっとも成績の弱かった窓と現行モデルの学習期間が一致するため、直近だけを見ると依然ほぼ収支トントン〜やや赤字。3fold全体の傾向（黒字）と、現行モデル単体の直近実績（わずかに赤字）の両方を正直に併記しておく。
   - **`backtest_simulation.py`の複勝オッズ近似式が`race_simulator.py`/`strategy_optimizer.py`と異なる式（`odds/3.5`）になっていたバグも本件で発見・修正**（`odds ** 0.45`に統一）。旧式のままだと3〜5倍帯ではEV1.4を満たす馬が事実上存在せず0件になっていた。

5. **ワイドのローリングwalk-forward検証**: 複勝・単勝に続き、ワイドも同じ3fold検証（各foldでモデル再学習・Val選定→Test適用）で検証した。ワイドは1頭ごとの確率だけでは判定できず「2頭とも3着以内」という組み合わせ確率が必要なため、`strategy_optimizer.py`に`prep_wide_eval_df`を追加し、過去レースごとに本番と同じ`MonteCarloRaceSimulator`でシミュレーションして実際の着順と突き合わせた（216レースの検証で約1.6秒と高速に処理できることを確認済み）。
   - Fold1 ROI 118.41%（4,614件）、Fold2 ROI 119.81%（5,103件）、Fold3 ROI 121.54%（5,361件）と**3fold全てが黒字、かつ118〜122%と極めて狭い範囲に収まった**。**3fold中2foldがEV≥1.0・的中確率≥0.3・想定オッズ3.0〜10.0倍という完全に同一のルールを独立選定**（残り1foldもEV≥1.25とごく近い）。3fold合算15,078件・的中率21.76%・**プールROI 120.00%**。326件だった複勝より遥かに大きいサンプルで、この一連の検証の中で最も統計的に信頼できる結果。
   - この結果を受けて`predict.py`のワイド推奨閾値を更新（旧: EV≥1.25・確率≥0.15・オッズ帯制限なし → 新: EV≥1.0・確率≥0.3・オッズ3〜10倍）。
   - **`backtest_simulation.py`にワイドのケリー資金配分シミュレーションを追加する過程で、複利計算が破綻するバグを発見**: ワイドは複勝よりベット頻度が桁違いに高く（同じレース内で馬を共有する複数ペアが同時に条件を満たしやすく、しかもそれらは結果が強く連動する）、bankroll比例のケリー複利をそのまま適用したところ、たった8ヶ月のシミュレーションで総投資額が「718京円」という物理的に有り得ない金額まで複利成長した。レース単位でベット額をその開始時点のbankroll基準にする修正、さらにレース単位の合計エクスポージャーに上限をかける修正を施しても、桁は減ったものの依然非現実的な規模（数百億円）に達したため、根本原因は「現実の馬券市場が持つ資金吸収力の限界（賭け金が大きいほどオッズが不利に動く）をモデル化していないこと」と判断し、ワイドはケリー複利ではなく1点100円のフラットステークで評価する方式に切り替えた（複勝は購入頻度が低くこの問題が表面化しないためケリー資金配分のまま）。フラットステークでの再評価は**5,369件・的中率21.98%・ROI 121.65%・純利益+116,230円**となり、`rolling_walk_forward.py`のFold3単体結果（121.54%）とほぼ一致し整合性が取れている。

**総括**: 複勝×オッズ3〜5倍帯という組み合わせは、確率較正チェックという独立した分析から仮説が生まれ、3つの独立した再学習・再検証で繰り返し支持された、という点で単なる偶然のグリッドサーチ結果より信頼できる。ただし直近の実績単体はまだ弱含みであり、「安定して儲かる」と断定するにはサンプルサイズ（326件）・観測期間ともに十分とは言えない。**β運用では小額から慎重に、回収率を保証や訴求文言として使わずに進めることを推奨する。**

### 6.2 優先度: 低（将来構想）

1. **血統知識グラフ埋め込み（Embedding）**
    - 種牡馬・母父・系統のネットワーク表現学習による血統適性のベクトル化。

2. **調教時計・追い切り評価の数値化**
    - 最終追い切り時計、加速ラップ判定、調教本数の特徴量化。

3. **3連複・3連単へのフォーメーション展開**
    - モンテカルロシミュレーションの上位馬群を用いた3連系券種の期待値算出と買い目絞り込み。