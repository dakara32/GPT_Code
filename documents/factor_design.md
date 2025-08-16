# factor.py 詳細設計書

## 概要
- ファクタースコアと相関低減を組み合わせた銘柄選定パイプライン。
- 入力データの取得からスコア算出、G/D枠の選定、結果出力までを一体化。

## 定数・設定
- `current_tickers.csv`/`candidate_tickers.csv`から既存銘柄と候補銘柄を読み込む。
- 価格上限`CAND_PRICE_MAX`、ベンチマーク`bench`、枠サイズ`N_G`/`N_D`を定義。
- ファクター重み`g_weights`/`D_weights`、DRRS関連パラメータ`corrM`、`DRRS_G`/`DRRS_D`、
  残差相関シュリンク`DRRS_SHRINK`、G-D間ペナルティ`CROSS_MU_GD`を保持。
- 選定結果は`results/`配下にJSONとして保存。

## DTO/Config
- **InputBundle**: Scorerへ渡す生データ一式。
- **FeatureBundle**: Scorerから返される特徴量・スコア・欠損ログ。
- **SelectionBundle**: Selectorから返される選定結果。
- **WeightsConfig**: G/Dファクターの重み。
- **DRRSParams**: DRRSのパラメータ群。
- **PipelineConfig**: 上記設定を束ねるコンテナ。

## 共通ユーティリティ
- `winsorize_s` / `robust_z` : 外れ値処理とZスコア化。
- `_safe_div` / `_safe_last` : 例外を潰した分割・末尾取得。
- `_load_prev` / `_save_sel` : 選定結果の読み書き。

## クラス設計
### Input
外部I/Oと前処理を担当し、`prepare_data`で`InputBundle`を生成。
主なメソッド:
- `impute_eps_ttm` : 四半期EPSからTTMを補完。
- `fetch_cfo_capex_ttm_yf` : yfinanceからCFO/CapEx TTMを取得。
- `fetch_cfo_capex_ttm_finnhub` : Finnhub APIから同値を取得。
- `compute_fcf_with_fallback` : 上記情報を統合しFCFを計算。
- `_build_eps_df` : EPS関連データフレーム構築。
- `prepare_data` : 価格・財務・EPS・FCF・リターンをまとめる。

### Scorer
特徴量計算とスコア合成を担当し、`FeatureBundle`を返す。
補助メソッドに`trend`、`rs`、`tr_str`、`rs_line_slope`、`ev_fallback`、
`dividend_status`、`div_streak`、`fetch_finnhub_metrics`、`calc_beta`などがある。
`aggregate_scores`の流れ:
1. 各銘柄について価格系列・財務情報・配当履歴を読み込み、トレンド指標（移動平均や52週高安）、相対強さ、ベータ、下方リスク、EPS/売上成長率、FCFなど多数の生指標を計算。欠損があればFinnhub API等で補完。
2. 主要指標を`winsorize_s`で外れ値処理後、`robust_z`でZスコア化し`df_z`に格納。サイズ・流動性の対数変換もここで行う。
3. 正規化された指標から、成長`GRW`、モメンタム`MOM`、トレンド`TRD`、ボラティリティ`VOL`の4因子を組み合わせ、`cfg.weights.g`の重みで合成して`g_score`を生成。
4. 同様に、守備的バケット用に`D_QAL`(FCFや財務健全性)、`D_YLD`(配当利回り/継続性)、`D_VOL_RAW`(ダウンサイド指標)、`D_TRD`(長期トレンド)を作り、`cfg.weights.d`で加重して`d_score_all`を算出。
5. 各指標が取得できなかった銘柄・項目は`missing_logs`として記録し後段で確認できるようにする。

### Selector
DRRSアルゴリズムで相関を抑えた銘柄選定を行い、`SelectionBundle`を返す。
主なメソッド:
- `residual_corr` : 収益率行列をZスコア化し、上位主成分を除去した残差から相関行列を求め、平均相関に応じてシュリンク。
- `rrqr_like_det` : スコアを重み付けしたQR分解風の手順で初期候補をk件抽出し、スコアの高い非相関な集合を得る。
- `swap_local_det` / `swap_local_det_cross` : `sum(score) - λ*within_corr - μ*cross_corr`を目的関数として、入れ替え探索で局所的に最適化。
- `select_bucket_drrs` : プール銘柄とスコアから残差相関を計算し、上記2段階(初期選択→入れ替え)でk銘柄を決定。過去採用銘柄との比較で目的値が劣化しなければ維持する。
- `select_buckets` : Gバケットを選定後、その結果を除いた候補からDバケットを選ぶ。D選定時はGとの相関ペナルティμを付与し、両バケットの分散を制御する。

### Output
結果整形と出力を担当。
主なメソッド:
- `display_results` : G/Dテーブル、IN/OUTリスト、パフォーマンス指標、分散化指標を表示。
- `notify_slack` : Slack Webhookへ同内容を送信。
- 補助:`_avg_offdiag`、`_resid_avg_rho`、`_raw_avg_rho`、`_cross_block_raw_rho`。

## エントリポイント
1. `PipelineConfig`を構築。
2. `Input.prepare_data`で`InputBundle`を生成。
3. `Scorer.aggregate_scores`で`FeatureBundle`を取得。
4. `Selector.select_buckets`で`SelectionBundle`を算出。
5. `Output.display_results`と`notify_slack`で結果を出力。
