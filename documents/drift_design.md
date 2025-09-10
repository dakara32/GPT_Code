# drift.py 詳細設計書

## 概要
- 25銘柄ポートフォリオのドリフトを日次監視し、閾値超過時に半戻し案をSlack通知するスクリプト。
- Finnhubとyfinanceから価格を取得（レジームは trend_template 本数に基づく）。

## 定数・設定
- `FINNHUB_API_KEY` / `SLACK_WEBHOOK_URL` を環境変数から取得。
- 無料枠を考慮したAPIレート制限: `RATE_LIMIT = 55`。
- デバッグ出力用フラグ `debug_mode`。

## 主な関数
### finnhub_get
- 基本的なレート制限付きでFinnhub APIを呼び出し、JSONレスポンスを辞書で返す。

### fetch_price
- `quote` エンドポイントで株価を取得し、失敗時は `NaN` を返す。

### fetch_vix_ma5
- yfinanceでVIX終値を取得する関数。将来再利用のため残置。

### load_portfolio
- `current_tickers.csv` から銘柄と保有株数を読み込み、目標比率4%を付与したリストを生成。

### compute_threshold_by_mode
- モード(NORMAL/CAUTION/EMERG) に応じて 10% / 12% / 停止(∞) を返す。

### build_dataframe
- 各銘柄の評価額や現在比率、ドリフト、半戻し後比率(`adjusted_ratio`)を計算しDataFrame化。

### simulate
- ドリフト合計が閾値を超えた場合、半戻し後の売買株数と新比率を試算し、シミュレート後ドリフトを返す。

### prepare_summary
- 評価額順に並べ替えた後、合計行を付与してSlack表示用テーブルを作成。

### formatters_for / currency
- 通貨・比率・株数の表示フォーマットを定義。

### build_header
- 現金保有率・閾値・ドリフト値およびアラート有無をSlackメッセージ用ヘッダに整形。

### send_slack / send_debug
- 通常通知およびデバッグ詳細をSlack Webhookへ送信。

### main
- 上記関数を順に呼び出し、日次ドリフトチェックの一連処理を実行。

## 実行フロー
1. `load_portfolio` で現ポートフォリオを読み込む。
2. `build_breadth_header` でモードを取得し、`compute_threshold_by_mode` で現金保有率とドリフト閾値を決定。
3. `build_dataframe` で現在比率とドリフトを計算。
4. `simulate` で閾値超過時の半戻し案を試算。
5. `prepare_summary` と `build_header` で通知本文とテーブルを構築。
6. `send_slack` で結果を送信。`debug_mode` がTrueなら `send_debug` も併用。
