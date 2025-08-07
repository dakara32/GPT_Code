# GPT_Code

## 週次レポート

`drift.py` は現在のポートフォリオのドリフトを計算し、Slack にレポートを送信するスクリプトです。

### 手動実行 (CLI エントリポイント)

環境変数 `SLACK_WEBHOOK_URL` を設定した上で、次のように実行できます。

```bash
python drift.py
# またはモジュールとして
python -m drift
# もしくは関数を直接呼び出す
python -c "from drift import weekly_report; weekly_report()"
```

### cron ジョブ例 (毎週月曜 9:00 実行)

```
0 9 * * MON export SLACK_WEBHOOK_URL=https://example.com && /usr/bin/python /path/to/drift.py
```

