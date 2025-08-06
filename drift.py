import yfinance as yf
import pandas as pd
import numpy as np
import requests

# --- 1. ポートフォリオ定義 ---
raw_list = """
NET,11
MRVL,29
NVDA,12
GOOG,10
VRT,14
TTD,22
ENB,40
WMB,31
TJX,14
ABBV,9
IBKR,28
ACGL,20
CLS,9
APO,12
CEG,5
AMZN,8
CME,6
UBER,19
VEEV,6
ICE,9
SYY,21
TSM,7
AAPL,8
AMAT,9
TSLA,5
"""
lines = [line.split(",") for line in raw_list.strip().splitlines()]
portfolio = [
    {"symbol": sym.strip().upper(), "shares": int(qty), "target_ratio": 1/25}
    for sym, qty in lines
]

# --- 2. VIX MA5 & 閾値設定 ---
vix_ma5 = yf.Ticker("^VIX").history(period="7d")["Close"].tail(5).mean()
drift_threshold = 10 if vix_ma5 < 20 else 12 if vix_ma5 < 26 else float("inf")

# --- 3. 株価取得＆ドリフト計算 ---
for stock in portfolio:
    try:
        price = yf.Ticker(stock["symbol"]).history(period="1d")["Close"].iloc[-1]
    except Exception:
        price = 0
    stock["price"] = price
    stock["value"] = price * stock["shares"]

df = pd.DataFrame(portfolio)
total_value        = df["value"].sum()
df["current_ratio"] = df["value"] / total_value
df["drift"]         = df["current_ratio"] - df["target_ratio"]
df["drift_abs"]     = df["drift"].abs()
total_drift_abs     = df["drift_abs"].sum()

# --- 4. 半戻し（ideal）計算 ---
df["adjusted_ratio"] = df["current_ratio"] - df["drift"] / 2
df["adjustable"]     = (df["adjusted_ratio"] * total_value) >= df["price"]

# --- 5. 閾値超過時のみシミュレーション ---
alert = drift_threshold != float("inf") and total_drift_abs * 100 > drift_threshold
if alert:
    df["trade_shares"] = df.apply(
        lambda r: int(round(((r["adjusted_ratio"] * total_value) - r["value"]) / r["price"]))
        if r["adjustable"] else 0,
        axis=1
    )
    df["new_shares"]            = df["shares"] + df["trade_shares"]
    df["new_value"]             = df["new_shares"] * df["price"]
    new_total_value             = df["new_value"].sum()
    df["simulated_ratio"]       = df["new_value"] / new_total_value
    df["simulated_drift_abs"]   = (df["simulated_ratio"] - df["target_ratio"]).abs()
    simulated_total_drift_abs   = df["simulated_drift_abs"].sum()
else:
    df["trade_shares"]        = np.nan
    simulated_total_drift_abs = np.nan

# --- 6. 合計行追加＆必要カラム抽出 ---
summary = {
    "symbol":    "合計",
    "shares":    df["shares"].sum(),
    "value":     df["value"].sum(),
    "current_ratio": np.nan,
    "drift_abs": total_drift_abs,
}
if alert:
    summary["trade_shares"] = np.nan

df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

if alert:
    cols = ["symbol", "shares", "value", "current_ratio", "drift_abs", "trade_shares"]
    df_small = df[cols].copy()
    df_small.columns = ["sym", "qty", "val", "now", "|d|", "Δqty"]
else:
    cols = ["symbol", "shares", "value", "current_ratio", "drift_abs"]
    df_small = df[cols].copy()
    df_small.columns = ["sym", "qty", "val", "now", "|d|"]

# --- 7. 表示整形フォーマッタ ---
def currency(x): return f"${x:,.0f}" if pd.notnull(x) else ""
formatters = {
    "val":    currency,
    "now":    "{:.2%}".format,
    "|d|":    "{:.2%}".format
}
if alert:
    formatters["Δqty"] = "{:.0f}".format

# --- 8. Slack Incoming Webhook 送信 ---
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

header = (
    f"*📈 VIX MA5:* {vix_ma5:.2f}\n"
    f"*📊 ドリフト閾値:* {'🔴(高VIX)' if drift_threshold == float('inf') else str(drift_threshold)+'%'}\n"
    f"*📉 現在のドリフト合計:* {total_drift_abs * 100:.2f}%\n"
)
if alert:
    header += f"*🔁 実際のドリフト合計:* {simulated_total_drift_abs * 100:.2f}%\n"
    header += "🚨 *アラート: 発生！！！！！！*\n"
else:
    header += "✅ アラートなし\n"

table_text = df_small.to_string(formatters=formatters, index=False)
payload    = {"text": header + "\n```" + table_text + "```"}

resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
resp.raise_for_status()

print("✅ Slack（Webhook）へ送信しました")
