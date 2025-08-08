import pandas as pd
import numpy as np
import requests
import os
import csv
import time
from pathlib import Path
import yfinance as yf

# --- Finnhub settings & helper ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not set (環境変数が未設定です)")

RATE_LIMIT = 55  # requests per minute (free tier is 60)
call_times = []


def finnhub_get(endpoint, params):
    """Call Finnhub API with basic rate limiting."""
    now = time.time()
    cutoff = now - 60
    while call_times and call_times[0] < cutoff:
        call_times.pop(0)
    if len(call_times) >= RATE_LIMIT:
        sleep_time = 60 - (now - call_times[0])
        time.sleep(sleep_time)
    params = {**params, "token": FINNHUB_API_KEY}
    try:
        resp = requests.get(f"https://finnhub.io/api/v1/{endpoint}", params=params)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"⚠️ Finnhub API JSON decode error: {e}")
        return {}
    except Exception as e:
        print(f"⚠️ Finnhub API error: {e}")
        return {}
    call_times.append(time.time())
    return data


def fetch_price(symbol):
    try:
        data = finnhub_get("quote", {"symbol": symbol})
        return data.get("c") or 0
    except Exception:
        return 0


def fetch_vix_ma5():
    """Retrieve VIX 5-day moving average via yfinance."""
    try:
        vix = (
            yf.download("^VIX", period="7d", interval="1d", progress=False)["Close"]
            .dropna()
            .tail(5)
        )
        if len(vix) < 5:
            return float("nan")
        return float(vix.mean())
    except Exception:
        return float("nan")

# --- 1. ポートフォリオ定義 ---
tickers_path = Path(__file__).with_name("current_tickers.csv")
with tickers_path.open() as f:
    reader = list(csv.reader(f))
portfolio = [
    {"symbol": sym.strip().upper(), "shares": int(qty), "target_ratio": 1/len(reader)}
    for sym, qty in reader
]

# --- 2. VIX MA5 & 閾値設定 ---
vix_ma5 = fetch_vix_ma5()
drift_threshold = 10 if vix_ma5 < 20 else 12 if vix_ma5 < 26 else float("inf")

# --- 3. 株価取得＆ドリフト計算 ---
for stock in portfolio:
    price = fetch_price(stock["symbol"])
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

# --- 8. Slack Incoming Webhook 送信（Secrets対応） ---
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")

header = (
    f"*📈 VIX MA5:* {vix_ma5:.2f}\n"
    f"*📊 ドリフト閾値:* {'🔴(高VIX)' if drift_threshold == float('inf') else str(drift_threshold)+'%'}\n"
    f"*📉 現在のドリフト合計:* {total_drift_abs * 100:.2f}%\n"
)
if alert:
    header += f"*🔁 半戻し後ドリフト合計:* {simulated_total_drift_abs * 100:.2f}%\n"
    header += "🚨 *アラート: 発生！！ Δqtyのマイナス分を売却、任意銘柄を買い増しバランスを取りましょう！*\n"
else:
    header += "✅ アラートなし\n"

table_text = df_small.to_string(formatters=formatters, index=False)
payload    = {"text": header + "\n```" + table_text + "```"}

try:
    resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
    resp.raise_for_status()
    print("✅ Slack（Webhook）へ送信しました")
except Exception as e:
    print(f"⚠️ Slack通知エラー: {e}")
