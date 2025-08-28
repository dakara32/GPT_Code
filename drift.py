import pandas as pd, yfinance as yf
import numpy as np
import requests
import os
import csv
import time
from pathlib import Path

# Debug flag
debug_mode = False  # set to True for detailed output

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
        price = data.get("c")
        return float(price) if price not in (None, 0) else float("nan")
    except Exception:
        return float("nan")


def fetch_vix_ma5():
    """Retrieve VIX 5-day moving average via yfinance."""
    try:
        vix = (
            yf.download("^VIX", period="7d", interval="1d", progress=False, auto_adjust=False)["Close"]
            .dropna()
            .tail(5)
        )
        if len(vix) < 5:
            return float("nan")
        return vix.mean().item()
    except Exception:
        return float("nan")


# === Minervini-like sell signals ===
def _yf_df(sym, period="6mo"):
    """日足/MA/出来高平均を取得。欠損時は None。"""
    try:
        df = yf.download(sym, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        return df.dropna().assign(
            ma20=lambda d: d["Close"].rolling(20).mean(),
            ma50=lambda d: d["Close"].rolling(50).mean(),
            vol50=lambda d: d["Volume"].rolling(50).mean(),
        )
    except Exception:
        return None


def _scalar(row, col):
    """Series/npスカラ→Pythonスカラ化（NaNはNaNのまま）"""
    try:
        v = row[col]
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass
        return v
    except Exception:
        return float("nan")


def _is_strict_down(seq):
    """数列が厳密に連続で切り下がっているか（len>=4を想定）。NaN含みはFalse。"""
    try:
        xs = [float(x) for x in seq]
        if any(pd.isna(x) for x in xs) or len(xs) < 4:
            return False
        return all(b < a for a, b in zip(xs[:-1], xs[1:]))
    except Exception:
        return False


def _signals_for_day(df, idx):
    """df.loc[idx] 1日分に対しシグナル配列を返す（値動き/出来高ベースのみ）。"""
    try:
        sig = []
        d = df.loc[idx]
        close = _scalar(d, "Close")
        open_ = _scalar(d, "Open")
        ma20 = _scalar(d, "ma20")
        ma50 = _scalar(d, "ma50")
        vol = _scalar(d, "Volume")
        vol50 = _scalar(df.iloc[-1], "vol50")
        if any(pd.isna(x) for x in (close, open_, vol, vol50)):
            return sig
        if pd.notna(ma20) and close < ma20:
            sig.append("20DMA↓")
        if pd.notna(ma50) and close < ma50 and vol > 1.5 * vol50:
            sig.append("50DMA↓(大商い)")

        last4 = df.loc[:idx].tail(4)
        lows_desc = _is_strict_down(last4["Low"].tolist())
        last10 = df.loc[:idx].tail(10)
        reds = int((last10["Close"] < last10["Open"]).sum())
        if lows_desc or reds > 5:
            sig.append("連続安値/陰線優勢")

        ups = int((last10["Close"] > last10["Open"]).sum())
        if ups >= 7:
            sig.append("上げ偏重(>70%)")

        last15 = df.loc[:idx].tail(15)
        base0 = _scalar(last15.iloc[0], "Close") if len(last15) > 0 else float("nan")
        if pd.notna(base0) and base0 != 0 and (close / base0 - 1) >= 0.25:
            sig.append("+25%/15日内")

        if len(df.loc[:idx]) >= 2:
            t1, t0 = df.loc[:idx].iloc[-2], df.loc[:idx].iloc[-1]
            t1_high = _scalar(t1, "High")
            t0_open = _scalar(t0, "Open")
            t0_close = _scalar(t0, "Close")
            if all(pd.notna(x) for x in (t1_high, t0_open, t0_close)):
                if (t0_open > t1_high * 1.02) and (t0_close < t0_open):
                    sig.append("GU→陰線")
        return sig
    except Exception:
        return []


def scan_sell_signals(symbols, lookback_days=5):
    """
    直近 lookback_days 日のうち一度でもシグナルが出たら {sym: [(date,[signals]),...]} を返す。
    日付は YYYY-MM-DD。Slackで列挙する。
    """
    out = {}
    for s in symbols:
        df = _yf_df(s)
        if df is None or len(df) < 60:
            continue
        alerts = []
        for idx in df.tail(lookback_days).index:
            tags = _signals_for_day(df, idx)
            if tags:
                alerts.append((idx.strftime("%Y-%m-%d"), tags))
        if alerts:
            out[s] = alerts
    return out


def load_portfolio():
    tickers_path = Path(__file__).with_name("current_tickers.csv")
    with tickers_path.open() as f:
        reader = list(csv.reader(f))
    return [
        {"symbol": sym.strip().upper(), "shares": int(qty), "target_ratio": 1 / len(reader)}
        for sym, qty in reader
    ]


def compute_threshold():
    vix_ma5 = fetch_vix_ma5()
    drift_threshold = 10 if vix_ma5 < 20 else 12 if vix_ma5 < 26 else float("inf")
    return vix_ma5, drift_threshold


def build_dataframe(portfolio):
    for stock in portfolio:
        price = fetch_price(stock["symbol"])
        stock["price"] = price
        stock["value"] = price * stock["shares"]

    df = pd.DataFrame(portfolio)
    total_value = df["value"].sum()
    df["current_ratio"] = df["value"] / total_value
    df["drift"] = df["current_ratio"] - df["target_ratio"]
    df["drift_abs"] = df["drift"].abs()
    total_drift_abs = df["drift_abs"].sum()
    df["adjusted_ratio"] = df["current_ratio"] - df["drift"] / 2
    df["adjustable"] = (
        (df["adjusted_ratio"] * total_value) >= df["price"]
    ) & df["price"].notna() & df["price"].gt(0)
    return df, total_value, total_drift_abs


def simulate(df, total_value, total_drift_abs, drift_threshold):
    alert = drift_threshold != float("inf") and total_drift_abs * 100 > drift_threshold
    if alert:
        df["trade_shares"] = df.apply(
            lambda r: int(round(((r["adjusted_ratio"] * total_value) - r["value"]) / r["price"]))
            if r["adjustable"] and r["price"] > 0 else 0,
            axis=1,
        )
        df["new_shares"] = df["shares"] + df["trade_shares"]
        df["new_value"] = df["new_shares"] * df["price"]
        new_total_value = df["new_value"].sum()
        df["simulated_ratio"] = df["new_value"] / new_total_value
        df["simulated_drift_abs"] = (df["simulated_ratio"] - df["target_ratio"]).abs()
        simulated_total_drift_abs = df["simulated_drift_abs"].sum()
    else:
        df["trade_shares"] = np.nan
        df["new_shares"] = np.nan
        df["new_value"] = np.nan
        new_total_value = np.nan
        df["simulated_ratio"] = np.nan
        df["simulated_drift_abs"] = np.nan
        simulated_total_drift_abs = np.nan
    return df, alert, new_total_value, simulated_total_drift_abs


def prepare_summary(df, total_drift_abs, alert):
    summary = {
        "symbol": "合計",
        "shares": df["shares"].sum(),
        "value": df["value"].sum(),
        "current_ratio": np.nan,
        "drift_abs": total_drift_abs,
    }
    if alert:
        summary["trade_shares"] = np.nan
    # Sort details by evaluation value descending before appending summary
    df = df.sort_values(by="value", ascending=False)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    if alert:
        cols = ["symbol", "shares", "value", "current_ratio", "drift_abs", "trade_shares"]
        df_small = df[cols].copy()
        df_small.columns = ["sym", "qty", "val", "now", "|d|", "Δqty"]
    else:
        cols = ["symbol", "shares", "value", "current_ratio", "drift_abs"]
        df_small = df[cols].copy()
        df_small.columns = ["sym", "qty", "val", "now", "|d|"]
    return df_small


def currency(x):
    return f"${x:,.0f}" if pd.notnull(x) else ""


def formatters_for(alert):
    formatters = {"val": currency, "now": "{:.2%}".format, "|d|": "{:.2%}".format}
    if alert:
        formatters["Δqty"] = "{:.0f}".format
    return formatters


def build_header(vix_ma5, drift_threshold, total_drift_abs, alert, simulated_total_drift_abs):
    header = (
        f"*📈 VIX MA5:* {vix_ma5:.2f}\n"
        f"*📊 ドリフト閾値:* {'🔴(高VIX)' if drift_threshold == float('inf') else str(drift_threshold)+'%'}\n"
        f"*📉 現在のドリフト合計:* {total_drift_abs * 100:.2f}%\n"
    )
    if alert:
        header += f"*🔁 半戻し後ドリフト合計(想定):* {simulated_total_drift_abs * 100:.2f}%\n"
        header += "🚨 *アラート: 発生！！ Δqtyのマイナス銘柄を売却、任意の銘柄を買い増してバランスを取りましょう！*\n"
    else:
        header += "✅ アラートなし\n"
    return header


def send_slack(text):
    SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")
    payload = {"text": text}
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
        print("✅ Slack（Webhook）へ送信しました")
    except Exception as e:
        print(f"⚠️ Slack通知エラー: {e}")


def send_debug(debug_text):
    SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")
    debug_payload = {"text": "```" + debug_text + "```"}
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=debug_payload)
        resp.raise_for_status()
        print("✅ Debug情報をSlackに送信しました")
    except Exception as e:
        print(f"⚠️ Slack通知エラー: {e}")


def main():
    portfolio = load_portfolio()
    symbols = [r["symbol"] for r in portfolio]
    sell_alerts = scan_sell_signals(symbols, lookback_days=5)
    vix_ma5, drift_threshold = compute_threshold()
    df, total_value, total_drift_abs = build_dataframe(portfolio)
    df, alert, new_total_value, simulated_total_drift_abs = simulate(
        df, total_value, total_drift_abs, drift_threshold
    )
    df_small = prepare_summary(df, total_drift_abs, alert)
    if 'df_small' in locals() and isinstance(df_small, pd.DataFrame) and not df_small.empty:
        col_sym = "sym" if "sym" in df_small.columns else ("symbol" if "symbol" in df_small.columns else None)
        if col_sym:
            df_small.insert(0, "⚠", df_small[col_sym].apply(lambda x: "🔴" if x in sell_alerts else ""))
    formatters = formatters_for(alert)
    header = build_header(
        vix_ma5, drift_threshold, total_drift_abs, alert, simulated_total_drift_abs
    )
    if sell_alerts:
        def fmt_pair(date_tags):
            date, tags = date_tags
            return f"{date}:" + "・".join(tags)
        listed = []
        for t, arr in sell_alerts.items():
            listed.append(f"*{t}*（" + ", ".join(fmt_pair(x) for x in arr) + "）")
        hits = ", ".join(listed)
        if "✅ アラートなし" in header:
            header = header.replace(
                "✅ アラートなし",
                f"⚠️ 売りシグナルあり: {len(sell_alerts)}銘柄\n🟥 {hits}",
            )
        else:
            header += f"\n🟥 {hits}"
    table_text = df_small.to_string(formatters=formatters, index=False)
    send_slack(header + "\n```" + table_text + "```")

    if debug_mode:
        debug_cols = [
            "symbol",
            "shares",
            "price",
            "value",
            "current_ratio",
            "drift",
            "drift_abs",
            "adjusted_ratio",
            "adjustable",
            "trade_shares",
            "new_shares",
            "new_value",
            "simulated_ratio",
            "simulated_drift_abs",
        ]
        debug_text = (
            "=== DEBUG: full dataframe ===\n"
            + df[debug_cols].to_string()
            + f"\n\ntotal_value={total_value}, new_total_value={new_total_value}\n"
            + f"total_drift_abs={total_drift_abs}, simulated_total_drift_abs={simulated_total_drift_abs}"
        )
        print("\n" + debug_text)
        send_debug(debug_text)


if __name__ == "__main__":
    main()

