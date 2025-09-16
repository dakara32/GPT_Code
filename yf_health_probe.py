import os
import sys
import time

import pandas as pd
import yfinance as yf

TICKERS = os.getenv("YF_PROBE_TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA").split(",")
PERIOD = os.getenv("YF_PROBE_PERIOD", "180d")
MIN_LEN = int(os.getenv("YF_PROBE_MIN_LEN", "120"))
MAX_NAN_RATIO = float(os.getenv("YF_PROBE_MAX_NAN", "0.15"))
RETRY_ON_EMPTY = int(os.getenv("YF_PROBE_RETRY", "1"))
SLACK_WEBHOOK = os.getenv("YF_PROBE_SLACK_WEBHOOK", "")
TIMEOUT_MS_WARN = int(os.getenv("YF_PROBE_TIMEOUT_MS_WARN", "5000"))


def per_ticker_retry(px: pd.DataFrame, bad: list[str]) -> pd.DataFrame:
    """Retry fetching history for tickers with missing data."""
    for ticker in bad:
        try:
            history = (
                yf.Ticker(ticker)
                .history(period=PERIOD, auto_adjust=True)["Close"]
                .rename(ticker)
            )
            if not history.dropna().empty:
                px[ticker] = history.reindex(px.index) if len(px.index) else history
        except Exception:  # pragma: no cover - best effort retry
            pass
    return px


def assess(px: pd.DataFrame) -> tuple[int, str, str, list[str]]:
    """Assess data quality for each ticker."""
    details: list[str] = []
    good = 0
    for ticker in TICKERS:
        if ticker not in px.columns:
            details.append(f"{ticker}:NF")
            continue

        series = px[ticker]
        total = series.shape[0]
        non_null = series.notna().sum()
        nan_ratio = 1.0 - (non_null / total if total else 0.0)
        head_nan = next((i for i, value in enumerate(series) if pd.notna(value)), len(series))
        tail_nan = next(
            (i for i, value in enumerate(reversed(series.tolist())) if pd.notna(value)),
            len(series),
        )

        status = "OK"
        if non_null == 0:
            status = "EMPTY"
        elif (
            non_null < MIN_LEN
            or nan_ratio > MAX_NAN_RATIO
            or head_nan > 5
            or tail_nan > 5
        ):
            status = "BAD"

        if status == "OK":
            good += 1
        details.append(f"{ticker}:{status}(len={non_null},nan={nan_ratio:.2f})")

    ok_ratio = good / len(TICKERS)
    if good == len(TICKERS):
        return 0, "HEALTHY", "✅", details
    if ok_ratio >= 0.5:
        return 10, "DEGRADED", "⚠️", details
    return 20, "DOWN", "🛑", details


def post_slack(text: str) -> None:
    """Send message to Slack webhook if configured."""
    if not SLACK_WEBHOOK:
        return
    try:
        import requests

        requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=5)
    except Exception:  # pragma: no cover - avoid crashing on notification failure
        pass


def main() -> None:
    t0 = time.time()
    data = yf.download(
        TICKERS,
        period=PERIOD,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data else pd.DataFrame()
    bad = [
        ticker
        for ticker in TICKERS
        if (ticker not in close.columns)
        or close.get(ticker, pd.Series(dtype=float)).dropna().empty
    ]
    if bad and RETRY_ON_EMPTY:
        close = per_ticker_retry(close, bad)
    code, level, emoji, details = assess(close)
    latency = int((time.time() - t0) * 1000)
    speed = "🚀" if latency < TIMEOUT_MS_WARN else "🐢"
    summary = (
        f"{emoji} YF_HEALTH {level} "
        f"ok={len([detail for detail in details if 'OK' in detail])}/{len(TICKERS)} "
        f"latency={latency}ms {speed}\n" + " | ".join(details)
    )
    print(summary)
    post_slack(summary)
    sys.exit(code)


if __name__ == "__main__":
    main()
