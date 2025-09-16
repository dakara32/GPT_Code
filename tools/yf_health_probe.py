import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests
import yfinance as yf

DEFAULT_FALLBACK = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
PERIOD = os.getenv("YF_PROBE_PERIOD", "180d")
MIN_LEN = int(os.getenv("YF_PROBE_MIN_LEN", "120"))
MAX_NAN_RATIO = float(os.getenv("YF_PROBE_MAX_NAN", "0.15"))
RETRY_ON_EMPTY = int(os.getenv("YF_PROBE_RETRY", "1"))
TIMEOUT_MS_WARN = int(os.getenv("YF_PROBE_TIMEOUT_MS_WARN", "5000"))
END_OFFSET_DAYS = int(os.getenv("END_OFFSET_DAYS", "1"))

SLACK_WEBHOOK = (
    os.getenv("SLACK_WEBHOOK_URL")
    or os.getenv("SLACK_WEBHOOK")
    or os.getenv("YF_PROBE_SLACK_WEBHOOK")
)


def _parse_tickers(text: str) -> List[str]:
    raw = [x.strip().upper() for x in text.replace(",", "\n").splitlines()]
    return [ticker for ticker in raw if ticker and not ticker.startswith("#")]


def load_candidates() -> List[str]:
    """Load candidate tickers from env, file, or fallback list."""

    env_list = os.getenv("CAND_TICKERS", "").strip()
    if env_list:
        tickers = _parse_tickers(env_list)
        if tickers:
            return tickers

    path = os.getenv("CAND_FILE", "data/candidates.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                tickers = _parse_tickers(file.read())
                if tickers:
                    return tickers
        except Exception:
            pass

    return DEFAULT_FALLBACK


def per_ticker_retry(px: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    for ticker in tickers:
        try:
            history = (
                yf.Ticker(ticker)
                .history(period=PERIOD, auto_adjust=True)["Close"]
                .rename(ticker)
            )
            if not history.dropna().empty:
                px[ticker] = history.reindex(px.index) if len(px.index) else history
        except Exception:
            pass
    return px


def assess(px: pd.DataFrame, tickers: List[str]):
    details: List[str] = []
    good = 0
    for ticker in tickers:
        if ticker not in px.columns:
            details.append(f"{ticker}:NF")
            continue

        series = px[ticker]
        total = series.shape[0]
        non_nan = series.notna().sum()
        nan_ratio = 1.0 - (non_nan / total if total else 0.0)
        head_nan = next((i for i, value in enumerate(series) if pd.notna(value)), len(series))
        tail_nan = next(
            (i for i, value in enumerate(reversed(series.tolist())) if pd.notna(value)),
            len(series),
        )

        status = "OK"
        if non_nan == 0:
            status = "EMPTY"
        elif (
            non_nan < MIN_LEN
            or nan_ratio > MAX_NAN_RATIO
            or head_nan > 5
            or tail_nan > 5
        ):
            status = "BAD"

        if status == "OK":
            good += 1

        details.append(f"{ticker}:{status}(len={non_nan},nan={nan_ratio:.2f})")

    ok_ratio = good / len(tickers) if tickers else 0.0
    if good == len(tickers):
        return 0, "HEALTHY", "✅", details
    if ok_ratio >= 0.5:
        return 10, "DEGRADED", "⚠️", details
    return 20, "DOWN", "🛑", details


def send_slack(text: str) -> None:
    if not SLACK_WEBHOOK:
        print("[SLACK] Missing webhook. Set 'SLACK_WEBHOOK_URL'.")
        sys.exit(78)

    response = requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=5)
    print(f"[SLACK] status={response.status_code}")
    response.raise_for_status()


def main() -> None:
    tickers = load_candidates()
    if not tickers:
        print("[ERR] no tickers")
        sys.exit(78)

    def preview(items: List[str], show: int = 12) -> str:
        head = ",".join(items[:show])
        return head + (" …" if len(items) > show else "")

    print(f"[UNIVERSE] {len(tickers)} tickers: {preview(tickers)}")

    start = time.time()
    kwargs = dict(period=PERIOD, auto_adjust=True, progress=False, threads=True)
    if END_OFFSET_DAYS > 0:
        kwargs["end"] = (datetime.now(timezone.utc) - timedelta(days=END_OFFSET_DAYS)).date()

    data = yf.download(tickers, **kwargs)
    close = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data else pd.DataFrame()

    bad = [
        ticker
        for ticker in tickers
        if (ticker not in close.columns)
        or close.get(ticker, pd.Series(dtype=float)).dropna().empty
    ]
    if bad and RETRY_ON_EMPTY:
        close = per_ticker_retry(close, bad)

    code, level, emoji, details = assess(close, tickers)
    latency = int((time.time() - start) * 1000)
    ok_count = sum(1 for detail in details if ":OK(" in detail)
    speed = "🚀" if latency < TIMEOUT_MS_WARN else "🐢"
    universe_note = f"universe={len(tickers)} [{preview(tickers)}]"
    summary = (
        f"{emoji} YF_HEALTH {level} ok={ok_count}/{len(tickers)} latency={latency}ms {speed}\n"
        f"{universe_note}\n" + " | ".join(details)
    )

    print(summary)
    send_slack(summary)
    sys.exit(code)


if __name__ == "__main__":
    main()
