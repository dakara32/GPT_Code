# file: tools/yf_health_probe.py など
import os, time, sys
import pandas as pd
import yfinance as yf
import requests

TICKERS = os.getenv(
    "YF_PROBE_TICKERS",
    (
        "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,BRK-B,JPM,JNJ,V,UNH,XOM,PG,MA,HD,AVGO,"
        "KO,PEP,MRK,ABBV,COST,CRM,DIS,NFLX,AMD,ORCL,INTC,TSM,NKE"
    ),
).split(",")
PERIOD  = os.getenv("YF_PROBE_PERIOD", "180d")
MIN_LEN = int(os.getenv("YF_PROBE_MIN_LEN", "120"))
MAX_NAN_RATIO = float(os.getenv("YF_PROBE_MAX_NAN", "0.15"))  # 参考値として残す
RETRY_ON_EMPTY = int(os.getenv("YF_PROBE_RETRY", "1"))
SLACK_WEBHOOK = (
    os.getenv("SLACK_WEBHOOK_URL")
    or os.getenv("SLACK_WEBHOOK")
    or os.getenv("YF_PROBE_SLACK_WEBHOOK")
)
TIMEOUT_MS_WARN = int(os.getenv("YF_PROBE_TIMEOUT_MS_WARN", "5000"))

# 追加: 欠損検知の詳細閾値（任意）
MAX_HEAD_TAIL_NAN = int(os.getenv("YF_PROBE_MAX_HEADTAIL_N", "0"))  # 先頭/末尾の連続NaN許容（既定0=1つでもNG）
MAX_INTERNAL_NAN  = int(os.getenv("YF_PROBE_MAX_INTERNAL_N", "0"))  # 途中の連続NaN許容（既定0）

def per_ticker_retry(px, bad):
    for t in bad:
        try:
            h = yf.Ticker(t).history(period=PERIOD, auto_adjust=True)["Close"].rename(t)
            if not h.dropna().empty:
                px[t] = h.reindex(px.index) if len(px.index) else h
        except Exception:
            pass
    return px

# 追加: 連続NaNの本数/最大長と最初/最後のNaN日を計算
def _nan_profile(s: pd.Series):
    if s.empty:
        return dict(n_gaps=0, max_gap=0, first_nan=None, last_nan=None,
                    head_run=0, tail_run=0, total_nan=0)
    isna = s.isna().values
    total_nan = int(isna.sum())
    # 先頭/末尾の連続NaN
    head_run = 0
    for v in isna:
        if v: head_run += 1
        else: break
    tail_run = 0
    for v in isna[::-1]:
        if v: tail_run += 1
        else: break
    # 途中の連続NaN（区間数と最長）
    n_gaps = 0
    max_gap = 0
    cur = 0
    for v in isna:
        if v:
            cur += 1
        else:
            if cur > 0:
                n_gaps += 1
                if cur > max_gap: max_gap = cur
                cur = 0
    if cur > 0:
        n_gaps += 1
        if cur > max_gap: max_gap = cur
    # 最初/最後のNaN日
    first_nan = s.index[isna.argmax()] if total_nan > 0 else None
    last_nan  = s.index[::-1][isna[::-1].argmax()] if total_nan > 0 else None
    return dict(n_gaps=n_gaps, max_gap=int(max_gap), first_nan=first_nan, last_nan=last_nan,
                head_run=int(head_run), tail_run=int(tail_run), total_nan=int(total_nan))

def assess(px):
    details, good = [], 0
    missing_brief = []  # Slack用の簡易まとめ
    for t in TICKERS:
        if t not in px.columns:
            details.append(f"{t}:NF")
            missing_brief.append(f"{t}:NF")
            continue
        s = px[t]
        n = s.shape[0]
        nn = int(s.notna().sum())
        nan_ratio = 1.0 - (nn / n if n else 0.0)
        prof = _nan_profile(s)

        # ステータス判定
        if nn == 0:
            status = "EMPTY"
        elif prof["total_nan"] > 0:
            status = "MISSING"  # Close列にNaNが1つでもあれば欠損扱い
            # 追加の安全弾（既存基準も併用）:
            if nn < MIN_LEN or nan_ratio > MAX_NAN_RATIO or \
               prof["head_run"] > MAX_HEAD_TAIL_NAN or prof["tail_run"] > MAX_HEAD_TAIL_NAN or \
               prof["max_gap"]  > MAX_INTERNAL_NAN:
                status = "MISSING"
        else:
            # NaNゼロでも長さが短い等はBAD
            status = "OK" if nn >= MIN_LEN else "BAD"

        if status == "OK":
            good += 1
        else:
            # Slackの簡易欄: 最長欠損長と先頭/末尾NaNを付ける
            if status in ("MISSING", "EMPTY"):
                brief = f"{t}:{status}"
                if status == "MISSING":
                    extra = []
                    if prof["max_gap"]>0: extra.append(f"maxGap={prof['max_gap']}")
                    if prof["head_run"]>0: extra.append(f"head={prof['head_run']}")
                    if prof["tail_run"]>0: extra.append(f"tail={prof['tail_run']}")
                    if prof["first_nan"]:  extra.append(f"first={prof['first_nan'].date()}")
                    if prof["last_nan"]:   extra.append(f"last={prof['last_nan'].date()}")
                    if extra: brief += "(" + ",".join(extra) + ")"
                missing_brief.append(brief)

        # 詳細欄
        det = f"{t}:{status}(len={nn},nan={nan_ratio:.2f}"
        if prof["total_nan"]>0:
            det += f",gaps={prof['n_gaps']},maxGap={prof['max_gap']},head={prof['head_run']},tail={prof['tail_run']}"
        det += ")"
        details.append(det)

    ok_ratio = good / len(TICKERS) if TICKERS else 0.0
    if good == len(TICKERS):
        code, level, emoji = 0, "HEALTHY", "✅"
    elif ok_ratio >= 0.5:
        code, level, emoji = 10, "DEGRADED", "⚠️"
    else:
        code, level, emoji = 20, "DOWN", "🛑"

    return code, level, emoji, details, missing_brief

def send_slack(text):
    if not SLACK_WEBHOOK:
        print("[SLACK] Missing webhook. Set 'SLACK_WEBHOOK_URL' (preferred) or 'SLACK_WEBHOOK'.")
        sys.exit(78)
    try:
        r = requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=5)
        print(f"[SLACK] status={r.status_code}")
        r.raise_for_status()
    except Exception as e:
        print(f"[SLACK] send error: {e}")

def main():
    t0 = time.time()
    data = yf.download(TICKERS, period=PERIOD, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data else pd.DataFrame()

    bad = [t for t in TICKERS if (t not in close.columns) or close.get(t, pd.Series(dtype=float)).dropna().empty]
    if bad and RETRY_ON_EMPTY:
        close = per_ticker_retry(close, bad)

    code, level, emoji, details, missing_brief = assess(close)
    latency = int((time.time() - t0) * 1000)
    speed = "🚀" if latency < TIMEOUT_MS_WARN else "🐢"

    # Slack本文：先頭に欠損サマリを明示
    missing_line = ""
    if missing_brief:
        # 先頭10件だけ表示、残りは省略
        head = " | ".join(missing_brief[:10])
        if len(missing_brief) > 10:
            head += f" …(+{len(missing_brief)-10})"
        missing_line = f"\n❗Missing Close: {head}"

    summary = (
        f"{emoji} YF_HEALTH {level} ok={len([d for d in details if ':OK(' in d])}/{len(TICKERS)} "
        f"latency={latency}ms {speed}"
        f"{missing_line}\n" + " | ".join(details)
    )

    print(summary)
    send_slack(summary)
    sys.exit(code)

if __name__ == "__main__":
    main()