import os, time, sys
import pandas as pd
import yfinance as yf
import requests

TICKERS = os.getenv("YF_PROBE_TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA").split(",")
PERIOD  = os.getenv("YF_PROBE_PERIOD", "180d")
MIN_LEN = int(os.getenv("YF_PROBE_MIN_LEN", "120"))
MAX_NAN_RATIO = float(os.getenv("YF_PROBE_MAX_NAN", "0.15"))
RETRY_ON_EMPTY = int(os.getenv("YF_PROBE_RETRY", "1"))
SLACK_WEBHOOK = (
    os.getenv("SLACK_WEBHOOK_URL")
    or os.getenv("SLACK_WEBHOOK")
    or os.getenv("YF_PROBE_SLACK_WEBHOOK")
)
TIMEOUT_MS_WARN = int(os.getenv("YF_PROBE_TIMEOUT_MS_WARN", "5000"))

def per_ticker_retry(px, bad):
    for t in bad:
        try:
            h = yf.Ticker(t).history(period=PERIOD, auto_adjust=True)["Close"].rename(t)
            if not h.dropna().empty:
                px[t] = h.reindex(px.index) if len(px.index) else h
        except Exception: 
            pass
    return px

def assess(px):
    details, good = [], 0
    for t in TICKERS:
        if t not in px.columns:
            details.append(f"{t}:NF"); continue
        s = px[t]; n = s.shape[0]; nn = s.notna().sum()
        nan_ratio = 1.0 - (nn/n if n else 0.0)
        head_nan = next((i for i,v in enumerate(s) if pd.notna(v)), len(s))
        tail_nan = next((i for i,v in enumerate(reversed(s.tolist())) if pd.notna(v)), len(s))
        status = "OK"
        if nn==0: status="EMPTY"
        elif nn<MIN_LEN or nan_ratio>MAX_NAN_RATIO or head_nan>5 or tail_nan>5: status="BAD"
        if status=="OK": good+=1
        details.append(f"{t}:{status}(len={nn},nan={nan_ratio:.2f})")
    ok_ratio = good/len(TICKERS)
    if good==len(TICKERS): return 0,"HEALTHY","✅",details
    elif ok_ratio>=0.5: return 10,"DEGRADED","⚠️",details
    else: return 20,"DOWN","🛑",details

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
    t0=time.time()
    data=yf.download(TICKERS,period=PERIOD,auto_adjust=True,progress=False,threads=True)
    close=data["Close"] if isinstance(data,pd.DataFrame) and "Close" in data else pd.DataFrame()
    bad=[t for t in TICKERS if (t not in close.columns) or close.get(t,pd.Series(dtype=float)).dropna().empty]
    if bad and RETRY_ON_EMPTY: close=per_ticker_retry(close,bad)
    code,level,emoji,details=assess(close)
    latency=int((time.time()-t0)*1000); speed="🚀" if latency<TIMEOUT_MS_WARN else "🐢"
    summary=f"{emoji} YF_HEALTH {level} ok={len([d for d in details if 'OK' in d])}/{len(TICKERS)} latency={latency}ms {speed}\n" + " | ".join(details)
    print(summary); send_slack(summary); sys.exit(code)

if __name__=="__main__": main()
