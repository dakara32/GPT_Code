import pandas as pd, yfinance as yf
import numpy as np
import requests
import os
import csv
import json
import time
from pathlib import Path
import config

MODE_LABELS_JA = {"NORMAL": "通常", "CAUTION": "警戒", "EMERG": "緊急"}
MODE_EMOJIS = {"NORMAL": "🟢", "CAUTION": "⚠️", "EMERG": "🚨"}
MODE_RANK = {"NORMAL": 0, "CAUTION": 1, "EMERG": 2}

# --- breadth utilities (factor parity) ---
BENCH = "^GSPC"
CAND_PRICE_MAX = 450.0
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_PATH = Path(RESULTS_DIR) / "ts_signal_log.csv"
AUDIT_PATH = Path(RESULTS_DIR) / "ts_eod_audit.csv"


def _state_file():
    return str(Path(RESULTS_DIR) / "breadth_state.json")


def load_mode(default="NORMAL"):
    try:
        m = json.loads(open(_state_file()).read()).get("mode", default)
        return m if m in ("EMERG","CAUTION","NORMAL") else default
    except Exception:
        return default


def save_mode(mode: str):
    try:
        open(_state_file(),"w").write(json.dumps({"mode": mode}))
    except Exception:
        pass


def _read_csv_list(fname):
    p = Path(__file__).with_name(fname)
    if not p.exists(): return []
    return pd.read_csv(p, header=None).iloc[:,0].astype(str).str.upper().tolist()


def _load_universe():
    # exist + candidate を使用。candidate は価格上限で事前フィルタ
    exist = _read_csv_list("current_tickers.csv")
    cand  = _read_csv_list("candidate_tickers.csv")
    cand_info = yf.Tickers(" ".join(cand)) if cand else None
    cand_keep = []
    for t in cand:
        try:
            px = cand_info.tickers[t].fast_info.get("lastPrice", float("inf"))
        except Exception:
            px = float("inf")
        if pd.notna(px) and float(px) <= CAND_PRICE_MAX:
            cand_keep.append(t)
    tickers = sorted(set(exist + cand_keep))
    return exist, cand_keep, tickers


def _fetch_prices_600d(tickers):
    data = yf.download(
        tickers + [BENCH],
        period="600d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    close = data["Close"]
    px = close.dropna(how="all", axis=1).ffill(limit=2)
    spx = close[BENCH].reindex(px.index).ffill()
    return px, spx


def trend_template_breadth_series(px: pd.DataFrame, spx: pd.Series, win_days: int | None = None) -> pd.Series:
    # scorer.py の実装をそのまま移植（ベクトル化版）
    import numpy as np, pandas as pd
    if px is None or px.empty:
        return pd.Series(dtype=int)
    px = px.dropna(how="all", axis=1)
    if win_days and win_days > 0:
        px = px.tail(win_days)
    if px.empty:
        return pd.Series(dtype=int)
    # 欠損吸収
    px = px.ffill(limit=2)
    spx = spx.reindex(px.index).ffill()

    ma50  = px.rolling(50,  min_periods=50).mean()
    ma150 = px.rolling(150, min_periods=150).mean()
    ma200 = px.rolling(200, min_periods=200).mean()

    tt = (px > ma150)
    tt &= (px > ma200)
    tt &= (ma150 > ma200)
    tt &= (ma200 - ma200.shift(21) > 0)
    tt &= (ma50  > ma150)
    tt &= (ma50  > ma200)
    tt &= (px    > ma50)

    lo252 = px.rolling(252, min_periods=252).min()
    hi252 = px.rolling(252, min_periods=252).max()
    tt &= (px.divide(lo252).sub(1.0) >= 0.30)
    tt &= (px >= (0.75 * hi252))

    r12  = px.divide(px.shift(252)).sub(1.0)
    br12 = spx.divide(spx.shift(252)).sub(1.0)
    r1   = px.divide(px.shift(22)).sub(1.0)
    br1  = spx.divide(spx.shift(22)).sub(1.0)
    rs   = 0.7*(r12.sub(br12, axis=0)) + 0.3*(r1.sub(br1, axis=0))
    tt &= (rs >= 0.10)

    return tt.fillna(False).sum(axis=1).astype(int)


def build_breadth_header():
    # factor._build_breadth_lead_lines と同一挙動
    exist, cand, tickers = _load_universe()
    if not tickers:
        return "", "NORMAL", 0
    px, spx = _fetch_prices_600d(tickers)
    win = int(os.getenv("BREADTH_CALIB_WIN_DAYS", "600"))
    C_ts = trend_template_breadth_series(px, spx, win_days=win)
    if C_ts.empty:
        return "", "NORMAL", 0
    warmup = int(os.getenv("BREADTH_WARMUP_DAYS","252"))
    base = C_ts.iloc[warmup:] if len(C_ts)>warmup else C_ts
    C_full = int(C_ts.iloc[-1])

    q05 = int(np.nan_to_num(base.quantile(float(os.getenv("BREADTH_Q_EMERG_IN",  "0.05"))), nan=0.0))
    q20 = int(np.nan_to_num(base.quantile(float(os.getenv("BREADTH_Q_EMERG_OUT", "0.20"))), nan=0.0))
    q60 = int(np.nan_to_num(base.quantile(float(os.getenv("BREADTH_Q_WARN_OUT",  "0.60"))), nan=0.0))

    # G枠サイズ（Breadth基準）
    N_G = config.N_G
    th_in_rec   = max(N_G, q05)
    th_out_rec  = max(int(np.ceil(1.5*N_G)), q20)
    th_norm_rec = max(3*N_G, q60)

    use_calib = os.getenv("BREADTH_USE_CALIB", "true").strip().lower() == "true"
    if use_calib:
        th_in, th_out, th_norm, th_src = th_in_rec, th_out_rec, th_norm_rec, "自動"
    else:
        th_in   = int(os.getenv("GTT_EMERG_IN", str(N_G)))
        th_out  = int(os.getenv("GTT_EMERG_OUT", str(int(1.5*N_G))))
        th_norm = int(os.getenv("GTT_CAUTION_OUT", str(3*N_G)))
        th_src = "手動"

    prev = load_mode("NORMAL")
    if   prev == "EMERG":
        mode = "EMERG"   if (C_full < th_out)  else ("CAUTION" if (C_full < th_norm) else "NORMAL")
    elif prev == "CAUTION":
        mode = "CAUTION" if (C_full < th_norm) else "NORMAL"
    else:
        mode = "EMERG"   if (C_full < th_in)   else ("CAUTION" if (C_full < th_norm) else "NORMAL")
    save_mode(mode)

    mode_ja, emoji = MODE_LABELS_JA.get(mode, mode), MODE_EMOJIS.get(mode, "ℹ️")
    eff_days = len(base)

    lead_lines = [
        f"{emoji} *現在モード: {mode_ja}*",
        f"テンプレ合格本数: *{C_full}本*",
        "しきい値（{0}）".format(th_src),
        f"  ・緊急入り: <{th_in}本",
        f"  ・緊急解除: ≥{th_out}本",
        f"  ・通常復帰: ≥{th_norm}本",
        f"参考指標（過去~{win}営業日, 有効={eff_days}日）",
        f"  ・下位5%: {q05}本",
        f"  ・下位20%: {q20}本",
        f"  ・60%分位: {q60}本",
    ]
    return "```" + "\n".join(lead_lines) + "```", mode, C_full


def _ensure_log_header():
    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", newline="") as f:
            f.write("date,symbol,breach\n")


def _ensure_audit_header():
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIT_PATH.exists():
        with open(AUDIT_PATH, "w", newline="") as f:
            f.write("date,symbol,high60,low_today,baseTS,threshold,breach\n")


def _load_growth_symbols(portfolio: list[dict]) -> list[str]:
    growth = []
    for row in portfolio:
        bucket = str(row.get("bucket", "")).strip().upper()
        if bucket == "G":
            sym = str(row.get("symbol", "")).strip().upper()
            if sym:
                growth.append(sym)
    return sorted(set(growth))


def _upsert_ts_hits(date_str: str, hits: set[str]):
    _ensure_log_header()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=["date", "symbol", "breach"])
    if df.empty:
        df = pd.DataFrame(columns=["date", "symbol", "breach"])
    df = df[df["date"] != date_str]
    if hits:
        add = pd.DataFrame(
            {
                "date": date_str,
                "symbol": sorted({h.upper() for h in hits}),
                "breach": 1,
            }
        )
        df = pd.concat([df, add], ignore_index=True)
    df = df.sort_values(["date", "symbol"])
    df.to_csv(LOG_PATH, index=False)


def _count_unique_hits_5d(today_utc: pd.Timestamp) -> int:
    if not LOG_PATH.exists():
        return 0
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        return 0
    if df.empty or "date" not in df.columns or "symbol" not in df.columns:
        return 0
    if "breach" in df.columns:
        try:
            df = df[df["breach"].astype(int) == 1]
        except Exception:
            df = df[df["breach"] == 1]
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception:
        return 0
    today = today_utc.normalize()
    start = today - pd.offsets.BDay(4)
    mask = (df["date"] >= start) & (df["date"] <= today)
    if not mask.any():
        return 0
    return int(df.loc[mask, "symbol"].str.upper().nunique())


def _combine_modes(mode_a: str, mode_b: str) -> str:
    a = MODE_RANK.get((mode_a or "NORMAL").upper(), 0)
    b = MODE_RANK.get((mode_b or "NORMAL").upper(), 0)
    for mode, rank in MODE_RANK.items():
        if rank == max(a, b):
            return mode
    return "NORMAL"


def _format_mode(mode: str) -> str:
    upper = (mode or "NORMAL").upper()
    return f"{MODE_EMOJIS.get(upper, 'ℹ️')} {MODE_LABELS_JA.get(upper, upper)}"


def _ts_mode_growth_eod(g_syms: list[str], ref_mode: str) -> tuple[str, int, list[str]]:
    now_utc = pd.Timestamp.today(tz="UTC")
    if not g_syms:
        k = _count_unique_hits_5d(now_utc)
        mode1 = "EMERG" if k >= 8 else "CAUTION" if k >= 6 else "NORMAL"
        return mode1, k, []

    try:
        df = yf.download(
            g_syms,
            period="90d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
    except Exception:
        df = None

    hi = lo = None
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            hi = df["High"] if "High" in df.columns else None
            lo = df["Low"] if "Low" in df.columns else None
        except Exception:
            hi = lo = None
        if isinstance(hi, pd.Series):
            hi = hi.to_frame(name=g_syms[0])
        if isinstance(lo, pd.Series):
            lo = lo.to_frame(name=g_syms[0])

    if hi is None or lo is None or hi.empty or lo.empty:
        roll_hi = pd.Series(dtype=float)
        low_today = pd.Series(dtype=float)
    else:
        try:
            roll_hi = hi.rolling(60, min_periods=20).max().tail(1).iloc[0]
            low_today = lo.tail(1).iloc[0]
        except Exception:
            roll_hi = pd.Series(dtype=float)
            low_today = pd.Series(dtype=float)

    base = float(config.TS_BASE_BY_MODE.get((ref_mode or "NORMAL").upper(), 0.15))
    hits = set()
    audit_rows = []
    today = now_utc.date().isoformat()
    _ensure_audit_header()

    def _fmt(val: float) -> str:
        if pd.isna(val):
            return ""
        return f"{float(val):.6g}"

    for s in g_syms:
        rh = float(roll_hi.get(s, float("nan"))) if hasattr(roll_hi, "get") else float("nan")
        lt = float(low_today.get(s, float("nan"))) if hasattr(low_today, "get") else float("nan")
        threshold = float("nan")
        breach = 0
        if pd.notna(rh) and rh > 0 and pd.notna(lt) and lt > 0:
            threshold = rh * (1.0 - base)
            breach = int(lt <= threshold)
            if breach:
                hits.add(s)
        audit_rows.append(
            {
                "date": today,
                "symbol": s,
                "high60": _fmt(rh),
                "low_today": _fmt(lt),
                "baseTS": f"{base:.3f}",
                "threshold": _fmt(threshold),
                "breach": str(breach),
            }
        )

    if audit_rows:
        with open(AUDIT_PATH, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "symbol", "high60", "low_today", "baseTS", "threshold", "breach"],
            )
            writer.writerows(audit_rows)

    _upsert_ts_hits(today, hits)
    k = _count_unique_hits_5d(now_utc)
    mode1 = "EMERG" if k >= 8 else "CAUTION" if k >= 6 else "NORMAL"
    return mode1, k, sorted(hits)
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
        ma20 = _scalar(d, "ma20")
        ma50 = _scalar(d, "ma50")
        vol = _scalar(d, "Volume")
        vol50 = _scalar(d, "vol50")

        if pd.notna(close) and pd.notna(ma20) and close < ma20:
            sig.append("20DMA↓")

        if all(pd.notna(x) for x in (close, ma50, vol, vol50)) and close < ma50 and vol > 1.5 * vol50:
            sig.append("50DMA↓(大商い)")

        last4 = df.loc[:idx].tail(4)
        last10 = df.loc[:idx].tail(10)

        lows_desc = _is_strict_down(last4["Low"].tolist()) if last4["Low"].notna().all() else False
        reds = int((last10["Close"] < last10["Open"]).sum()) if last10[["Close", "Open"]].notna().all().all() else 0
        if lows_desc or reds > 5:
            sig.append("連続安値/陰線優勢")

        ups = int((last10["Close"] > last10["Open"]).sum()) if last10[["Close", "Open"]].notna().all().all() else 0
        if ups >= 7:
            sig.append("上げ偏重(>70%)")

        last15 = df.loc[:idx].tail(15)
        base0 = _scalar(last15.iloc[0], "Close") if len(last15) > 0 else float("nan")
        if pd.notna(base0) and pd.notna(close) and base0 != 0 and (close / base0 - 1) >= 0.25:
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
        rows = [row for row in csv.reader(f) if row and row[0].strip()]
    n = len(rows)
    portfolio = []
    for row in rows:
        sym = row[0].strip().upper()
        qty = int(row[1]) if len(row) > 1 and row[1].strip() else 0
        bucket = row[2].strip().upper() if len(row) > 2 else ""
        entry = {
            "symbol": sym,
            "shares": qty,
            "target_ratio": 1 / n if n else 0.0,
            "bucket": bucket,
        }
        portfolio.append(entry)
    return portfolio


def compute_threshold():
    vix_ma5 = fetch_vix_ma5()
    drift_threshold = 10 if vix_ma5 < 20 else 12 if vix_ma5 < 26 else float("inf")
    return vix_ma5, drift_threshold


def compute_threshold_by_mode(mode: str):
    """モードに応じて現金保有率とドリフト閾値を返す（README準拠）"""
    m = (mode or "NORMAL").upper()
    cash_map = {"NORMAL": 0.10, "CAUTION": 0.125, "EMERG": 0.20}
    drift_map = config.DRIFT_THRESHOLD_BY_MODE
    return cash_map.get(m, 0.10), drift_map.get(m, 12)


def recommended_counts_by_mode(mode: str) -> tuple[int, int, int]:
    """
    モード別の推奨保有数 (G_count, D_count, cash_slots) を返す。
    cash_slotsは「外すG枠の数」（各枠=5%）。
    NORMAL: G12/D8/現金化0, CAUTION: G10/D8/現金化2, EMERG: G8/D8/現金化4
    """
    m = (mode or "NORMAL").upper()
    base = config.COUNTS_BY_MODE.get("NORMAL", config.COUNTS_BASE)
    now  = config.COUNTS_BY_MODE.get(m, base)
    cash_slots = max(0, base["G"] - now["G"])
    return now["G"], now["D"], cash_slots


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


def build_header(mode, cash_ratio, drift_threshold, total_drift_abs, alert, simulated_total_drift_abs):
    header = (
        f"*💼 現金保有率:* {cash_ratio*100:.1f}%\n"
        f"*📊 ドリフト閾値:* {'🔴(停止)' if drift_threshold == float('inf') else str(drift_threshold)+'%'}\n"
        f"*📉 現在のドリフト合計:* {total_drift_abs * 100:.2f}%\n"
    )
    if alert:
        header += f"*🔁 半戻し後ドリフト合計(想定):* {simulated_total_drift_abs * 100:.2f}%\n"
        header += "🚨 *アラート: 発生！！ Δqtyのマイナス銘柄を売却、任意の銘柄を買い増してバランスを取りましょう！*\n"
    else:
        header += "✅ アラートなし\n"
    # ★ 追記: TSルール（G/D共通）と推奨保有数
    # TS(基本)をモードで動的表示。段階TSは「基本から -3/-6/-8 pt」固定。
    base_ts = config.TS_BASE_BY_MODE.get(mode.upper(), config.TS_BASE_BY_MODE["NORMAL"])
    d1, d2, d3 = config.TS_STEP_DELTAS_PT
    ts_line = f"*🛡 TS:* 基本 -{base_ts*100:.0f}% / +30%→-{max(base_ts*100 - d1, 0):.0f}% / +60%→-{max(base_ts*100 - d2, 0):.0f}% / +100%→-{max(base_ts*100 - d3, 0):.0f}%\n"
    header += ts_line
    g_cnt, d_cnt, cash_slots = recommended_counts_by_mode(mode)
    cash_pct = cash_slots * (100 / (config.TOTAL_TARGETS))  # 1枠=総数分割の%（20銘柄なら5%）
    header += f"*📋 推奨保有数:* G {g_cnt} / D {d_cnt}（現金化枠 {cash_slots}枠 ≒ {cash_pct:.0f}%）\n"
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
    g_syms = _load_growth_symbols(portfolio)
    sell_alerts = scan_sell_signals(symbols, lookback_days=5)

    breadth_block, breadth_mode, breadth_score = build_breadth_header()
    ts_mode, k5, ts_hits = _ts_mode_growth_eod(g_syms, breadth_mode)
    combo_mode = _combine_modes(ts_mode, breadth_mode)

    cash_ratio, drift_threshold = compute_threshold_by_mode(breadth_mode)

    df, total_value, total_drift_abs = build_dataframe(portfolio)
    df, alert, new_total_value, simulated_total_drift_abs = simulate(
        df, total_value, total_drift_abs, drift_threshold
    )
    df_small = prepare_summary(df, total_drift_abs, alert)
    if 'df_small' in locals() and isinstance(df_small, pd.DataFrame) and not df_small.empty:
        col_sym = "sym" if "sym" in df_small.columns else ("symbol" if "symbol" in df_small.columns else None)
        if col_sym:
            alert_keys = {str(k) for k in sell_alerts.keys()}
            df_small[col_sym] = df_small[col_sym].astype(str)
            df_small.insert(0, "⚠", df_small[col_sym].map(lambda x: "🔴" if x in alert_keys else ""))
            latest_tag = {s: " / ".join(sell_alerts[s][-1][1]) for s in sell_alerts}
            df_small.insert(1, "sig", df_small[col_sym].map(latest_tag).fillna(""))
    formatters = formatters_for(alert)
    header_core = build_header(
        breadth_mode, cash_ratio, drift_threshold, total_drift_abs, alert, simulated_total_drift_abs
    )

    g_count = len(g_syms)
    hits_line = "なし" if not ts_hits else ", ".join(sorted(ts_hits))
    summary_lines = [
        f"① Growth TS: {_format_mode(ts_mode)} （5Dユニーク: {k5} / G={g_count}）",
        f"・当日ヒット: {hits_line}",
        f"② Breadth: {_format_mode(breadth_mode)} （テンプレ合格本数: {breadth_score}）",
        f"総合（OR悪化/AND回復）: {_format_mode(combo_mode)}",
    ]
    prepend_block = "\n".join(summary_lines)

    if breadth_block:
        if breadth_block.startswith("```"):
            remainder = breadth_block[len("```") :]
            if remainder.startswith("\n"):
                remainder = remainder[1:]
            breadth_block = "```\n" + prepend_block + "\n" + remainder
        else:
            breadth_block = prepend_block + "\n" + breadth_block
        header = breadth_block + "\n" + header_core
    else:
        header = prepend_block + "\n" + header_core
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

