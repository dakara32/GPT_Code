import pandas as pd, yfinance as yf
import numpy as np
import requests
import os
import json
import time
from pathlib import Path
import csv
import config

# --- GコンポジットDDのしきい値（Growthの平均DD基準）---
CD_CAUTION = 0.10   # -10% で警戒
CD_EMERG = 0.15   # -15% で緊急

MODE_LABELS_JA = {"NORMAL": "通常", "CAUTION": "警戒", "EMERG": "緊急"}
# Slack通知用のモードアイコン
MODE_EMOJIS = {"NORMAL": "🟢", "CAUTION": "⚠️", "EMERG": "🔴"}
MODE_RANK = {"NORMAL": 0, "CAUTION": 1, "EMERG": 2}

# --- breadth utilities (factor parity) ---
BENCH = "^GSPC"
CAND_PRICE_MAX = 450.0
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _state_file():
    return str(Path(RESULTS_DIR) / "breadth_state.json")


def _load_state_dict() -> dict:
    try:
        with open(_state_file()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state_dict(state: dict):
    try:
        with open(_state_file(), "w") as fh:
            json.dump(state, fh)
    except Exception:
        pass


def load_breadth_mode(default: str = "NORMAL") -> str:
    state = _load_state_dict()
    mode = state.get("breadth_mode", state.get("mode", default))
    return mode if mode in MODE_RANK else default


def save_breadth_mode(mode: str):
    state = _load_state_dict()
    state["breadth_mode"] = mode
    _save_state_dict(state)


def load_final_mode(default: str = "NORMAL") -> str:
    state = _load_state_dict()
    mode = state.get("final_mode", state.get("mode", default))
    return mode if mode in MODE_RANK else default


def save_final_mode(mode: str):
    state = _load_state_dict()
    state["final_mode"] = mode
    state.setdefault("breadth_mode", state.get("breadth_mode", mode))
    state["mode"] = mode
    _save_state_dict(state)


def _read_csv_list(fname):
    p = Path(__file__).with_name(fname)
    if not p.exists(): return []
    return pd.read_csv(p, header=None).iloc[:,0].astype(str).str.upper().tolist()


# leaders.csv 読み込み（results/leaders.csv, 1列想定）
def _read_leaders_symbols() -> list[str]:
    p = Path(__file__).with_name("results").joinpath("leaders.csv")
    df = pd.read_csv(p, header=None)
    return sorted(set(df.iloc[:,0].astype(str).str.strip().str.upper().tolist()))

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

    prev = load_breadth_mode("NORMAL")
    if   prev == "EMERG":
        mode = "EMERG"   if (C_full < th_out)  else ("CAUTION" if (C_full < th_norm) else "NORMAL")
    elif prev == "CAUTION":
        mode = "CAUTION" if (C_full < th_norm) else "NORMAL"
    else:
        mode = "EMERG"   if (C_full < th_in)   else ("CAUTION" if (C_full < th_norm) else "NORMAL")
    save_breadth_mode(mode)

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


def _format_mode(mode: str) -> str:
    upper = (mode or "NORMAL").upper()
    return f"{MODE_EMOJIS.get(upper, 'ℹ️')} {MODE_LABELS_JA.get(upper, upper)}"


def _gcd_mode_today(g_syms: list[str]) -> tuple[str, float]:
    """
    現在のGrowth群について、Low_today / Peak60(High) の等加重平均から G-CD(%) を算出し、モードを返す。
    戻り値: (gcd_mode, gcd_pct)  ※gcd_pctは正の%（例 11.3 は -11.3%の下落）
    """

    if not g_syms:
        print("📝 audit[G-CD details]: G銘柄が空のため算出対象がありません")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    try:
        df = yf.download(
            g_syms,
            period="100d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"⚠️ audit[G-CD details]: 株価データ取得に失敗しました ({e})")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("⚠️ audit[G-CD details]: 株価データが空のため算出できません")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    hi_all = df.get("High") if isinstance(df, pd.DataFrame) else None
    lo_all = df.get("Low") if isinstance(df, pd.DataFrame) else None
    if hi_all is None or lo_all is None:
        print("⚠️ audit[G-CD details]: High/Low データが欠落しています")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    if isinstance(hi_all, pd.Series):
        hi_all = hi_all.to_frame(name=g_syms[0])
    if isinstance(lo_all, pd.Series):
        lo_all = lo_all.to_frame(name=g_syms[0])

    if hi_all.empty or lo_all.empty:
        print("⚠️ audit[G-CD details]: High/Low データが空のため算出できません")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    peak60 = hi_all.rolling(60, min_periods=20).max().tail(1).iloc[0]
    low_today = lo_all.tail(1).iloc[0]

    details: list[tuple[str, float, float, float, float]] = []
    for sym in g_syms:
        p = float(peak60.get(sym, float("nan"))) if hasattr(peak60, "get") else float("nan")
        lt = float(low_today.get(sym, float("nan"))) if hasattr(low_today, "get") else float("nan")
        if pd.notna(p) and p > 0 and pd.notna(lt) and lt > 0:
            ratio = lt / p
            ddpct = (1.0 - ratio) * 100.0
            details.append((sym, p, lt, ratio, ddpct))

    if not details:
        print("⚠️ audit[G-CD details]: 有効な銘柄データがありません")
        print("📝 audit[G-CD summary]: avg_low/peak60=1.0000  drawdown=0.00%  => NORMAL")
        return "NORMAL", 0.0

    details.sort(key=lambda x: x[4], reverse=True)
    today = pd.Timestamp.today(tz="America/New_York").date().isoformat()
    print(f"📝 audit[G-CD details] {today}  G={len(g_syms)}")
    print("  SYMBOL        Peak60(H)     Low(T)     ratio    DD%")
    for sym, peak, low, ratio, ddpct in details:
        print(f"  {sym:<8}  {peak:>12.6g}  {low:>10.6g}   {ratio:>6.3f}  {ddpct:>6.2f}")

    avg_ratio = float(np.mean([r for _, _, _, r, _ in details]))
    gcd_pct = max(0.0, (1.0 - avg_ratio) * 100.0)
    mode = "EMERG" if gcd_pct >= CD_EMERG * 100 else "CAUTION" if gcd_pct >= CD_CAUTION * 100 else "NORMAL"
    print(
        f"📝 audit[G-CD summary]: avg_low/peak60={avg_ratio:.4f}  drawdown={gcd_pct:.2f}%  => {mode}"
    )
    return mode, gcd_pct
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
    cash_ratio = config.CASH_RATIO_BY_MODE.get(
        m, config.CASH_RATIO_BY_MODE.get("NORMAL", 0.10)
    )
    drift_threshold = config.DRIFT_THRESHOLD_BY_MODE.get(
        m, config.DRIFT_THRESHOLD_BY_MODE.get("NORMAL", 12)
    )
    return cash_ratio, drift_threshold


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


def _mode_tail_line(final_mode: str) -> str:
    """①ブロック表示：改行＋アイコンで整形"""
    fm = (final_mode or "NORMAL").upper()
    base_ts = config.TS_BASE_BY_MODE.get(fm, config.TS_BASE_BY_MODE.get("NORMAL", 0.15))
    ts_base_pct = int(round(base_ts * 100))
    d1, d2, d3 = config.TS_STEP_DELTAS_PT
    step30 = max(ts_base_pct - d1, 0)
    step60 = max(ts_base_pct - d2, 0)
    step100 = max(ts_base_pct - d3, 0)
    g_cnt, d_cnt, cash_slots = recommended_counts_by_mode(fm)
    cash_pct = config.CASH_RATIO_BY_MODE.get(fm, config.CASH_RATIO_BY_MODE.get("NORMAL", 0.10)) * 100
    drift_th = config.DRIFT_THRESHOLD_BY_MODE.get(fm, config.DRIFT_THRESHOLD_BY_MODE.get("NORMAL", 12))
    drift_str = "🔴(停止)" if drift_th == float("inf") else f"{int(drift_th)}%"
    return "\n".join([
        "〔このモードの設定〕",
        f"🎯 TS基本: -{ts_base_pct}％（+30%→-{step30}％／+60%→-{step60}％／+100%→-{step100}％）",
        f"🧩 推奨保有: G{g_cnt}・D{d_cnt}（現金化枠 {cash_slots}）",
        f"💼 推奨現金比率: {cash_pct:.0f}％",
        f"📊 ドリフト閾値: {drift_str}",
    ])


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
    # 💼は①に集約し非表示。📊は維持。
    header  = f"*📊 ドリフト閾値:* {'🔴(停止)' if drift_threshold == float('inf') else str(int(drift_threshold)) + '%'}\n"
    header += f"*📉 現在のドリフト合計:* {total_drift_abs * 100:.2f}%\n"
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
    # G集合は leaders.csv を使用（存在前提）
    g_syms = _read_leaders_symbols()
    sell_alerts = scan_sell_signals(symbols, lookback_days=5)

    breadth_block, breadth_mode, breadth_score = build_breadth_header()
    gcd_mode, gcd_pct = _gcd_mode_today(g_syms)

    # モードは GコンポジットDD のみで決定（シンプル化）
    final_mode = gcd_mode
    save_final_mode(final_mode)

    cash_ratio, drift_threshold = compute_threshold_by_mode(final_mode)

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
        final_mode, cash_ratio, drift_threshold, total_drift_abs, alert, simulated_total_drift_abs
    )

    # --- Slack 送信：①ブロック（判定＋このモードの設定〜推奨現金比率）を独立、②以降は別ブロック ---
    me_g = MODE_EMOJIS.get(gcd_mode, "")
    me_b = MODE_EMOJIS.get(breadth_mode, "")
    block_gcd = (
        f"① GコンポジットDD: -{gcd_pct:.1f}%"
        f"（基準: C={CD_CAUTION*100:.0f}% / E={CD_EMERG*100:.0f}%） 判定: {me_g} {gcd_mode}"
    )
    # ①ブロック：ここまで＋このモードの設定〜推奨現金比率まで
    first_block = "```\n" + block_gcd + "\n" + _mode_tail_line(final_mode) + "\n```"

    # ②以降ブロック：Breadthのみ（“総合（参考表示）”は削除）
    block_breadth = f"② Breadth: {me_b} {breadth_mode}（テンプレ合格本数: {breadth_score}）"
    # breadth_block の中身（コードフェンス除去＋重複行は除去）
    breadth_details = ""
    if breadth_block:
        inner = breadth_block
        if inner.startswith("```"):
            inner = inner[len("```"):]
            if inner.startswith("\n"):
                inner = inner[1:]
            if inner.endswith("```"):
                inner = inner[:-3]
        inner_lines = [ln for ln in inner.splitlines() if ("現在モード" not in ln and "テンプレ合格本数" not in ln)]
        breadth_details = "\n".join(inner_lines).strip()
    second_body = block_breadth + ("\n" + breadth_details if breadth_details else "")
    second_block = "```\n" + second_body.strip() + "\n```"

    header = first_block + "\n" + second_block + "\n" + header_core
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

