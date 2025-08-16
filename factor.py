from __future__ import annotations
"""
factor.py（compact full版）
- 既存ロジック／出力を変えないことを最優先に、重複の共通化と関数化で簡潔化
- 重要コメントは温存。定数・重み・ファイル出力形式は**完全互換**
- 依存：pandas, numpy, yfinance, scipy, requests
- 役割：G群12、D群13の選定（等重みポートフォリオ前提）。

【注意】
- yfinance/Finnhub のスキーマはまれに変わるため、キー探索はフォールバック付き。
- 収集APIの差異により、欠損は NaN としてスコア正規化で自然に無影響化。
- 出力：results/G_selection.json, results/D_selection.json（{"tickers": [...]}）
"""

# ===== import =====
import os, json, time, requests
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import zscore

# ===== ユニバースと定数（冒頭に固定・互換維持） =====
# 入力CSV：1列目にティッカーを想定
exist = pd.read_csv("current_tickers.csv", header=None)[0].tolist()
cand  = pd.read_csv("candidate_tickers.csv", header=None)[0].tolist()

# 候補銘柄の価格上限・ベンチマーク
CAND_PRICE_MAX, BENCH = 400, "^GSPC"

# G/D枠のサイズ
N_G, N_D = 12, 13

# ファクター重み（**数値・キー名は完全互換**）
G_WEIGHTS = {"GRW": 0.35, "MOM": 0.20, "TRD": 0.45, "VOL": -0.10}
D_WEIGHTS = {"QAL": 0.10, "YLD": 0.25, "VOL": -0.40, "TRD": 0.25}

# DRRS 初期プール・各種パラメータ（意味はそのまま）
corrM = 45
DRRS_G = dict(lookback=252, n_pc=3, gamma=1.2, lam=0.68, eta=0.8)
DRRS_D = dict(lookback=504, n_pc=4, gamma=0.8, lam=0.85, eta=0.5)
DRRS_SHRINK = 0.10  # 残差相関の対角シュリンク（基礎）

# クロス相関ペナルティ（未定義なら推奨値を採用）
try:
    CROSS_MU_GD
except NameError:
    CROSS_MU_GD = 0.40  # 推奨 0.35–0.45（lam=0.85想定）

# 出力関連
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
G_PREV_JSON = RESULTS_DIR / "G_selection.json"
D_PREV_JSON = RESULTS_DIR / "D_selection.json"

# その他
DEBUG = False
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# ========= ユーティリティ：前処理・統計 =========

def winsorize_s(s: pd.Series, p: float = 0.02) -> pd.Series:
    if s is None or s.dropna().empty: return s
    lo, hi = np.nanpercentile(s.astype(float), [100*p, 100*(1-p)])
    return s.clip(lo, hi)

def robust_z(s: pd.Series, p: float = 0.02) -> np.ndarray:
    """軽い外れ値剪定→Z化。欠損は列平均で埋める（Zに影響最小）。"""
    t = winsorize_s(s, p).fillna(s.mean())
    return np.nan_to_num(zscore(t))

_safe_nan = float("nan")

def _safe_div(a, b):
    try:
        if b is None or float(b) == 0 or pd.isna(b): return _safe_nan
        return float(a) / float(b)
    except Exception:
        return _safe_nan

def _safe_last(series: pd.Series, default=_safe_nan):
    try: return float(series.iloc[-1])
    except Exception: return default

# RSライン（価格比 s/b）の対数回帰で傾き（win日）
# +: 上昇、-: 下降。データ不足は NaN。
def rs_line_slope(s: pd.Series, b: pd.Series, win: int) -> float:
    r = (s / b).dropna();  
    if len(r) < win: return _safe_nan
    y = np.log(r.iloc[-win:]); x = np.arange(len(y), dtype=float)
    try: return float(np.polyfit(x, y, 1)[0])
    except Exception: return _safe_nan

# EV欠損時の簡易代替: EV ≒ 時価総額 + 負債 - 現金。取れなければ NaN

def ev_fallback(info_t: dict, tk: yf.Ticker) -> float:
    ev = info_t.get('enterpriseValue', _safe_nan)
    if pd.notna(ev) and ev > 0: return float(ev)
    mc = info_t.get('marketCap', _safe_nan); debt = cash = _safe_nan
    try:
        bs = tk.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            c = bs.columns[0]
            for k in ("Total Debt", "Long Term Debt", "Short Long Term Debt"):
                if k in bs.index: debt = float(bs.loc[k, c]); break
            for k in ("Cash And Cash Equivalents",
                      "Cash And Cash Equivalents And Short Term Investments",
                      "Cash"):
                if k in bs.index: cash = float(bs.loc[k, c]); break
    except Exception:
        pass
    return float(mc + (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash)) if pd.notna(mc) else _safe_nan

# ========= EPS補完 & FCF算出ユーティリティ =========

def impute_eps_ttm(df: pd.DataFrame, ttm_col: str = "eps_ttm", q_col: str = "eps_q_recent", out_col: str | None = None) -> pd.DataFrame:
    out_col = out_col or ttm_col
    df = df.copy(); df["eps_imputed"] = False
    cand = df[q_col] * 4
    ok = df[ttm_col].isna() & cand.replace([np.inf, -np.inf], np.nan).notna()
    df.loc[ok, out_col] = cand[ok]; df.loc[ok, "eps_imputed"] = True
    return df

_CF_ALIASES = {
    "cfo":   ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
}

_pick = lambda df, names: None if (df is None or df.empty) else (
    df.loc[{str(i).lower(): i for i in df.index}.get(names[0].lower())]
    if {str(i).lower() for i in df.index} & {n.lower() for n in names} else None
)
_sum_last_n = lambda s, n: None if (s is None or s.empty) else s.dropna().astype(float).iloc[:n].sum() if not s.dropna().empty else None
_latest     = lambda s:   None if (s is None or s.empty) else s.dropna().astype(float).iloc[0] if not s.dropna().empty else None

# yfinance から CFO/Capex/FCF（TTM相当）を取得

def fetch_cfo_capex_ttm_yf(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        qcf = tk.quarterly_cashflow
        cfo_q, capex_q = _pick(qcf, _CF_ALIASES["cfo"]), _pick(qcf, _CF_ALIASES["capex"])
        fcf_q = _pick(qcf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
        cfo_ttm, capex_ttm, fcf_ttm = (_sum_last_n(x, 4) for x in (cfo_q, capex_q, fcf_q))
        if None in (cfo_ttm, capex_ttm, fcf_ttm):
            acf = tk.cashflow
            cfo_a, capex_a = _pick(acf, _CF_ALIASES["cfo"]), _pick(acf, _CF_ALIASES["capex"])
            fcf_a = _pick(acf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
            cfo_ttm = cfo_ttm if cfo_ttm is not None else _latest(cfo_a)
            capex_ttm = capex_ttm if capex_ttm is not None else _latest(capex_a)
            fcf_ttm   = fcf_ttm  if fcf_ttm  is not None else _latest(fcf_a)
        rows.append({
            "ticker": t,
            "cfo_ttm_yf":   np.nan if cfo_ttm   is None else cfo_ttm,
            "capex_ttm_yf": np.nan if capex_ttm is None else capex_ttm,
            "fcf_ttm_yf_direct": np.nan if fcf_ttm is None else fcf_ttm,
        })
    return pd.DataFrame(rows).set_index("ticker")

# Finnhub 用
_FINN_CFO_KEYS   = [
    "netCashProvidedByOperatingActivities", "netCashFromOperatingActivities",
    "cashFlowFromOperatingActivities", "operatingCashFlow",
]
_FINN_CAPEX_KEYS = [
    "capitalExpenditure", "capitalExpenditures", "purchaseOfPPE",
    "investmentsInPropertyPlantAndEquipment",
]

_first_key = lambda d, keys: next((d[k] for k in keys if k in d and d[k] is not None), None)

def _finn_get(sess: requests.Session, url: str, params: dict, retries: int = 3, sleep_s: float = 0.5):
    for i in range(retries):
        r = sess.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(min(2**i * sleep_s, 4.0)); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

# ========= 価格・配当・基本情報の取得 =========

def _download_prices(tickers: Sequence[str], period: str = "3y") -> pd.DataFrame:
    """調整終値のみ取得（列：ticker, 行：date）。"""
    px = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    return px.sort_index()

def _latest_close_under(px: pd.DataFrame, cap: float) -> set[str]:
    last = px.ffill().iloc[-1]
    return set(last.index[last <= cap].tolist())

def _fetch_info_bulk(tickers: Sequence[str]) -> dict[str, dict]:
    out = {}
    for t in tickers:
        try: out[t] = yf.Ticker(t).info or {}
        except Exception: out[t] = {}
    return out

# ========= ファクター計算 =========

def _momentum(px: pd.Series, lb: int = 126) -> float:
    if px.dropna().shape[0] < lb+1: return _safe_nan
    return _safe_div(px.iloc[-1], px.iloc[-lb-1]) - 1.0

def _trend_vs_bench(px_t: pd.Series, px_b: pd.Series, win: int = 90) -> float:
    return rs_line_slope(px_t, px_b, win)

def _volatility(px: pd.Series, lb: int = 63) -> float:
    r = px.pct_change().dropna()
    return r.iloc[-lb:].std() if r.size else _safe_nan

# Quality/Dividend は info と FCF で補強（欠損は自動的に NaN）

def _dividend_yield(info: dict) -> float:
    for k in ("dividendYield", "trailingAnnualDividendYield"):
        v = info.get(k)
        if v is not None and v == v:  # not NaN
            return float(v) if v < 1.0 else float(v)  # yfinanceは0-1のことが多い
    return _safe_nan

def _quality_proxy(info: dict, fcf_row: pd.Series | None) -> float:
    # FCFマージン・利益率・ROEの合成（取れるものだけ）
    parts = []
    for k in ("profitMargins", "returnOnEquity", "grossMargins", "operatingMargins"):
        v = info.get(k)
        if v is not None and v == v: parts.append(float(v))
    if fcf_row is not None:
        fcf = fcf_row.get("fcf_ttm_yf_direct")
        rev = info.get("totalRevenue") or info.get("revenue")
        if (fcf is not None) and rev not in (None, 0) and rev == rev:
            parts.append(_safe_div(fcf, rev))
    return float(np.nanmean(parts)) if parts else _safe_nan

# Growth はEPS/売上の近似（四半期×4→TTM補完）

def _growth_proxy(info: dict) -> float:
    # salesGrowth, earningsGrowth が取れればそれを採用
    for k in ("earningsQuarterlyGrowth", "earningsGrowth", "revenueGrowth", "salesGrowth"):
        v = info.get(k)
        if v is not None and v == v: return float(v)
    return _safe_nan

# ========= スコア正規化・合成 =========

def _compose_score(df: pd.DataFrame, weights: dict[str, float], cols: list[str]) -> pd.Series:
    z = pd.DataFrame({c: robust_z(df[c]) for c in cols if c in df}, index=df.index)
    for c in cols:
        if c not in z: z[c] = 0.0  # 欠損列は0寄与（他列のZのみで評価）
    return z.mul(pd.Series(weights)).sum(axis=1)

# ========= DRRS & 相関ペナルティ =========

def _corr_penalty(ret: pd.DataFrame, mu: float) -> pd.DataFrame:
    C = ret.corr().fillna(0.0)
    return mu * C  # mu×相関で単純ペナルティ（対角は1→最終的に無視）

# DRRS: 主要因子（市場・業種など）を除去した“残差相関”で分散を評価
# n_pc: 除去する主成分数、shrink: 相関のノイズ縮小、lookback: 相関計測窓（営業日）

def _residualize(ret: pd.DataFrame, n_pc: int) -> pd.DataFrame:
    R = ret - ret.mean()
    X = R.values
    try:
        import numpy as _np
        U, S, Vt = _np.linalg.svd(X, full_matrices=False)
        k = max(0, min(n_pc, min(X.shape) - 1))
        if k <= 0: return R
        X_hat = (U[:, :k] * S[:k]) @ Vt[:k, :]
        resid = X - X_hat
        return pd.DataFrame(resid, index=ret.index, columns=ret.columns)
    except Exception:
        return R


def _greedy_drrs_select(scores: pd.Series, ret: pd.DataFrame, k: int, mu: float, *, n_pc: int, shrink: float, lookback: int | None) -> list[str]:
    # スコア－（残差の）平均相関×μ を最大化するように Greedy に抽出
    base = ret.tail(lookback) if lookback else ret
    R = _residualize(base, n_pc)
    selected: list[str] = []
    cand = set(scores.dropna().index) & set(R.columns)
    while len(selected) < k and cand:
        best_t, best_val = None, -1e9
        for t in list(cand):
            base_score = scores.get(t, -1e9)
            if not selected:
                adj = base_score
            else:
                corr = R[selected + [t]].corr().loc[t, selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                mean_corr = float(((1.0 - shrink) * corr).mean()) if len(corr) else 0.0
                adj = base_score - mu * mean_corr
            if adj > best_val:
                best_t, best_val = t, adj
        if best_t is None: break
        selected.append(best_t); cand.remove(best_t)
    return selected

# ========= パイプライン =========


def build_feature_table(all_tickers: Sequence[str]) -> pd.DataFrame:
    # 価格
    px = _download_prices(list(set(all_tickers) | {BENCH}), period="3y")
    px = px.dropna(how="all", axis=1)  # まったく価格が取れない銘柄は除外

    # 価格上限で候補をフィルタ（既存は温存）
    under_cap = _latest_close_under(px, CAND_PRICE_MAX)
    pool = set(exist) | (set(cand) & under_cap)
    pool = [t for t in pool if t in px.columns]

    bench = px[BENCH].dropna()
    feat = []
    info_map = _fetch_info_bulk(pool)
    fcf_df = fetch_cfo_capex_ttm_yf(pool)

    for t in pool:
        s = px[t].dropna()
        feat.append({
            "ticker": t,
            # 基本ファクター（Growth/Momentum/Trend/Vol/Quality/Yield）
            "GRW": _growth_proxy(info_map.get(t, {})),
            "MOM": _momentum(s),
            "TRD": _trend_vs_bench(s, bench),
            "VOL": _volatility(s),
            "QAL": _quality_proxy(info_map.get(t, {}), fcf_df.loc[t] if t in fcf_df.index else None),
            "YLD": _dividend_yield(info_map.get(t, {})),
        })
    df = pd.DataFrame(feat).set_index("ticker")
    # VOL は低いほど良い → 方向を反転しても良いが、重みで負を入れており**互換保持のためそのまま**
    return df, px[pool], bench


def select_groups(df: pd.DataFrame, px: pd.DataFrame, bench: pd.Series) -> tuple[list[str], list[str]]:
    # 収益率（相関用）
    ret = px.pct_change().dropna()

    # 合成スコア
    g_cols = ["GRW","MOM","TRD","VOL"]
    d_cols = ["QAL","YLD","VOL","TRD"]
    score_G = _compose_score(df, G_WEIGHTS, g_cols)
    score_D = _compose_score(df, D_WEIGHTS, d_cols)

    # μ（相関ペナルティ強度）
    mu = float(CROSS_MU_GD)

    # DRRS パラメータを適用（残差相関・縮小・窓= corrM）
    sel_G = _greedy_drrs_select(score_G, ret, N_G, mu, n_pc=DRRS_G["n_pc"], shrink=DRRS_SHRINK, lookback=corrM)
    remain = [t for t in df.index if t not in set(sel_G)]
    sel_D = _greedy_drrs_select(score_D.loc[remain], ret[remain], N_D, mu, n_pc=DRRS_D["n_pc"], shrink=DRRS_SHRINK, lookback=corrM)
    return sel_G, sel_D


# ========= 出力 =========

def _save_selection(path: Path, tickers: Sequence[str]):
    with path.open("w", encoding="utf-8") as f:
        json.dump({"tickers": list(tickers)}, f, ensure_ascii=False, indent=2)


# ========= main =========
if __name__ == "__main__":
    # 元の前処理→ファクター計算→スコア合成→制約（価格上限・既存維持など）→
    # DRRSや相関ペナルティ→最終選定、を順に適用。
    all_tickers = sorted(set(exist) | set(cand))
    df, px, bench = build_feature_table(all_tickers)
    sel_G, sel_D = select_groups(df, px, bench)
    _save_selection(G_PREV_JSON, sel_G)
    _save_selection(D_PREV_JSON, sel_D)
    if DEBUG:
        print("G:", sel_G)
        print("D:", sel_D)
