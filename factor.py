# === NOTE: 機能・入出力・ログ文言・例外挙動は不変。安全な短縮（import統合/複数代入/内包表記/メソッドチェーン/一行化/空行圧縮など）のみ適用 ===
import yfinance as yf, pandas as pd, numpy as np, os, requests, time, json
from scipy.stats import zscore

# ===== ユニバースと定数（冒頭に固定） =====
exist, cand = [pd.read_csv(f, header=None)[0].tolist() for f in ("current_tickers.csv","candidate_tickers.csv")]
CAND_PRICE_MAX, bench = 400, '^GSPC'  # 価格上限・ベンチマーク
N_G, N_D = 12, 13  # G/D枠サイズ
g_weights = {'GRW':0.35,'MOM':0.20,'TRD':0.45,'VOL':-0.10}
D_weights = {'QAL':0.1,'YLD':0.25,'VOL':-0.4,'TRD':0.25}

# DRRS 初期プール・各種パラメータ
corrM = 45
DRRS_G, DRRS_D = dict(lookback=252,n_pc=3,gamma=1.2,lam=0.68,eta=0.8), dict(lookback=504,n_pc=4,gamma=0.8,lam=0.85,eta=0.5)
DRRS_SHRINK = 0.10  # 残差相関の対角シュリンク（基礎）

# クロス相関ペナルティ（未定義なら設定）
try: CROSS_MU_GD
except NameError: CROSS_MU_GD = 0.40  # 推奨 0.35–0.45（lam=0.85想定）

# 出力関連
RESULTS_DIR, G_PREV_JSON, D_PREV_JSON = "results", os.path.join("results","G_selection.json"), os.path.join("results","D_selection.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# その他
debug_mode, FINNHUB_API_KEY = False, os.environ.get("FINNHUB_API_KEY")

def winsorize_s(s: pd.Series, p=0.02):
    if s is None or s.dropna().empty: return s
    lo, hi = np.nanpercentile(s.astype(float), [100*p, 100*(1-p)]); return s.clip(lo, hi)

def robust_z(s: pd.Series, p=0.02):
    """軽い外れ値剪定→Z化。欠損は列平均で埋める（Zに影響最小）。"""
    s2 = winsorize_s(s, p); return np.nan_to_num(zscore(s2.fillna(s2.mean())))

def _safe_div(a, b):
    try:
        if b is None or float(b)==0 or pd.isna(b): return np.nan
        return float(a)/float(b)
    except Exception: return np.nan

def _safe_last(series: pd.Series, default=np.nan):
    try: return float(series.iloc[-1])
    except Exception: return default

def rs_line_slope(s: pd.Series, b: pd.Series, win: int) -> float:
    """RSライン（価格比 s/b）の対数に回帰して傾きを返す（win日）。+:上昇、-:下降。データ不足は NaN。"""
    r = (s/b).dropna()
    if len(r)<win: return np.nan
    y, x = np.log(r.iloc[-win:]), np.arange(win, dtype=float)
    try: return float(np.polyfit(x, y, 1)[0])
    except Exception: return np.nan

def ev_fallback(info_t: dict, tk: yf.Ticker) -> float:
    """EV欠損時の簡易代替: EV ≒ 時価総額 + 負債 - 現金。取れなければ NaN"""
    ev = info_t.get('enterpriseValue', np.nan)
    if pd.notna(ev) and ev>0: return float(ev)
    mc, debt, cash = info_t.get('marketCap', np.nan), np.nan, np.nan
    try:
        bs = tk.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            c = bs.columns[0]
            for k in ("Total Debt","Long Term Debt","Short Long Term Debt"):
                if k in bs.index: debt = float(bs.loc[k,c]); break
            for k in ("Cash And Cash Equivalents","Cash And Cash Equivalents And Short Term Investments","Cash"):
                if k in bs.index: cash = float(bs.loc[k,c]); break
    except Exception: pass
    if pd.notna(mc): return float(mc + (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash))
    return np.nan

# ========= EPS補完 & FCF算出ユーティリティ =========
def impute_eps_ttm(df: pd.DataFrame, ttm_col: str="eps_ttm", q_col: str="eps_q_recent", out_col: str|None=None) -> pd.DataFrame:
    out_col = out_col or ttm_col; df = df.copy(); df["eps_imputed"] = False
    cand = df[q_col]*4; ok = df[ttm_col].isna() & cand.replace([np.inf,-np.inf], np.nan).notna()
    df.loc[ok, out_col], df.loc[ok,"eps_imputed"] = cand[ok], True; return df

_CF_ALIASES = {"cfo":["Operating Cash Flow","Total Cash From Operating Activities"], "capex":["Capital Expenditure","Capital Expenditures"]}

def _pick_row(df: pd.DataFrame, names: list[str]) -> pd.Series|None:
    if df is None or df.empty: return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for name in names:
        key = name.lower()
        if key in idx_lower: return df.loc[idx_lower[key]]
    return None

def _sum_last_n(s: pd.Series|None, n: int) -> float|None:
    if s is None or s.empty: return None
    vals = s.dropna().astype(float); return None if vals.empty else vals.iloc[:n].sum()

def _latest(s: pd.Series|None) -> float|None:
    if s is None or s.empty: return None
    vals = s.dropna().astype(float); return vals.iloc[0] if not vals.empty else None

def fetch_cfo_capex_ttm_yf(tickers: list[str]) -> pd.DataFrame:
    rows=[]
    for t in tickers:
        tk = yf.Ticker(t); qcf = tk.quarterly_cashflow
        cfo_q, capex_q = _pick_row(qcf,_CF_ALIASES["cfo"]), _pick_row(qcf,_CF_ALIASES["capex"])
        fcf_q = _pick_row(qcf, ["Free Cash Flow","FreeCashFlow","Free cash flow"])
        cfo_ttm, capex_ttm, fcf_ttm_direct = _sum_last_n(cfo_q,4), _sum_last_n(capex_q,4), _sum_last_n(fcf_q,4)
        if cfo_ttm is None or capex_ttm is None or fcf_ttm_direct is None:
            acf = tk.cashflow
            cfo_a, capex_a, fcf_a = _pick_row(acf,_CF_ALIASES["cfo"]), _pick_row(acf,_CF_ALIASES["capex"]), _pick_row(acf,["Free Cash Flow","FreeCashFlow","Free cash flow"])
            if cfo_ttm is None: cfo_ttm = _latest(cfo_a)
            if capex_ttm is None: capex_ttm = _latest(capex_a)
            if fcf_ttm_direct is None: fcf_ttm_direct = _latest(fcf_a)
        rows.append({"ticker":t,"cfo_ttm_yf":cfo_ttm if cfo_ttm is not None else np.nan,"capex_ttm_yf":capex_ttm if capex_ttm is not None else np.nan,"fcf_ttm_yf_direct":fcf_ttm_direct if fcf_ttm_direct is not None else np.nan})
    return pd.DataFrame(rows).set_index("ticker")

_FINN_CFO_KEYS = ["netCashProvidedByOperatingActivities","netCashFromOperatingActivities","cashFlowFromOperatingActivities","operatingCashFlow"]
_FINN_CAPEX_KEYS = ["capitalExpenditure","capitalExpenditures","purchaseOfPPE","investmentsInPropertyPlantAndEquipment"]

def _first_key(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return None

def _finn_get(session: requests.Session, url: str, params: dict, retries: int=3, sleep_s: float=0.5):
    for i in range(retries):
        r = session.get(url, params=params, timeout=15)
        if r.status_code==429: time.sleep(min(2**i*sleep_s,4.0)); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

def fetch_cfo_capex_ttm_finnhub(tickers: list[str], api_key: str|None=None) -> pd.DataFrame:
    api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not api_key: raise ValueError("Finnhub API key not provided. Set FINNHUB_API_KEY or pass api_key=")
    base, s, rows = "https://finnhub.io/api/v1", requests.Session(), []
    for sym in tickers:
        cfo_ttm = capex_ttm = None
        try:
            j = _finn_get(s, f"{base}/stock/cash-flow", {"symbol":sym,"frequency":"quarterly","limit":8,"token":api_key})
            arr = j.get("cashFlow") or []; cfo_vals, capex_vals = [], []
            for item in arr[:4]:
                cfo_vals.append(_first_key(item,_FINN_CFO_KEYS)); capex_vals.append(_first_key(item,_FINN_CAPEX_KEYS))
            if any(v is not None for v in cfo_vals): cfo_ttm = float(np.nansum([np.nan if v is None else float(v) for v in cfo_vals]))
            if any(v is not None for v in capex_vals): capex_ttm = float(np.nansum([np.nan if v is None else float(v) for v in capex_vals]))
        except Exception: pass
        if cfo_ttm is None or capex_ttm is None:
            try:
                j = _finn_get(s, f"{base}/stock/cash-flow", {"symbol":sym,"frequency":"annual","limit":1,"token":api_key})
                arr = j.get("cashFlow") or []
                if arr:
                    item0 = arr[0]
                    if cfo_ttm is None:
                        v = _first_key(item0,_FINN_CFO_KEYS)
                        if v is not None: cfo_ttm = float(v)
                    if capex_ttm is None:
                        v = _first_key(item0,_FINN_CAPEX_KEYS)
                        if v is not None: capex_ttm = float(v)
            except Exception: pass
        rows.append({"ticker":sym,"cfo_ttm_fh":np.nan if cfo_ttm is None else cfo_ttm,"capex_ttm_fh":np.nan if capex_ttm is None else capex_ttm})
    return pd.DataFrame(rows).set_index("ticker")

def compute_fcf_with_fallback(tickers: list[str], finnhub_api_key: str|None=None) -> pd.DataFrame:
    yf_df, fh_df = fetch_cfo_capex_ttm_yf(tickers), fetch_cfo_capex_ttm_finnhub(tickers, api_key=finnhub_api_key)
    df = yf_df.join(fh_df, how="outer")
    df["cfo_ttm"]  = df["cfo_ttm_yf"].where(df["cfo_ttm_yf"].notna(), df["cfo_ttm_fh"])
    df["capex_ttm"] = df["capex_ttm_yf"].where(df["capex_ttm_yf"].notna(), df["capex_ttm_fh"])
    cfo, capex = pd.to_numeric(df["cfo_ttm"], errors="coerce"), pd.to_numeric(df["capex_ttm"], errors="coerce").abs()
    fcf_calc = cfo - capex
    fcf_direct = pd.to_numeric(df.get("fcf_ttm_yf_direct"), errors="coerce")
    df["fcf_ttm"] = fcf_calc.where(fcf_calc.notna(), fcf_direct)
    df["cfo_source"]  = np.where(df["cfo_ttm_yf"].notna(),"yfinance",np.where(df["cfo_ttm_fh"].notna(),"finnhub",""))
    df["capex_source"] = np.where(df["capex_ttm_yf"].notna(),"yfinance",np.where(df["capex_ttm_fh"].notna(),"finnhub",""))
    df["fcf_imputed"] = df[["cfo_ttm","capex_ttm"]].isna().any(axis=1) & df["fcf_ttm"].notna()
    cols = ["cfo_ttm_yf","capex_ttm_yf","cfo_ttm_fh","capex_ttm_fh","cfo_ttm","capex_ttm","fcf_ttm","fcf_ttm_yf_direct","cfo_source","capex_source","fcf_imputed"]
    return df[cols].sort_index()

def prepare_data():
    """Fetch price and fundamental data for all tickers."""
    global cand_info, cand_prices, cand, tickers, data, px, spx, tickers_bulk, info, eps_df, fcf_df, returns
    cand_info = yf.Tickers(" ".join(cand)); cand_prices = {}
    for t in cand:
        try: cand_prices[t] = cand_info.tickers[t].fast_info.get("lastPrice", np.inf)
        except Exception as e: print(f"{t}: price fetch failed ({e})"); cand_prices[t] = np.inf
    cand = [t for t,p in cand_prices.items() if p<=CAND_PRICE_MAX]
    tickers = sorted(set(exist + cand))
    data = yf.download(tickers + [bench], period="600d", auto_adjust=True, progress=False)
    px, spx = data["Close"], data["Close"][bench]
    tickers_bulk, info = yf.Tickers(" ".join(tickers)), {}
    for t in tickers:
        try: info[t] = tickers_bulk.tickers[t].info
        except Exception as e: print(f"{t}: info fetch failed ({e})"); info[t] = {}
    # EPSとFCFの補完データ
    eps_rows=[]
    for t in tickers:
        info_t, eps_ttm, eps_q = info[t], info[t].get("trailingEps", np.nan), np.nan
        try:
            qearn, so = tickers_bulk.tickers[t].quarterly_earnings, info_t.get("sharesOutstanding")
            if so and qearn is not None and not qearn.empty and "Earnings" in qearn.columns:
                eps_ttm_q = qearn["Earnings"].head(4).sum()/so
                if pd.notna(eps_ttm_q) and (pd.isna(eps_ttm) or (abs(eps_ttm)>0 and abs(eps_ttm/eps_ttm_q)>3)): eps_ttm = eps_ttm_q
                eps_q = qearn["Earnings"].iloc[-1]/so
        except Exception: pass
        eps_rows.append({"ticker":t,"eps_ttm":eps_ttm,"eps_q_recent":eps_q})
    eps_df = impute_eps_ttm(pd.DataFrame(eps_rows).set_index("ticker"))
    fcf_df = compute_fcf_with_fallback(tickers, finnhub_api_key=FINNHUB_API_KEY)
    returns = px[tickers].pct_change()  # ここでは dropna しない

# ===== DRRS helpers (決定論RRQR・残差相関スワップ) =====
def _z_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32); m = np.nanmean(X, axis=0, keepdims=True); s = np.nanstd(X, axis=0, keepdims=True)+1e-9
    return (np.nan_to_num(X)-m)/s

def residual_corr(R: np.ndarray, n_pc: int=3, shrink: float=0.1) -> np.ndarray:
    Z = _z_np(R); U,S,_ = np.linalg.svd(Z, full_matrices=False); F = U[:,:n_pc]*S[:n_pc]; B = np.linalg.lstsq(F, Z, rcond=None)[0]
    E = Z - F@B; C = np.corrcoef(E, rowvar=False)
    off = C - np.diag(np.diag(C)); iu = np.triu_indices_from(off,1); avg_abs = np.nanmean(np.abs(off[iu])) if iu[0].size else 0.0
    shrink_eff = float(np.clip(shrink + 0.5*avg_abs, 0.1, 0.6)); N = C.shape[0]
    return (1.0 - shrink_eff)*C + shrink_eff*np.eye(N, dtype=C.dtype)

def rrqr_like_det(R: np.ndarray, score: np.ndarray, k: int, gamma: float=1.0):
    """スコア重み付き RRQR 風の決定論初期選定（乱数なし・タイブレーク固定）。"""
    Z, w = _z_np(R), (score-score.min())/(np.ptp(score)+1e-12); X = Z*(1.0 + gamma*w)
    N, k = X.shape[1], max(0, min(k, X.shape[1]))
    if k==0: return []
    S, selected, Rres = [], np.zeros(N, dtype=bool), X.copy()
    for _ in range(k):
        norms = (Rres*Rres).sum(axis=0); cand = np.where(~selected)[0]
        j = sorted(cand, key=lambda c:(-norms[c], -w[c], c))[0]
        S.append(j); selected[j]=True; u = X[:,j:j+1]; u/=(np.linalg.norm(u)+1e-12); Rres = Rres - u @ (u.T @ Rres)
    return sorted(S)

def _obj(corrM: np.ndarray, score: np.ndarray, idx, lam: float) -> float:
    idx = list(idx); P = corrM[np.ix_(idx, idx)]; s = (score-score.mean())/(score.std()+1e-9)
    return float(s[idx].sum() - lam*((P.sum()-np.trace(P))/2.0))

def swap_local_det(corrM: np.ndarray, score: np.ndarray, idx, lam: float=0.6, max_pass: int=15):
    """1入替のbest-improvementのみ。固定順序・微小差バリア付き（決定論）。"""
    S, best, improved, passes = sorted(idx), _obj(corrM, score, idx, lam), True, 0
    while improved and passes<max_pass:
        improved, passes = False, passes+1
        for i,out in enumerate(list(S)):
            for inn in range(len(score)):
                if inn in S: continue
                cand = sorted(S[:i]+[inn]+S[i+1:]); v = _obj(corrM, score, cand, lam)
                if v>best+1e-10: S, best, improved = cand, v, True; break
            if improved: break
    return S, best

if '_obj_with_cross' not in globals():
    def _obj_with_cross(C_within: np.ndarray, C_cross: np.ndarray|None, score: np.ndarray, idx, lam: float, mu: float) -> float:
        """合計スコア − λ×(D内ペア相関総和) − μ×(G↔Dクロス相関総和)"""
        idx = list(idx); P = C_within[np.ix_(idx, idx)]; s = (score-score.mean())/(score.std()+1e-9)
        within = (P.sum()-np.trace(P))/2.0; cross = 0.0
        if C_cross is not None and C_cross.size>0: cross = C_cross[idx,:].sum()
        return float(s[idx].sum() - lam*within - mu*cross)

if 'swap_local_det_cross' not in globals():
    def swap_local_det_cross(C_within: np.ndarray, C_cross: np.ndarray|None, score: np.ndarray, idx, lam: float=0.6, mu: float=0.3, max_pass: int=15):
        """クロス相関ペナルティ入りのbest-improvement 1入替（決定論）。"""
        S, best, improved, passes, N = sorted(idx), _obj_with_cross(C_within,C_cross,score,idx,lam,mu), True, 0, len(score)
        while improved and passes<max_pass:
            improved, passes = False, passes+1
            for i,out in enumerate(list(S)):
                for inn in range(N):
                    if inn in S: continue
                    cand = sorted(S[:i]+[inn]+S[i+1:]); v = _obj_with_cross(C_within,C_cross,score,cand,lam,mu)
                    if v>best+1e-10: S, best, improved = cand, v, True; break
                if improved: break
        return S, best

def avg_corr(C: np.ndarray, idx) -> float:
    k = len(idx); P = C[np.ix_(idx, idx)]
    return float((P.sum()-np.trace(P))/(k*(k-1)+1e-12))

def _load_prev(path: str):
    try: return json.load(open(path)).get("tickers")
    except Exception: return None

def _save_sel(path: str, tickers: list[str], avg_r: float, sum_score: float, objective: float):
    with open(path,"w") as f:
        json.dump({"tickers":tickers,"avg_res_corr":round(avg_r,6),"sum_score":round(sum_score,6),"objective":round(objective,6)}, f, indent=2)

def select_bucket_drrs(returns_df: pd.DataFrame, score_ser: pd.Series, pool_tickers: list[str], k: int, *, n_pc: int, gamma: float, lam: float, eta: float, lookback: int, prev_tickers: list[str]|None, shrink: float=0.10, g_fixed_tickers: list[str]|None=None, mu: float=0.0):
    """
    returns_df: T×N の日次リターン（pct_change済）。
    pool_tickers: 候補集合（この中から k を選ぶ）。
    g_fixed_tickers: Gを固定してDを最適化する場合に指定。None/[]なら従来動作。
    mu: G↔Dクロス相関のペナルティ係数（総和に掛ける）。
    """
    g_fixed = [t for t in (g_fixed_tickers or []) if t in returns_df.columns]
    union = [t for t in pool_tickers if t in returns_df.columns]
    for t in g_fixed:
        if t not in union: union.append(t)
    Rdf_all = returns_df[union]; Rdf_all = Rdf_all.iloc[-lookback:] if len(Rdf_all)>=lookback else Rdf_all; Rdf_all = Rdf_all.dropna()
    pool_eff, g_eff = [t for t in pool_tickers if t in Rdf_all.columns], [t for t in g_fixed if t in Rdf_all.columns]
    if len(pool_eff)==0: return dict(idx=[], tickers=[], avg_res_corr=np.nan, sum_score=0.0, objective=-np.inf)
    score = score_ser.reindex(pool_eff).to_numpy(dtype=np.float32)
    C_all = residual_corr(Rdf_all.to_numpy(), n_pc=n_pc, shrink=shrink)
    col_pos = {c:i for i,c in enumerate(Rdf_all.columns)}; pool_pos = [col_pos[t] for t in pool_eff]
    C_within, C_cross = C_all[np.ix_(pool_pos,pool_pos)], None
    if len(g_eff)>0 and mu>0.0:
        g_pos = [col_pos[t] for t in g_eff]; C_cross = C_all[np.ix_(pool_pos,g_pos)]
    R_pool = Rdf_all[pool_eff].to_numpy(); S0 = rrqr_like_det(R_pool, score, k, gamma=gamma)
    S, Jn = (swap_local_det_cross(C_within, C_cross, score, S0, lam=lam, mu=mu, max_pass=15) if C_cross is not None else swap_local_det(C_within, score, S0, lam=lam, max_pass=15))
    if prev_tickers:
        prev_idx = [pool_eff.index(t) for t in prev_tickers if t in pool_eff]
        if len(prev_idx)==min(k,len(pool_eff)):
            Jp = (_obj_with_cross(C_within,C_cross,score,prev_idx,lam,mu) if C_cross is not None else _obj(C_within,score,prev_idx,lam))
            if Jn < Jp + eta: S, Jn = sorted(prev_idx), Jp
    selected_tickers = [pool_eff[i] for i in S]
    return dict(idx=S, tickers=selected_tickers, avg_res_corr=avg_corr(C_within,S), sum_score=float(score[S].sum()), objective=float(Jn))

# ----- ファクター計算関数 -----
def trend(s: pd.Series):
    if len(s)<200: return np.nan
    sma50, sma150, sma200 = s.rolling(50).mean().iloc[-1], s.rolling(150).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
    prev200, p = s.rolling(200).mean().iloc[-21], s.iloc[-1]
    lo_52 = s[-252:].min() if len(s)>=252 else s.min(); hi_52 = s[-252:].max() if len(s)>=252 else s.max()
    rng = (hi_52 - lo_52) if hi_52>lo_52 else np.nan
    clip = lambda x,lo,hi: (np.nan if pd.isna(x) else max(lo,min(hi,x)))
    a = clip(p/(s.rolling(50).mean().iloc[-1]) - 1, -0.5, 0.5)
    b = clip(sma50/sma150 - 1, -0.5, 0.5)
    c = clip(sma150/sma200 - 1, -0.5, 0.5)
    d = clip(sma200/prev200 - 1, -0.2, 0.2)
    e = clip((p - lo_52) / (rng if rng and rng>0 else np.nan) - 0.5, -0.5, 0.5)
    parts = [0.0 if pd.isna(x) else x for x in (a,b,c,d,e)]
    return 0.30*parts[0] + 0.20*parts[1] + 0.15*parts[2] + 0.15*parts[3] + 0.20*parts[4]

def rs(s, b):
    n, nb = len(s), len(b)
    if n<60 or nb<60: return np.nan
    L12 = 252 if n>=252 and nb>=252 else min(n,nb)-1; L1 = 22 if n>=22 and nb>=22 else max(5, min(n,nb)//3)
    r12, r1, br12, br1 = s.iloc[-1]/s.iloc[-L12]-1, s.iloc[-1]/s.iloc[-L1]-1, b.iloc[-1]/b.iloc[-L12]-1, b.iloc[-1]/b.iloc[-L1]-1
    return (r12 - br12)*0.7 + (r1 - br1)*0.3

def tr_str(s):
    """終値が50日移動平均からどれだけ乖離しているかで短期トレンドの強さを測定。"""
    if len(s)<50: return np.nan
    return s.iloc[-1]/s.rolling(50).mean().iloc[-1] - 1

def dividend_status(ticker: str) -> str:
    """配当状況簡易判定: has/none_confident/maybe_missing/unknown"""
    t = yf.Ticker(ticker)
    try:
        if not t.dividends.empty: return "has"
    except Exception: return "unknown"
    try:
        a = t.actions
        if a is not None and not a.empty and "Stock Splits" in a.columns and a["Stock Splits"].abs().sum()>0: return "none_confident"
    except Exception: pass
    try:
        fi = t.fast_info
        if any(getattr(fi,k,None) for k in ("last_dividend_date","dividend_rate","dividend_yield")): return "maybe_missing"
    except Exception: pass
    return "unknown"

def div_streak(t):
    """企業が何年連続で配当を増やしているか。"""
    try:
        divs = yf.Ticker(t).dividends.dropna(); ann = divs.groupby(divs.index.year).sum(); ann = ann[ann.index<pd.Timestamp.today().year]
        years, streak = sorted(ann.index), 0
        for i in range(len(years)-1,0,-1):
            if ann[years[i]] > ann[years[i-1]]: streak += 1
            else: break
        return streak
    except Exception: return 0

def fetch_finnhub_metrics(symbol):
    """finnhub API から不足データを取得"""
    if not FINNHUB_API_KEY: return {}
    url, params = "https://finnhub.io/api/v1/stock/metric", {"symbol":symbol,"metric":"all","token":FINNHUB_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10); r.raise_for_status(); m = r.json().get("metric",{})
        return {'EPS':m.get('epsGrowthTTMYoy'),'REV':m.get('revenueGrowthTTMYoy'),'ROE':m.get('roeTTM'),'BETA':m.get('beta'),'DIV':m.get('dividendYieldIndicatedAnnual'),'FCF':(m.get('freeCashFlowTTM')/m.get('enterpriseValue')) if m.get('freeCashFlowTTM') and m.get('enterpriseValue') else None}
    except Exception: return {}

def calc_beta(series: pd.Series, market: pd.Series, lookback=252):
    r, m = series.pct_change().dropna(), market.pct_change().dropna()
    n = min(len(r), len(m), lookback)
    if n<60: return np.nan
    r, m = r.iloc[-n:], m.iloc[-n:]; cov, var = np.cov(r, m)[0,1], np.var(m)
    return np.nan if var==0 else cov/var

if 'aggregate_scores' not in globals():
    def aggregate_scores():
        """特徴量→Z化→合成スコア作成。相関・選定は触らない。"""
        global df, missing_logs, df_z, g_score, d_score_all
        df, missing_logs = pd.DataFrame(index=tickers), []
        # ---- 特徴量生成 ----
        for t in tickers:
            d, s = info[t], px[t]; ev = ev_fallback(d, tickers_bulk.tickers[t])
            df.loc[t,'TR'], df.loc[t,'EPS'], df.loc[t,'REV'], df.loc[t,'ROE'] = trend(s), eps_df.loc[t,'eps_ttm'], d.get('revenueGrowth',np.nan), d.get('returnOnEquity',np.nan)
            df.loc[t,'BETA'] = calc_beta(s, spx, lookback=252)
            div = d.get('dividendYield') if d.get('dividendYield') is not None else d.get('trailingAnnualDividendYield')
            if div is None or pd.isna(div):
                try:
                    divs = yf.Ticker(t).dividends
                    if divs is not None and not divs.empty:
                        last_close = s.iloc[-1]; div_1y = divs[divs.index >= (divs.index.max() - pd.Timedelta(days=365))].sum()
                        if last_close and last_close>0: div = float(div_1y/last_close)
                except Exception: pass
            df.loc[t,'DIV'] = 0.0 if (div is None or pd.isna(div)) else float(div)
            fcf_val = fcf_df.loc[t,'fcf_ttm'] if t in fcf_df.index else np.nan
            df.loc[t,'FCF'] = (fcf_val/ev) if (pd.notna(fcf_val) and pd.notna(ev) and ev>0) else np.nan
            df.loc[t,'RS'], df.loc[t,'TR_str'] = rs(s, spx), tr_str(s)
            # --- リスク系（日足ベース） ---
            r, rm, n = s.pct_change().dropna(), spx.pct_change().dropna(), int(min(len(s.pct_change().dropna()), len(spx.pct_change().dropna())))
            DOWNSIDE_DEV = np.nan
            if n>=60:
                r6 = r.iloc[-min(len(r),126):]; neg = r6[r6<0]
                if len(neg)>=10: DOWNSIDE_DEV = float(neg.std(ddof=0)*np.sqrt(252))
            df.loc[t,'DOWNSIDE_DEV'] = DOWNSIDE_DEV
            MDD_1Y = np.nan
            try:
                w = s.iloc[-min(len(s),252):].dropna()
                if len(w)>=30: roll_max = w.cummax(); MDD_1Y = float((w/roll_max - 1.0).min())
            except Exception: pass
            df.loc[t,'MDD_1Y'] = MDD_1Y
            RESID_VOL = np.nan
            if n>=120:
                rr, rrm = r.iloc[-n:].align(rm.iloc[-n:], join='inner')
                if len(rr)==len(rrm) and len(rr)>=120 and rrm.var()>0:
                    beta = float(np.cov(rr, rrm)[0,1]/np.var(rrm)); resid = rr - beta*rrm
                    RESID_VOL = float(resid.std(ddof=0)*np.sqrt(252))
            df.loc[t,'RESID_VOL'] = RESID_VOL
            DOWN_OUTPERF = np.nan
            if n>=60:
                m, x = rm.iloc[-n:], r.iloc[-n:]; mask = m<0
                if mask.sum()>=10:
                    mr, sr = float(m[mask].mean()), float(x[mask].mean())
                    DOWN_OUTPERF = (sr - mr)/abs(mr) if mr!=0 else np.nan
            df.loc[t,'DOWN_OUTPERF'] = DOWN_OUTPERF
            sma200 = s.rolling(200).mean(); df.loc[t,'EXT_200'] = np.nan
            if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1]!=0: df.loc[t,'EXT_200'] = abs(float(s.iloc[-1]/sma200.iloc[-1]-1.0))
            # --- 配当の"質" ---
            DIV_TTM_PS=DIV_VAR5=DIV_YOY=DIV_FCF_COVER=np.nan
            try:
                divs = yf.Ticker(t).dividends.dropna()
                if not divs.empty:
                    last_close = s.iloc[-1]; div_1y = float(divs[divs.index >= (divs.index.max()-pd.Timedelta(days=365))].sum())
                    DIV_TTM_PS = div_1y if div_1y>0 else np.nan
                    ann = divs.groupby(divs.index.year).sum()
                    if len(ann)>=2 and ann.iloc[-2]!=0: DIV_YOY = float(ann.iloc[-1]/ann.iloc[-2]-1.0)
                    tail = ann.iloc[-5:] if len(ann)>=5 else ann
                    if len(tail)>=3 and tail.mean()!=0: DIV_VAR5 = float(tail.std(ddof=1)/abs(tail.mean()))
                so, fcf_ttm = d.get('sharesOutstanding',None), fcf_df.loc[t,'fcf_ttm'] if t in fcf_df.index else np.nan
                if so and pd.notna(DIV_TTM_PS) and pd.notna(fcf_ttm) and fcf_ttm!=0: DIV_FCF_COVER = float((fcf_ttm)/(DIV_TTM_PS*float(so)))
            except Exception: pass
            df.loc[t,'DIV_TTM_PS'], df.loc[t,'DIV_VAR5'], df.loc[t,'DIV_YOY'], df.loc[t,'DIV_FCF_COVER'] = DIV_TTM_PS, DIV_VAR5, DIV_YOY, DIV_FCF_COVER
            # --- 財務健全性・収益安定性 ---
            df.loc[t,'DEBT2EQ'], df.loc[t,'CURR_RATIO'] = d.get('debtToEquity',np.nan), d.get('currentRatio',np.nan)
            EPS_VAR_8Q = np.nan
            try:
                qe, so = tickers_bulk.tickers[t].quarterly_earnings, d.get('sharesOutstanding',None)
                if qe is not None and not qe.empty and so:
                    eps_q = (qe['Earnings'].dropna().astype(float)/float(so)).replace([np.inf,-np.inf],np.nan)
                    if len(eps_q)>=4: EPS_VAR_8Q = float(eps_q.iloc[-min(8,len(eps_q)):].std(ddof=1))
            except Exception: pass
            df.loc[t,'EPS_VAR_8Q'] = EPS_VAR_8Q
            # --- サイズ＆流動性 ---
            df.loc[t,'MARKET_CAP'] = d.get('marketCap',np.nan); adv60 = np.nan
            try:
                vol_series = data['Volume'][t].dropna()
                if len(vol_series)>=5 and len(s)==len(vol_series):
                    dv = (vol_series*s).rolling(60).mean(); adv60 = float(dv.iloc[-1])
            except Exception: pass
            df.loc[t,'ADV60_USD'] = adv60
            REV_Q_YOY=EPS_Q_YOY=REV_YOY_ACC=REV_YOY_VAR=np.nan
            try:
                qe, so = tickers_bulk.tickers[t].quarterly_earnings, d.get('sharesOutstanding',None)
                if qe is not None and not qe.empty:
                    if 'Revenue' in qe.columns:
                        rev = qe['Revenue'].dropna().astype(float)
                        if len(rev)>=5: REV_Q_YOY = _safe_div(rev.iloc[-1]-rev.iloc[-5], rev.iloc[-5])
                        if len(rev)>=6:
                            yoy_now = _safe_div(rev.iloc[-1]-rev.iloc[-5], rev.iloc[-5]); yoy_prev = _safe_div(rev.iloc[-2]-rev.iloc[-6], rev.iloc[-6])
                            if pd.notna(yoy_now) and pd.notna(yoy_prev): REV_YOY_ACC = yoy_now - yoy_prev
                        yoy_list=[]
                        for k in range(1,5):
                            if len(rev)>=4+k:
                                y = _safe_div(rev.iloc[-k]-rev.iloc[-(k+4)], rev.iloc[-(k+4)])
                                if pd.notna(y): yoy_list.append(y)
                        if len(yoy_list)>=2: REV_YOY_VAR = float(np.std(yoy_list, ddof=1))
                    if 'Earnings' in qe.columns and so:
                        eps_series = (qe['Earnings'].dropna().astype(float)/float(so)).replace([np.inf,-np.inf],np.nan)
                        if len(eps_series)>=5 and pd.notna(eps_series.iloc[-5]) and eps_series.iloc[-5]!=0: EPS_Q_YOY = _safe_div(eps_series.iloc[-1]-eps_series.iloc[-5], eps_series.iloc[-5])
            except Exception: pass
            df.loc[t,'REV_Q_YOY'], df.loc[t,'EPS_Q_YOY'], df.loc[t,'REV_YOY_ACC'], df.loc[t,'REV_YOY_VAR'] = REV_Q_YOY, EPS_Q_YOY, REV_YOY_ACC, REV_YOY_VAR
            total_rev_ttm = d.get('totalRevenue',np.nan); fcf_ttm = fcf_df.loc[t,'fcf_ttm'] if t in fcf_df.index else np.nan
            FCF_MGN = _safe_div(fcf_ttm, total_rev_ttm); df.loc[t,'FCF_MGN'] = FCF_MGN
            rule40 = np.nan
            try: r = df.loc[t,'REV']; rule40 = (r if pd.notna(r) else np.nan) + (FCF_MGN if pd.notna(FCF_MGN) else np.nan)
            except Exception: pass
            df.loc[t,'RULE40'] = rule40
            sma50, sma150, sma200, p = s.rolling(50).mean(), s.rolling(150).mean(), s.rolling(200).mean(), _safe_last(s)
            df.loc[t,'P_OVER_150'] = p/_safe_last(sma150)-1 if pd.notna(_safe_last(sma150)) and _safe_last(sma150)!=0 else np.nan
            df.loc[t,'P_OVER_200'] = p/_safe_last(sma200)-1 if pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan
            df.loc[t,'MA50_OVER_200'] = _safe_last(sma50)/_safe_last(sma200)-1 if pd.notna(_safe_last(sma50)) and pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan
            df.loc[t,'MA200_SLOPE_5M'] = np.nan
            if len(sma200.dropna())>=105:
                cur200, old200 = _safe_last(sma200), float(sma200.iloc[-105])
                if old200 and old200!=0: df.loc[t,'MA200_SLOPE_5M'] = cur200/old200 - 1
            lo52 = s[-252:].min() if len(s)>=252 else s.min()
            df.loc[t,'LOW52PCT25_EXCESS'] = np.nan if (lo52 is None or lo52<=0 or pd.isna(p)) else (p/(lo52*1.25)-1)
            hi52 = s[-252:].max() if len(s)>=252 else s.max(); df.loc[t,'NEAR_52W_HIGH'] = np.nan
            if hi52 and hi52>0 and pd.notna(p):
                d_hi = (p/hi52)-1.0; df.loc[t,'NEAR_52W_HIGH'] = -abs(min(0.0, d_hi))
            df.loc[t,'RS_SLOPE_6W'], df.loc[t,'RS_SLOPE_13W'] = rs_line_slope(s, spx, 30), rs_line_slope(s, spx, 65)
            prior_50_high = s.rolling(50).max().shift(1); df.loc[t,'BASE_BRK_SIMPLE'] = np.nan
            if len(prior_50_high.dropna())>0 and pd.notna(p):
                ph = float(prior_50_high.iloc[-1]); cond50 = pd.notna(_safe_last(sma50)) and (p>_safe_last(sma50))
                df.loc[t,'BASE_BRK_SIMPLE'] = (p/ph-1) if (ph and ph>0 and cond50) else -0.0
            df.loc[t,'DIV_STREAK'] = div_streak(t)
            fin_cols = ['REV','ROE','BETA','DIV','FCF']; need_finnhub = [col for col in fin_cols if pd.isna(df.loc[t,col])]
            if need_finnhub:
                fin_data = fetch_finnhub_metrics(t)
                for col in need_finnhub:
                    val = fin_data.get(col)
                    if val is not None and not pd.isna(val): df.loc[t,col] = val
            for col in fin_cols + ['EPS','RS','TR_str','DIV_STREAK']:
                if pd.isna(df.loc[t,col]):
                    if col=='DIV':
                        status = dividend_status(t)
                        if status!='none_confident': missing_logs.append({'Ticker':t,'Column':col,'Status':status})
                    else: missing_logs.append({'Ticker':t,'Column':col})
        # ---- 安定化＆Z化 ----
        for col in ['ROE','FCF','REV','EPS']: df[f'{col}_W'] = winsorize_s(df[col], 0.02)
        df_z = pd.DataFrame(index=df.index)
        for col in ['EPS','REV','ROE','FCF','RS','TR_str','BETA','DIV','DIV_STREAK']: df_z[col] = robust_z(df[col])
        df_z['REV'], df_z['EPS'], df_z['TR'] = robust_z(df['REV_W']), robust_z(df['EPS_W']), robust_z(df['TR'])
        for col in ['P_OVER_150','P_OVER_200','MA50_OVER_200','MA200_SLOPE_5M','LOW52PCT25_EXCESS','NEAR_52W_HIGH','RS_SLOPE_6W','RS_SLOPE_13W','BASE_BRK_SIMPLE']: df_z[col] = robust_z(df[col])
        for col in ['REV_Q_YOY','EPS_Q_YOY','REV_YOY_ACC','REV_YOY_VAR','FCF_MGN','RULE40']: df_z[col] = robust_z(df[col])
        for col in ['DOWNSIDE_DEV','MDD_1Y','RESID_VOL','DOWN_OUTPERF','EXT_200','DIV_TTM_PS','DIV_VAR5','DIV_YOY','DIV_FCF_COVER','DEBT2EQ','CURR_RATIO','EPS_VAR_8Q','MARKET_CAP','ADV60_USD']: df_z[col] = robust_z(df[col])
        df_z['SIZE'], df_z['LIQ'] = robust_z(np.log1p(df['MARKET_CAP'])), robust_z(np.log1p(df['ADV60_USD']))
        df_z['QUALITY_F'] = robust_z(0.6*df['FCF_W'] + 0.4*df['ROE_W']).clip(-3.0,3.0)
        df_z['YIELD_F']   = 0.3*df_z['DIV'] + 0.7*df_z['DIV_STREAK']
        df_z['GROWTH_F']  = robust_z(0.30*df_z['REV'] + 0.20*df_z['EPS_Q_YOY'] + 0.15*df_z['REV_Q_YOY'] + 0.15*df_z['REV_YOY_ACC'] + 0.10*df_z['RULE40'] + 0.10*df_z['FCF_MGN'] - 0.05*df_z['REV_YOY_VAR']).clip(-3.0,3.0)
        df_z['MOM_F']     = robust_z(0.45*df_z['RS'] + 0.15*df_z['TR_str'] + 0.20*df_z['RS_SLOPE_6W'] + 0.20*df_z['RS_SLOPE_13W']).clip(-3.0,3.0)
        df_z['TREND']     = robust_z(0.20*df_z['TR'] + 0.12*df_z['P_OVER_150'] + 0.12*df_z['P_OVER_200'] + 0.16*df_z['MA50_OVER_200'] + 0.16*df_z['MA200_SLOPE_5M'] + 0.12*df_z['LOW52PCT25_EXCESS'] + 0.07*df_z['NEAR_52W_HIGH'] + 0.05*df_z['BASE_BRK_SIMPLE']).clip(-3.0,3.0)
        df_z['VOL'] = robust_z(df['BETA'])
        df_z.rename(columns={'GROWTH_F':'GRW','MOM_F':'MOM','TREND':'TRD','QUALITY_F':'QAL','YIELD_F':'YLD'}, inplace=True)
        if 'BETA' not in df_z.columns: df_z['BETA'] = robust_z(df['BETA'])
        # --- D側 ---
        df_z['D_VOL_RAW'] = robust_z(0.40*df_z['DOWNSIDE_DEV'] + 0.22*df_z['RESID_VOL'] + 0.18*df_z['MDD_1Y'] - 0.10*df_z['DOWN_OUTPERF'] - 0.05*df_z['EXT_200'] - 0.08*df_z['SIZE'] - 0.10*df_z['LIQ'] + 0.10*df_z['BETA'])
        df_z['D_QAL']     = robust_z(0.35*df_z['QAL'] + 0.20*df_z['FCF'] + 0.15*df_z['CURR_RATIO'] - 0.15*df_z['DEBT2EQ'] - 0.15*df_z['EPS_VAR_8Q'])
        df_z['D_YLD']     = robust_z(0.45*df_z['DIV'] + 0.25*df_z['DIV_STREAK'] + 0.20*df_z['DIV_FCF_COVER'] - 0.10*df_z['DIV_VAR5'])
        df_z['D_TRD']     = robust_z(0.40*df_z.get('MA200_SLOPE_5M',0) - 0.30*df_z.get('EXT_200',0) + 0.15*df_z.get('NEAR_52W_HIGH',0) + 0.15*df_z['TR'])
        g_score = df_z.mul(pd.Series(g_weights)).sum(axis=1)
        d_comp  = pd.concat({'QAL':df_z['D_QAL'],'YLD':df_z['D_YLD'],'VOL':df_z['D_VOL_RAW'],'TRD':df_z['D_TRD']}, axis=1)
        dw = pd.Series(D_weights, dtype=float).reindex(['QAL','YLD','VOL','TRD']).fillna(0.0)
        globals()['D_WEIGHTS_EFF'] = dw.copy()
        d_score_all = d_comp.mul(dw, axis=1).sum(axis=1)
        if debug_mode:
            eps = 0.1; _base = d_comp.mul(dw, axis=1).sum(axis=1); _test = d_comp.assign(VOL=d_comp['VOL']+eps).mul(dw, axis=1).sum(axis=1)
            print("VOL増→d_score低下の比率:", ((_test<=_base)|_test.isna()|_base.isna()).mean())
        return df, df_z, g_score, d_score_all, missing_logs

if 'select_buckets' not in globals():
    def select_buckets():
        """DRRSで相関低減しつつG/D選定。跨り相関μにも対応。"""
        global init_G, init_D, resG, resD, top_G, top_D
        init_G = g_score.nlargest(min(corrM, len(g_score))).index.tolist(); prevG = _load_prev(G_PREV_JSON)
        resG = select_bucket_drrs(returns_df=returns, score_ser=g_score, pool_tickers=init_G, k=N_G, n_pc=DRRS_G.get("n_pc",3), gamma=DRRS_G.get("gamma",1.0), lam=DRRS_G.get("lam",0.6), eta=DRRS_G.get("eta",0.5), lookback=DRRS_G.get("lookback",252), prev_tickers=prevG, shrink=DRRS_SHRINK, g_fixed_tickers=None, mu=0.0)
        top_G = resG["tickers"]
        D_pool_index = df_z.drop(top_G).index; d_score = d_score_all.drop(top_G)
        init_D = d_score.loc[D_pool_index].nlargest(min(corrM, len(D_pool_index))).index.tolist(); prevD = _load_prev(D_PREV_JSON)
        mu = globals().get('CROSS_MU_GD', 0.0)
        resD = select_bucket_drrs(returns_df=returns, score_ser=d_score_all, pool_tickers=init_D, k=N_D, n_pc=DRRS_D.get("n_pc",4), gamma=DRRS_D.get("gamma",0.8), lam=DRRS_D.get("lam",0.85), eta=DRRS_D.get("eta",0.5), lookback=DRRS_D.get("lookback",504), prev_tickers=prevD, shrink=DRRS_SHRINK, g_fixed_tickers=top_G, mu=mu)
        top_D = resD["tickers"]
        _save_sel(G_PREV_JSON, top_G, resG["avg_res_corr"], resG["sum_score"], resG["objective"])
        _save_sel(D_PREV_JSON, top_D, resD["avg_res_corr"], resD["sum_score"], resD["objective"])
        return resG, resD, top_G, top_D, init_G, init_D

def calculate_scores():
    """パイプライン：①スコア集計 → ②相関低減＆選定"""
    global df, missing_logs, df_z, g_score, d_score_all, init_G, init_D, resG, resD, top_G, top_D
    df, df_z, g_score, d_score_all, missing_logs = aggregate_scores()
    resG, resD, top_G, top_D, init_G, init_D = select_buckets()

def _avg_offdiag(A: np.ndarray) -> float:
    n = A.shape[0]; return np.nan if n<2 else float((A.sum()-np.trace(A))/(n*(n-1)))

def _resid_avg_rho(ret_df: pd.DataFrame, n_pc=3, shrink=DRRS_SHRINK) -> float:
    Rdf = ret_df.dropna()
    if Rdf.shape[0]<10 or Rdf.shape[1]<2: return np.nan
    Z = (Rdf - Rdf.mean())/(Rdf.std(ddof=0)+1e-9); U,S,_ = np.linalg.svd(Z.values, full_matrices=False)
    F = U[:,:n_pc]*S[:n_pc]; B = np.linalg.lstsq(F, Z.values, rcond=None)[0]; E = Z.values - F@B; C = np.corrcoef(E, rowvar=False)
    return _avg_offdiag(C)

def _raw_avg_rho(ret_df: pd.DataFrame) -> float:
    Rdf = ret_df.dropna()
    return np.nan if Rdf.shape[1]<2 else _avg_offdiag(Rdf.corr().values)

def _cross_block_raw_rho(ret_df: pd.DataFrame, left: list, right: list) -> float:
    R = ret_df[left+right].dropna()
    if len(left)<1 or len(right)<1 or R.shape[0]<10: return np.nan
    C = R.corr().loc[left,right].values; return float(np.nanmean(C))

def display_results():
    """Show results and prepare tables for Slack notification."""
    global miss_df, g_title, g_table, d_title, d_table, io_table, df_metrics_fmt, debug_table, div_details, g_formatters, d_formatters
    pd.set_option('display.float_format','{:.3f}'.format)
    print("📈 ファクター分散最適化の結果")
    miss_df = pd.DataFrame(missing_logs)
    if not miss_df.empty: print("Missing Data:"); print(miss_df.to_string(index=False))
    extra_G = [t for t in init_G if t not in top_G][:5]; G_UNI = top_G + extra_G
    g_table = pd.concat([df_z.loc[G_UNI,['GRW','MOM','TRD','VOL']], g_score[G_UNI].rename('GSC')], axis=1)
    g_table.index = [t + ("⭐️" if t in top_G else "") for t in G_UNI]
    g_formatters = {col:"{:.2f}".format for col in ['GRW','MOM','TRD','VOL']}; g_formatters['GSC'] = "{:.3f}".format
    g_title = (f"[G枠 / {N_G} / GRW{int(g_weights['GRW']*100)} MOM{int(g_weights['MOM']*100)} TRD{int(g_weights['TRD']*100)} VOL{int(g_weights['VOL']*100)} / corrM={corrM} / "
               f"LB={DRRS_G['lookback']} nPC={DRRS_G['n_pc']} γ={DRRS_G['gamma']} λ={DRRS_G['lam']} η={DRRS_G['eta']} shrink={DRRS_SHRINK}]")
    print(g_title); print(g_table.to_string(formatters=g_formatters))
    extra_D = [t for t in init_D if t not in top_D][:5]; D_UNI = top_D + extra_D
    cols_D = ['QAL','YLD','VOL','TRD']; d_disp = pd.DataFrame(index=D_UNI)
    d_disp['QAL'], d_disp['YLD'], d_disp['VOL'], d_disp['TRD'] = df_z.loc[D_UNI,'D_QAL'], df_z.loc[D_UNI,'D_YLD'], df_z.loc[D_UNI,'D_VOL_RAW'], df_z.loc[D_UNI,'D_TRD']
    d_table = pd.concat([d_disp, d_score_all[D_UNI].rename('DSC')], axis=1); d_table.index = [t + ("⭐️" if t in top_D else "") for t in D_UNI]
    d_formatters = {col:"{:.2f}".format for col in cols_D}; d_formatters['DSC']="{:.3f}".format
    d_title = (f"[D枠 / {N_D} / QAL{int(D_WEIGHTS_EFF['QAL']*100)} YLD{int(D_WEIGHTS_EFF['YLD']*100)} VOL{int(D_WEIGHTS_EFF['VOL']*100)} TRD{int(D_WEIGHTS_EFF['TRD']*100)} / corrM={corrM} / "
               f"LB={DRRS_D['lookback']} nPC={DRRS_D['n_pc']} γ={DRRS_D['gamma']} λ={DRRS_D['lam']} μ={CROSS_MU_GD} η={DRRS_D['eta']} shrink={DRRS_SHRINK}]")
    print(d_title); print(d_table.to_string(formatters=d_formatters))
    in_list = sorted(set(list(top_G)+list(top_D)) - set(exist)); out_list = sorted(set(exist) - set(list(top_G)+list(top_D)))
    in_df = pd.DataFrame({'IN':in_list}).reset_index(drop=True)
    out_df = pd.DataFrame({'/ OUT':out_list,'GSC':g_score.reindex(out_list).round(3).to_list(),'DSC':d_score_all.reindex(out_list).round(3).to_list()}).reset_index(drop=True)
    io_table = pd.concat([in_df, out_df], axis=1); print("Changes:"); print(io_table.to_string(index=False))
    all_tickers = list(set(exist + list(top_G) + list(top_D) + [bench])); prices = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False)['Close']
    ret = prices.pct_change(); portfolios = {'CUR':exist,'NEW':list(top_G)+list(top_D)}; metrics={}
    for name,ticks in portfolios.items():
        pr = ret[ticks].mean(axis=1, skipna=True).dropna(); cum = (1+pr).cumprod()-1; n = len(pr)
        if n>=252: ann_ret, ann_vol = (1+cum.iloc[-1])**(252/n)-1, pr.std()*np.sqrt(252)
        else: ann_ret, ann_vol = cum.iloc[-1], pr.std()*np.sqrt(n)
        sharpe, drawdown = ann_ret/ann_vol, (cum - cum.cummax()).min()
        if len(ticks)>=2:
            C_raw = ret[ticks].corr(); RAW_rho = C_raw.mask(np.eye(len(ticks), dtype=bool)).stack().mean()
            R = ret[ticks].dropna().to_numpy(); C_resid = residual_corr(R, n_pc=3, shrink=DRRS_SHRINK)
            RESID_rho = float((C_resid.sum()-np.trace(C_resid))/(C_resid.shape[0]*(C_resid.shape[0]-1)))
        else: RAW_rho = RESID_rho = np.nan
        metrics[name] = {'RET':ann_ret,'VOL':ann_vol,'SHP':sharpe,'MDD':drawdown,'RAWρ':RAW_rho,'RESIDρ':RESID_rho}
    ticks_G, ticks_D, ret_1y = list(top_G), list(top_D), ret
    div_details = {"NEW_rawρ":_raw_avg_rho(ret_1y[ticks_G+ticks_D]),"NEW_residρ":_resid_avg_rho(ret_1y[ticks_G+ticks_D], n_pc=max(DRRS_G["n_pc"],DRRS_D["n_pc"])),"G_rawρ":_raw_avg_rho(ret_1y[ticks_G]),"D_rawρ":_raw_avg_rho(ret_1y[ticks_D]),"G↔D_rawρ":_cross_block_raw_rho(ret_1y, ticks_G, ticks_D)}
    df_metrics = pd.DataFrame(metrics).T; df_metrics_pct = df_metrics.copy()
    for col in ['RET','VOL','MDD']: df_metrics_pct[col] = df_metrics_pct[col]*100
    cols_order = ['RET','VOL','SHP','MDD','RAWρ','RESIDρ']; df_metrics_pct = df_metrics_pct.reindex(columns=cols_order)
    def _fmt_row(s):
        return pd.Series({'RET':f"{s['RET']:.1f}%",'VOL':f"{s['VOL']:.1f}%",'SHP':f"{s['SHP']:.1f}",'MDD':f"{s['MDD']:.1f}%",'RAWρ':(f"{s['RAWρ']:.2f}" if pd.notna(s['RAWρ']) else "NaN"),'RESIDρ':(f"{s['RESIDρ']:.2f}" if pd.notna(s['RESIDρ']) else "NaN")})
    df_metrics_fmt = df_metrics_pct.apply(_fmt_row, axis=1); print("Performance Comparison:"); print(df_metrics_fmt.to_string())
    print("Diversification (NEW breakdown):"); 
    for k,v in div_details.items(): print(f"  {k}: {np.nan if v is None else round(v,3)}")
    debug_table = None
    if debug_mode:
        debug_table = pd.concat([df[['TR','EPS','REV','ROE','BETA','DIV','FCF','RS','TR_str','DIV_STREAK']], g_score.rename('GSC'), d_score_all.rename('DSC')], axis=1).round(3)
        print("Debug Data:"); print(debug_table.to_string())

def notify_slack():
    """Send results to Slack webhook."""
    SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
    if not SLACK_WEBHOOK_URL: raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")
    message = "📈 ファクター分散最適化の結果\n"
    if not miss_df.empty: message += "Missing Data\n```" + miss_df.to_string(index=False) + "```\n"
    message += g_title + "\n```" + g_table.to_string(formatters=g_formatters) + "```\n"
    message += d_title + "\n```" + d_table.to_string(formatters=d_formatters) + "```\n"
    message += "Changes\n```" + io_table.to_string(index=False) + "```\n"
    message += "Performance Comparison:\n```" + df_metrics_fmt.to_string() + "```"
    message += "\nDiversification (NEW breakdown):\n```" + "\n".join([f"{k}: {np.nan if v is None else round(v,3)}" for k,v in div_details.items()]) + "```"
    if debug_mode and debug_table is not None: message += "\nDebug Data\n```" + debug_table.to_string() + "```"
    payload = {"text": message}
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload); resp.raise_for_status(); print("✅ Slack（Webhook）へ送信しました")
    except Exception as e: print(f"⚠️ Slack通知エラー: {e}")

if __name__ == "__main__":
    prepare_data(); calculate_scores(); display_results(); notify_slack()
