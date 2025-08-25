# === NOTE: 機能・入出力・ログ文言・例外挙動は不変。安全な短縮（import統合/複数代入/内包表記/メソッドチェーン/一行化/空行圧縮など）のみ適用 ===
import yfinance as yf, pandas as pd, numpy as np, os, requests, time, json
from scipy.stats import zscore
from dataclasses import dataclass
from typing import Dict, List
from scorer import Scorer

# ===== ユニバースと定数（冒頭に固定） =====
exist, cand = [pd.read_csv(f, header=None)[0].tolist() for f in ("current_tickers.csv","candidate_tickers.csv")]
CAND_PRICE_MAX, bench = 400, '^GSPC'  # 価格上限・ベンチマーク
N_G, N_D = 12, 13  # G/D枠サイズ
g_weights = {'GRW':0.40,'MOM':0.40,'TRD':0.00,'VOL':-0.20}
D_BETA_MAX = float(os.environ.get("D_BETA_MAX", "0.8"))
D_D2E_MAX    = float(os.environ.get("D_D2E_MAX",   "500"))
D_GROWTH_MIN = float(os.environ.get("D_GROWTH_MIN","0"))
D_weights = {'QAL':0.15,'YLD':0.15,'VOL':-0.45,'TRD':0.25}

# DRRS 初期プール・各種パラメータ
corrM = 45
DRRS_G, DRRS_D = dict(lookback=252,n_pc=3,gamma=1.2,lam=0.68,eta=0.8), dict(lookback=504,n_pc=4,gamma=0.8,lam=0.85,eta=0.5)
DRRS_SHRINK = 0.10  # 残差相関の対角シュリンク（基礎）

# クロス相関ペナルティ（環境変数で指定可能）
CROSS_MU_GD = float(os.environ.get("CROSS_MU_GD", "0.40"))  # 推奨 0.35–0.45（lam=0.85想定）

# 出力関連
RESULTS_DIR, G_PREV_JSON, D_PREV_JSON = "results", os.path.join("results","G_selection.json"), os.path.join("results","D_selection.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# その他
debug_mode, FINNHUB_API_KEY = False, os.environ.get("FINNHUB_API_KEY")


# ===== 共有DTO（クラス間I/O契約）＋ Config =====
@dataclass(frozen=True)
class InputBundle:
    # Input → Scorer で受け渡す素材（I/O禁止の生データ）
    cand: List[str]
    tickers: List[str]
    bench: str
    data: pd.DataFrame              # yfinance download結果（'Close','Volume'等の階層列）
    px: pd.DataFrame                # data['Close']
    spx: pd.Series                  # data['Close'][bench]
    tickers_bulk: object            # yfinance.Tickers
    info: Dict[str, dict]           # yfinance info per ticker
    eps_df: pd.DataFrame            # ['eps_ttm','eps_q_recent',...]
    fcf_df: pd.DataFrame            # ['fcf_ttm', ...]
    returns: pd.DataFrame           # px[tickers].pct_change()

@dataclass(frozen=True)
class FeatureBundle:
    df: pd.DataFrame
    df_z: pd.DataFrame
    g_score: pd.Series
    d_score_all: pd.Series
    missing_logs: pd.DataFrame

@dataclass(frozen=True)
class SelectionBundle:
    resG: dict
    resD: dict
    top_G: List[str]
    top_D: List[str]
    init_G: List[str]
    init_D: List[str]

@dataclass(frozen=True)
class WeightsConfig:
    g: Dict[str,float]
    d: Dict[str,float]

@dataclass(frozen=True)
class DRRSParams:
    corrM: int
    shrink: float
    G: Dict[str,float]   # lookback, n_pc, gamma, lam, eta
    D: Dict[str,float]
    cross_mu_gd: float

@dataclass(frozen=True)
class PipelineConfig:
    weights: WeightsConfig
    drrs: DRRSParams
    price_max: float


# ===== 共通ユーティリティ（複数クラスで使用） =====
def winsorize_s(s: pd.Series, p=0.02):
    if s is None or s.dropna().empty: return s
    lo, hi = np.nanpercentile(s.astype(float), [100*p, 100*(1-p)]); return s.clip(lo, hi)

def robust_z(s: pd.Series, p=0.02):
    s2 = winsorize_s(s, p); return np.nan_to_num(zscore(s2.fillna(s2.mean())))

def _safe_div(a, b):
    try:
        if b is None or float(b)==0 or pd.isna(b): return np.nan
        return float(a)/float(b)
    except Exception: return np.nan

def _safe_last(series: pd.Series, default=np.nan):
    try: return float(series.iloc[-1])
    except Exception: return default

def _load_prev(path: str):
    try: return json.load(open(path)).get("tickers")
    except Exception: return None

def _save_sel(path: str, tickers: list[str], avg_r: float, sum_score: float, objective: float):
    with open(path,"w") as f:
        json.dump({"tickers":tickers,"avg_res_corr":round(avg_r,6),"sum_score":round(sum_score,6),"objective":round(objective,6)}, f, indent=2)


def _yf_growth_pos_5y(tk):
    """yfinance Growth Estimates の +5y 年率が D_GROWTH_MIN を超えるか"""
    try:
        ge = tk.growth_estimates  # Index: ['0q','+1q','0y','+1y','+5y','-5y']; Columns: ['stock','industry','sector','index']
        v = pd.to_numeric(ge.loc['+5y','stock'], errors='coerce')
        return bool(pd.notna(v) and float(v) > D_GROWTH_MIN)
    except Exception:
        return False


def _info_beta_lt(info_d: dict, thr: float) -> bool:
    """info['beta'] を用いて β<thr を判定（欠損/例外は False）"""
    try:
        b = info_d.get('beta')
        return bool(pd.notna(b) and float(b) < float(thr))
    except Exception:
        return False


def _guard_D_candidates(tickers, tickers_bulk, info: dict):
    """D枠プリフィルター：β<D_BETA_MAX 且つ +5y成長率>D_GROWTH_MIN 且つ D/E<D_D2E_MAX"""
    tks = getattr(tickers_bulk, 'tickers', {}) if tickers_bulk is not None else {}
    ok = []
    for t in tickers:
        inf = (info.get(t) or {})
        if not _info_beta_lt(inf, D_BETA_MAX):  # βゲート（欠損は落とす）
            continue
        d2e = inf.get('debtToEquity')
        if not (pd.notna(d2e) and float(d2e) < D_D2E_MAX):
            continue
        tk = tks.get(t) if isinstance(tks, dict) else yf.Ticker(t)
        if _yf_growth_pos_5y(tk):
            ok.append(t)
    return ok


# ===== Input：外部I/Oと前処理（CSV/API・欠損補完） =====
class Input:
    def __init__(self, cand, exist, bench, price_max, finnhub_api_key=None):
        self.cand, self.exist, self.bench, self.price_max = cand, exist, bench, price_max
        self.api_key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY")

    # ---- （Input専用）EPS補完・FCF算出系 ----
    @staticmethod
    def impute_eps_ttm(df: pd.DataFrame, ttm_col: str="eps_ttm", q_col: str="eps_q_recent", out_col: str|None=None) -> pd.DataFrame:
        out_col = out_col or ttm_col; df = df.copy(); df["eps_imputed"] = False
        cand = df[q_col]*4; ok = df[ttm_col].isna() & cand.replace([np.inf,-np.inf], np.nan).notna()
        df.loc[ok, out_col], df.loc[ok,"eps_imputed"] = cand[ok], True; return df

    _CF_ALIASES = {"cfo":["Operating Cash Flow","Total Cash From Operating Activities"], "capex":["Capital Expenditure","Capital Expenditures"]}

    @staticmethod
    def _pick_row(df: pd.DataFrame, names: list[str]) -> pd.Series|None:
        if df is None or df.empty: return None
        idx_lower = {str(i).lower(): i for i in df.index}
        for name in names:
            key = name.lower()
            if key in idx_lower: return df.loc[idx_lower[key]]
        return None

    @staticmethod
    def _sum_last_n(s: pd.Series|None, n: int) -> float|None:
        if s is None or s.empty: return None
        vals = s.dropna().astype(float); return None if vals.empty else vals.iloc[:n].sum()

    @staticmethod
    def _latest(s: pd.Series|None) -> float|None:
        if s is None or s.empty: return None
        vals = s.dropna().astype(float); return vals.iloc[0] if not vals.empty else None

    def fetch_cfo_capex_ttm_yf(self, tickers: list[str]) -> pd.DataFrame:
        rows=[]
        for t in tickers:
            tk = yf.Ticker(t); qcf = tk.quarterly_cashflow
            cfo_q, capex_q = self._pick_row(qcf,self._CF_ALIASES["cfo"]), self._pick_row(qcf,self._CF_ALIASES["capex"])
            fcf_q = self._pick_row(qcf, ["Free Cash Flow","FreeCashFlow","Free cash flow"])
            cfo_ttm, capex_ttm, fcf_ttm_direct = self._sum_last_n(cfo_q,4), self._sum_last_n(capex_q,4), self._sum_last_n(fcf_q,4)
            if cfo_ttm is None or capex_ttm is None or fcf_ttm_direct is None:
                acf = tk.cashflow
                cfo_a, capex_a, fcf_a = self._pick_row(acf,self._CF_ALIASES["cfo"]), self._pick_row(acf,self._CF_ALIASES["capex"]), self._pick_row(acf,["Free Cash Flow","FreeCashFlow","Free cash flow"])
                if cfo_ttm is None: cfo_ttm = self._latest(cfo_a)
                if capex_ttm is None: capex_ttm = self._latest(capex_a)
                if fcf_ttm_direct is None: fcf_ttm_direct = self._latest(fcf_a)
            rows.append({"ticker":t,"cfo_ttm_yf":cfo_ttm if cfo_ttm is not None else np.nan,"capex_ttm_yf":capex_ttm if capex_ttm is not None else np.nan,"fcf_ttm_yf_direct":fcf_ttm_direct if fcf_ttm_direct is not None else np.nan})
        return pd.DataFrame(rows).set_index("ticker")

    _FINN_CFO_KEYS = ["netCashProvidedByOperatingActivities","netCashFromOperatingActivities","cashFlowFromOperatingActivities","operatingCashFlow"]
    _FINN_CAPEX_KEYS = ["capitalExpenditure","capitalExpenditures","purchaseOfPPE","investmentsInPropertyPlantAndEquipment"]

    @staticmethod
    def _first_key(d: dict, keys: list[str]):
        for k in keys:
            if k in d and d[k] is not None: return d[k]
        return None

    @staticmethod
    def _finn_get(session: requests.Session, url: str, params: dict, retries: int=3, sleep_s: float=0.5):
        for i in range(retries):
            r = session.get(url, params=params, timeout=15)
            if r.status_code==429: time.sleep(min(2**i*sleep_s,4.0)); continue
            r.raise_for_status(); return r.json()
        r.raise_for_status()

    def fetch_cfo_capex_ttm_finnhub(self, tickers: list[str], api_key: str|None=None) -> pd.DataFrame:
        api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not api_key: raise ValueError("Finnhub API key not provided. Set FINNHUB_API_KEY or pass api_key=")
        base, s, rows = "https://finnhub.io/api/v1", requests.Session(), []
        for sym in tickers:
            cfo_ttm = capex_ttm = None
            try:
                j = self._finn_get(s, f"{base}/stock/cash-flow", {"symbol":sym,"frequency":"quarterly","limit":8,"token":api_key})
                arr = j.get("cashFlow") or []; cfo_vals, capex_vals = [], []
                for item in arr[:4]:
                    cfo_vals.append(self._first_key(item,self._FINN_CFO_KEYS)); capex_vals.append(self._first_key(item,self._FINN_CAPEX_KEYS))
                if any(v is not None for v in cfo_vals): cfo_ttm = float(np.nansum([np.nan if v is None else float(v) for v in cfo_vals]))
                if any(v is not None for v in capex_vals): capex_ttm = float(np.nansum([np.nan if v is None else float(v) for v in capex_vals]))
            except Exception: pass
            if cfo_ttm is None or capex_ttm is None:
                try:
                    j = self._finn_get(s, f"{base}/stock/cash-flow", {"symbol":sym,"frequency":"annual","limit":1,"token":api_key})
                    arr = j.get("cashFlow") or []
                    if arr:
                        item0 = arr[0]
                        if cfo_ttm is None:
                            v = self._first_key(item0,self._FINN_CFO_KEYS)
                            if v is not None: cfo_ttm = float(v)
                        if capex_ttm is None:
                            v = self._first_key(item0,self._FINN_CAPEX_KEYS)
                            if v is not None: capex_ttm = float(v)
                except Exception: pass
            rows.append({"ticker":sym,"cfo_ttm_fh":np.nan if cfo_ttm is None else cfo_ttm,"capex_ttm_fh":np.nan if capex_ttm is None else capex_ttm})
        return pd.DataFrame(rows).set_index("ticker")

    def compute_fcf_with_fallback(self, tickers: list[str], finnhub_api_key: str|None=None) -> pd.DataFrame:
        yf_df, fh_df = self.fetch_cfo_capex_ttm_yf(tickers), self.fetch_cfo_capex_ttm_finnhub(tickers, api_key=finnhub_api_key)
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

    def _build_eps_df(self, tickers, tickers_bulk, info):
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
        return self.impute_eps_ttm(pd.DataFrame(eps_rows).set_index("ticker"))

    def prepare_data(self):
        """Fetch price and fundamental data for all tickers."""
        cand_info = yf.Tickers(" ".join(self.cand)); cand_prices = {}
        for t in self.cand:
            try: cand_prices[t] = cand_info.tickers[t].fast_info.get("lastPrice", np.inf)
            except Exception as e: print(f"{t}: price fetch failed ({e})"); cand_prices[t] = np.inf
        cand_f = [t for t,p in cand_prices.items() if p<=self.price_max]
        tickers = sorted(set(self.exist + cand_f))
        data = yf.download(tickers + [self.bench], period="600d", auto_adjust=True, progress=False)
        px, spx = data["Close"], data["Close"][self.bench]
        tickers_bulk, info = yf.Tickers(" ".join(tickers)), {}
        for t in tickers:
            try: info[t] = tickers_bulk.tickers[t].info
            except Exception as e: print(f"{t}: info fetch failed ({e})"); info[t] = {}
        eps_df = self._build_eps_df(tickers, tickers_bulk, info)
        fcf_df = self.compute_fcf_with_fallback(tickers, finnhub_api_key=self.api_key)
        returns = px[tickers].pct_change()
        return dict(cand=cand_f, tickers=tickers, data=data, px=px, spx=spx, tickers_bulk=tickers_bulk, info=info, eps_df=eps_df, fcf_df=fcf_df, returns=returns)


# ===== Selector：相関低減・選定（スコア＆リターンだけ読む） =====
class Selector:
    # ---- DRRS helpers（Selector専用） ----
    @staticmethod
    def _z_np(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32); m = np.nanmean(X, axis=0, keepdims=True); s = np.nanstd(X, axis=0, keepdims=True)+1e-9
        return (np.nan_to_num(X)-m)/s

    @classmethod
    def residual_corr(cls, R: np.ndarray, n_pc: int=3, shrink: float=0.1) -> np.ndarray:
        Z = cls._z_np(R); U,S,_ = np.linalg.svd(Z, full_matrices=False); F = U[:,:n_pc]*S[:n_pc]; B = np.linalg.lstsq(F, Z, rcond=None)[0]
        E = Z - F@B; C = np.corrcoef(E, rowvar=False)
        off = C - np.diag(np.diag(C)); iu = np.triu_indices_from(off,1); avg_abs = np.nanmean(np.abs(off[iu])) if iu[0].size else 0.0
        shrink_eff = float(np.clip(shrink + 0.5*avg_abs, 0.1, 0.6)); N = C.shape[0]
        return (1.0 - shrink_eff)*C + shrink_eff*np.eye(N, dtype=C.dtype)

    @classmethod
    def rrqr_like_det(cls, R: np.ndarray, score: np.ndarray, k: int, gamma: float=1.0):
        Z, w = cls._z_np(R), (score-score.min())/(np.ptp(score)+1e-12); X = Z*(1.0 + gamma*w)
        N, k = X.shape[1], max(0, min(k, X.shape[1]))
        if k==0: return []
        S, selected, Rres = [], np.zeros(N, dtype=bool), X.copy()
        for _ in range(k):
            norms = (Rres*Rres).sum(axis=0); cand = np.where(~selected)[0]
            j = sorted(cand, key=lambda c:(-norms[c], -w[c], c))[0]
            S.append(j); selected[j]=True; u = X[:,j:j+1]; u/=(np.linalg.norm(u)+1e-12); Rres = Rres - u @ (u.T @ Rres)
        return sorted(S)

    @staticmethod
    def _obj(corrM: np.ndarray, score: np.ndarray, idx, lam: float) -> float:
        idx = list(idx); P = corrM[np.ix_(idx, idx)]; s = (score-score.mean())/(score.std()+1e-9)
        return float(s[idx].sum() - lam*((P.sum()-np.trace(P))/2.0))

    @classmethod
    def swap_local_det(cls, corrM: np.ndarray, score: np.ndarray, idx, lam: float=0.6, max_pass: int=15):
        S, best, improved, passes = sorted(idx), cls._obj(corrM, score, idx, lam), True, 0
        while improved and passes<max_pass:
            improved, passes = False, passes+1
            for i,out in enumerate(list(S)):
                for inn in range(len(score)):
                    if inn in S: continue
                    cand = sorted(S[:i]+[inn]+S[i+1:]); v = cls._obj(corrM, score, cand, lam)
                    if v>best+1e-10: S, best, improved = cand, v, True; break
                if improved: break
        return S, best

    @staticmethod
    def _obj_with_cross(C_within: np.ndarray, C_cross: np.ndarray|None, score: np.ndarray, idx, lam: float, mu: float) -> float:
        idx = list(idx); P = C_within[np.ix_(idx, idx)]; s = (score-score.mean())/(score.std()+1e-9)
        within = (P.sum()-np.trace(P))/2.0; cross = 0.0
        if C_cross is not None and C_cross.size>0: cross = C_cross[idx,:].sum()
        return float(s[idx].sum() - lam*within - mu*cross)

    @classmethod
    def swap_local_det_cross(cls, C_within: np.ndarray, C_cross: np.ndarray|None, score: np.ndarray, idx, lam: float=0.6, mu: float=0.3, max_pass: int=15):
        S, best, improved, passes, N = sorted(idx), cls._obj_with_cross(C_within,C_cross,score,idx,lam,mu), True, 0, len(score)
        while improved and passes<max_pass:
            improved, passes = False, passes+1
            for i,out in enumerate(list(S)):
                for inn in range(N):
                    if inn in S: continue
                    cand = sorted(S[:i]+[inn]+S[i+1:]); v = cls._obj_with_cross(C_within,C_cross,score,cand,lam,mu)
                    if v>best+1e-10: S, best, improved = cand, v, True; break
                if improved: break
        return S, best

    @staticmethod
    def avg_corr(C: np.ndarray, idx) -> float:
        k = len(idx); P = C[np.ix_(idx, idx)]
        return float((P.sum()-np.trace(P))/(k*(k-1)+1e-12))

    @classmethod
    def select_bucket_drrs(cls, returns_df: pd.DataFrame, score_ser: pd.Series, pool_tickers: list[str], k: int, *, n_pc: int, gamma: float, lam: float, eta: float, lookback: int, prev_tickers: list[str]|None, shrink: float=0.10, g_fixed_tickers: list[str]|None=None, mu: float=0.0):
        g_fixed = [t for t in (g_fixed_tickers or []) if t in returns_df.columns]
        union = [t for t in pool_tickers if t in returns_df.columns]
        for t in g_fixed:
            if t not in union: union.append(t)
        Rdf_all = returns_df[union]; Rdf_all = Rdf_all.iloc[-lookback:] if len(Rdf_all)>=lookback else Rdf_all; Rdf_all = Rdf_all.dropna()
        pool_eff, g_eff = [t for t in pool_tickers if t in Rdf_all.columns], [t for t in g_fixed if t in Rdf_all.columns]
        if len(pool_eff)==0: return dict(idx=[], tickers=[], avg_res_corr=np.nan, sum_score=0.0, objective=-np.inf)
        score = score_ser.reindex(pool_eff).to_numpy(dtype=np.float32)
        C_all = cls.residual_corr(Rdf_all.to_numpy(), n_pc=n_pc, shrink=shrink)
        col_pos = {c:i for i,c in enumerate(Rdf_all.columns)}; pool_pos = [col_pos[t] for t in pool_eff]
        C_within, C_cross = C_all[np.ix_(pool_pos,pool_pos)], None
        if len(g_eff)>0 and mu>0.0:
            g_pos = [col_pos[t] for t in g_eff]; C_cross = C_all[np.ix_(pool_pos,g_pos)]
        R_pool = Rdf_all[pool_eff].to_numpy(); S0 = cls.rrqr_like_det(R_pool, score, k, gamma=gamma)
        S, Jn = (cls.swap_local_det_cross(C_within, C_cross, score, S0, lam=lam, mu=mu, max_pass=15) if C_cross is not None else cls.swap_local_det(C_within, score, S0, lam=lam, max_pass=15))
        if prev_tickers:
            prev_idx = [pool_eff.index(t) for t in prev_tickers if t in pool_eff]
            if len(prev_idx)==min(k,len(pool_eff)):
                Jp = (cls._obj_with_cross(C_within,C_cross,score,prev_idx,lam,mu) if C_cross is not None else cls._obj(C_within,score,prev_idx,lam))
                if Jn < Jp + eta: S, Jn = sorted(prev_idx), Jp
        selected_tickers = [pool_eff[i] for i in S]
        return dict(idx=S, tickers=selected_tickers, avg_res_corr=cls.avg_corr(C_within,S), sum_score=float(score[S].sum()), objective=float(Jn))

    # ---- 選定（スコア Series / returns だけを受ける）----
    def select_buckets(self, returns_df: pd.DataFrame, g_score: pd.Series, d_score_all: pd.Series, cfg: PipelineConfig) -> SelectionBundle:
        init_G = g_score.nlargest(min(cfg.drrs.corrM, len(g_score))).index.tolist(); prevG = _load_prev(G_PREV_JSON)
        resG = self.select_bucket_drrs(returns_df=returns_df, score_ser=g_score, pool_tickers=init_G, k=N_G,
                                       n_pc=cfg.drrs.G.get("n_pc",3), gamma=cfg.drrs.G.get("gamma",1.2),
                                       lam=cfg.drrs.G.get("lam",0.68), eta=cfg.drrs.G.get("eta",0.8),
                                       lookback=cfg.drrs.G.get("lookback",252), prev_tickers=prevG,
                                       shrink=cfg.drrs.shrink, g_fixed_tickers=None, mu=0.0)
        top_G = resG["tickers"]

        # df_z に依存せず、スコアの index から D プールを構成（機能は同等）
        # === D candidates pre-filter (β + +5y成長 + D/E) ===
        cand_D = _guard_D_candidates(cand, tickers_bulk, info)
        d_score = d_score_all.drop(top_G, errors='ignore').reindex(cand_D).dropna()
        D_pool_index = d_score.index
        init_D = d_score.loc[D_pool_index].nlargest(min(cfg.drrs.corrM, len(D_pool_index))).index.tolist(); prevD = _load_prev(D_PREV_JSON)
        mu = cfg.drrs.cross_mu_gd
        resD = self.select_bucket_drrs(returns_df=returns_df, score_ser=d_score_all, pool_tickers=init_D, k=N_D,
                                       n_pc=cfg.drrs.D.get("n_pc",4), gamma=cfg.drrs.D.get("gamma",0.8),
                                       lam=cfg.drrs.D.get("lam",0.85), eta=cfg.drrs.D.get("eta",0.5),
                                       lookback=cfg.drrs.D.get("lookback",504), prev_tickers=prevD,
                                       shrink=cfg.drrs.shrink, g_fixed_tickers=top_G, mu=mu)
        top_D = resD["tickers"]

        _save_sel(G_PREV_JSON, top_G, resG["avg_res_corr"], resG["sum_score"], resG["objective"])
        _save_sel(D_PREV_JSON, top_D, resD["avg_res_corr"], resD["sum_score"], resD["objective"])
        return SelectionBundle(resG=resG, resD=resD, top_G=top_G, top_D=top_D, init_G=init_G, init_D=init_D)


# ===== Output：出力整形と送信（表示・Slack） =====
class Output:
    def __init__(self, debug=False):
        self.debug = debug
        self.miss_df = self.g_table = self.d_table = self.io_table = self.df_metrics_fmt = self.debug_table = None
        self.g_title = self.d_title = ""
        self.div_details = {}
        self.g_formatters = self.d_formatters = {}
        # 低スコア（GSC+DSC）Top10 表示/送信用
        self.low10_table = None

    # --- メトリクス補助（Output専用） ---
    @staticmethod
    def _avg_offdiag(A: np.ndarray) -> float:
        n = A.shape[0]; return np.nan if n<2 else float((A.sum()-np.trace(A))/(n*(n-1)))

    @classmethod
    def _resid_avg_rho(cls, ret_df: pd.DataFrame, n_pc=3, shrink=DRRS_SHRINK) -> float:
        Rdf = ret_df.dropna()
        if Rdf.shape[0]<10 or Rdf.shape[1]<2: return np.nan
        Z = (Rdf - Rdf.mean())/(Rdf.std(ddof=0)+1e-9); U,S,_ = np.linalg.svd(Z.values, full_matrices=False)
        F = U[:,:n_pc]*S[:n_pc]; B = np.linalg.lstsq(F, Z.values, rcond=None)[0]; E = Z.values - F@B; C = np.corrcoef(E, rowvar=False)
        return cls._avg_offdiag(C)

    @staticmethod
    def _raw_avg_rho(ret_df: pd.DataFrame) -> float:
        Rdf = ret_df.dropna()
        return np.nan if Rdf.shape[1]<2 else float(Output._avg_offdiag(Rdf.corr().values))

    @staticmethod
    def _cross_block_raw_rho(ret_df: pd.DataFrame, left: list, right: list) -> float:
        R = ret_df[left+right].dropna()
        if len(left)<1 or len(right)<1 or R.shape[0]<10: return np.nan
        C = R.corr().loc[left,right].values; return float(np.nanmean(C))

    # --- 表示（元 display_results のロジックそのまま） ---
    def display_results(self, exist, bench, df_z, g_score, d_score_all, init_G, init_D, top_G, top_D):
        pd.set_option('display.float_format','{:.3f}'.format)
        print("📈 ファクター分散最適化の結果")
        if self.miss_df is not None and not self.miss_df.empty: print("Missing Data:"); print(self.miss_df.to_string(index=False))

        extra_G = [t for t in init_G if t not in top_G][:5]; G_UNI = top_G + extra_G
        self.g_table = pd.concat([df_z.loc[G_UNI,['GRW','MOM','TRD','VOL']], g_score[G_UNI].rename('GSC')], axis=1)
        self.g_table.index = [t + ("⭐️" if t in top_G else "") for t in G_UNI]
        self.g_formatters = {col:"{:.2f}".format for col in ['GRW','MOM','TRD','VOL']}; self.g_formatters['GSC'] = "{:.3f}".format
        self.g_title = (f"[G枠 / {N_G} / GRW{int(g_weights['GRW']*100)} MOM{int(g_weights['MOM']*100)} TRD{int(g_weights['TRD']*100)} VOL{int(g_weights['VOL']*100)} / corrM={corrM} / "
                        f"LB={DRRS_G['lookback']} nPC={DRRS_G['n_pc']} γ={DRRS_G['gamma']} λ={DRRS_G['lam']} η={DRRS_G['eta']} shrink={DRRS_SHRINK}]")
        print(self.g_title); print(self.g_table.to_string(formatters=self.g_formatters))

        extra_D = [t for t in init_D if t not in top_D][:5]; D_UNI = top_D + extra_D
        cols_D = ['QAL','YLD','VOL','TRD']; d_disp = pd.DataFrame(index=D_UNI)
        d_disp['QAL'], d_disp['YLD'], d_disp['VOL'], d_disp['TRD'] = df_z.loc[D_UNI,'D_QAL'], df_z.loc[D_UNI,'D_YLD'], df_z.loc[D_UNI,'D_VOL_RAW'], df_z.loc[D_UNI,'D_TRD']
        self.d_table = pd.concat([d_disp, d_score_all[D_UNI].rename('DSC')], axis=1); self.d_table.index = [t + ("⭐️" if t in top_D else "") for t in D_UNI]
        self.d_formatters = {col:"{:.2f}".format for col in cols_D}; self.d_formatters['DSC']="{:.3f}".format
        import scorer
        dw_eff = scorer.D_WEIGHTS_EFF
        self.d_title = (f"[D枠 / {N_D} / QAL{int(dw_eff['QAL']*100)} YLD{int(dw_eff['YLD']*100)} VOL{int(dw_eff['VOL']*100)} TRD{int(dw_eff['TRD']*100)} / corrM={corrM} / "
                        f"LB={DRRS_D['lookback']} nPC={DRRS_D['n_pc']} γ={DRRS_D['gamma']} λ={DRRS_D['lam']} μ={CROSS_MU_GD} η={DRRS_D['eta']} shrink={DRRS_SHRINK}]")
        print(self.d_title); print(self.d_table.to_string(formatters=self.d_formatters))

        # === Changes（IN の GSC/DSC を表示。OUT は銘柄名のみ） ===
        in_list = sorted(set(list(top_G)+list(top_D)) - set(exist))
        out_list = sorted(set(exist) - set(list(top_G)+list(top_D)))

        changes = pd.DataFrame({'IN': pd.Series(in_list), '/ OUT': pd.Series(out_list)})
        s_in = g_score.combine_first(d_score_all)      # IN は G → D
        s_out = d_score_all.combine_first(g_score)      # OUT は D → G
        fmt = (lambda v: '—' if pd.isna(v) else f'{v:.3f}')
        changes['GSC'] = changes['IN'].map(s_in).map(fmt)
        changes['DSC'] = changes['/ OUT'].map(s_out).map(fmt)
        self.io_table = changes

        print("Changes:")
        print(self.io_table.to_string(index=False, na_rep="NaN"))

        all_tickers = list(set(exist + list(top_G) + list(top_D) + [bench])); prices = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False)['Close']
        ret = prices.pct_change(); portfolios = {'CUR':exist,'NEW':list(top_G)+list(top_D)}; metrics={}
        for name,ticks in portfolios.items():
            pr = ret[ticks].mean(axis=1, skipna=True).dropna(); cum = (1+pr).cumprod()-1; n = len(pr)
            if n>=252: ann_ret, ann_vol = (1+cum.iloc[-1])**(252/n)-1, pr.std()*np.sqrt(252)
            else: ann_ret, ann_vol = cum.iloc[-1], pr.std()*np.sqrt(n)
            sharpe, drawdown = ann_ret/ann_vol, (cum - cum.cummax()).min()
            if len(ticks)>=2:
                C_raw = ret[ticks].corr(); RAW_rho = C_raw.mask(np.eye(len(ticks), dtype=bool)).stack().mean()
                R = ret[ticks].dropna().to_numpy(); C_resid = Selector.residual_corr(R, n_pc=3, shrink=DRRS_SHRINK)
                RESID_rho = float((C_resid.sum()-np.trace(C_resid))/(C_resid.shape[0]*(C_resid.shape[0]-1)))
            else: RAW_rho = RESID_rho = np.nan
            metrics[name] = {'RET':ann_ret,'VOL':ann_vol,'SHP':sharpe,'MDD':drawdown,'RAWρ':RAW_rho,'RESIDρ':RESID_rho}
        ticks_G, ticks_D, ret_1y = list(top_G), list(top_D), ret
        self.div_details = {"NEW_rawρ":self._raw_avg_rho(ret_1y[ticks_G+ticks_D]),"NEW_residρ":self._resid_avg_rho(ret_1y, n_pc=max(DRRS_G["n_pc"],DRRS_D["n_pc"])),"G_rawρ":self._raw_avg_rho(ret_1y[ticks_G]),"D_rawρ":self._raw_avg_rho(ret_1y[ticks_D]),"G↔D_rawρ":self._cross_block_raw_rho(ret_1y, ticks_G, ticks_D)}
        df_metrics = pd.DataFrame(metrics).T; df_metrics_pct = df_metrics.copy()
        for col in ['RET','VOL','MDD']: df_metrics_pct[col] = df_metrics_pct[col]*100
        cols_order = ['RET','VOL','SHP','MDD','RAWρ','RESIDρ']; df_metrics_pct = df_metrics_pct.reindex(columns=cols_order)
        def _fmt_row(s):
            return pd.Series({'RET':f"{s['RET']:.1f}%",'VOL':f"{s['VOL']:.1f}%",'SHP':f"{s['SHP']:.1f}",'MDD':f"{s['MDD']:.1f}%",'RAWρ':(f"{s['RAWρ']:.2f}" if pd.notna(s['RAWρ']) else "NaN"),'RESIDρ':(f"{s['RESIDρ']:.2f}" if pd.notna(s['RESIDρ']) else "NaN")})
        self.df_metrics_fmt = df_metrics_pct.apply(_fmt_row, axis=1); print("Performance Comparison:"); print(self.df_metrics_fmt.to_string())
        print("Diversification (NEW breakdown):"); 
        for k,v in self.div_details.items(): print(f"  {k}: {np.nan if v is None else round(v,3)}")
        if self.debug:
            self.debug_table = pd.concat([df_z[['TR','EPS','REV','ROE','BETA','DIV','FCF','RS','TR_str','DIV_STREAK']], g_score.rename('GSC'), d_score_all.rename('DSC')], axis=1).round(3)
            print("Debug Data:"); print(self.debug_table.to_string())

        # === 追加: GSC+DSC が低い順 TOP10 ===
        try:
            all_scores = pd.DataFrame({'GSC': df_z['GSC'], 'DSC': df_z['DSC']}).copy()
            all_scores['G_plus_D'] = all_scores['GSC'] + all_scores['DSC']
            all_scores = all_scores.dropna(subset=['G_plus_D'])
            self.low10_table = all_scores.sort_values('G_plus_D', ascending=True).head(10).round(3)
            print("Low Score Candidates (GSC+DSC bottom 10):")
            print(self.low10_table.to_string())
        except Exception as e:
            print(f"[warn] low-score ranking failed: {e}")
            self.low10_table = None

    # --- Slack送信（元 notify_slack のロジックそのまま） ---
    def notify_slack(self):
        SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
        if not SLACK_WEBHOOK_URL: raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")
        message = "📈 ファクター分散最適化の結果\n"
        if self.miss_df is not None and not self.miss_df.empty: message += "Missing Data\n```" + self.miss_df.to_string(index=False) + "```\n"
        message += self.g_title + "\n```" + self.g_table.to_string(formatters=self.g_formatters) + "```\n"
        message += self.d_title + "\n```" + self.d_table.to_string(formatters=self.d_formatters) + "```\n"
        message += "Changes\n```" + self.io_table.to_string(index=False) + "```\n"
        # 低スコアTOP10（GSC+DSC）
        if self.low10_table is not None:
            message += "Low Score Candidates (GSC+DSC bottom 10)\n```" + self.low10_table.to_string() + "```\n"
        message += "Performance Comparison:\n```" + self.df_metrics_fmt.to_string() + "```"
        message += "\nDiversification (NEW breakdown):\n```" + "\n".join([f"{k}: {np.nan if v is None else round(v,3)}" for k,v in self.div_details.items()]) + "```"
        if self.debug and self.debug_table is not None: message += "\nDebug Data\n```" + self.debug_table.to_string() + "```"
        payload = {"text": message}
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json=payload); resp.raise_for_status(); print("✅ Slack（Webhook）へ送信しました")
        except Exception as e: print(f"⚠️ Slack通知エラー: {e}")


def _infer_g_universe(feature_df, selected12=None, near5=None):
    try:
        out = feature_df.index[feature_df['group'].astype(str).str.upper().eq('G')].tolist()
        if out: return out
    except Exception:
        pass
    base = set()
    for lst in (selected12 or []), (near5 or []):
        for x in (lst or []): base.add(x)
    return list(base) if base else list(feature_df.index)


def _fmt_with_fire_mark(tickers, feature_df):
    out = []
    for t in tickers or []:
        try:
            br = bool(feature_df.at[t, "G_BREAKOUT_recent_5d"])
            pb = bool(feature_df.at[t, "G_PULLBACK_recent_5d"])
            tt = bool(feature_df.at[t, "TT_PASS"])
            out.append(f"{t}{' 🔥' if tt and (br or pb) else ''}")
        except Exception:
            out.append(t)
    return out


def _label_recent_event(t, feature_df):
    try:
        br = bool(feature_df.at[t, "G_BREAKOUT_recent_5d"]); dbr = str(feature_df.at[t, "G_BREAKOUT_last_date"]) if br else ""
        pb = bool(feature_df.at[t, "G_PULLBACK_recent_5d"]); dpb = str(feature_df.at[t, "G_PULLBACK_last_date"]) if pb else ""
        if   br and not pb: return f"{t}（ブレイクアウト確定 {dbr}）"
        elif pb and not br: return f"{t}（押し目反発 {dpb}）"
        elif br and pb:     return f"{t}（ブレイクアウト確定 {dbr}／押し目反発 {dpb}）"
    except Exception:
        pass
    return t


# ===== エントリポイント =====
if __name__ == "__main__":
    # 0) Config を束ねる（元の定数をそのまま使用）
    cfg = PipelineConfig(
        weights=WeightsConfig(g=g_weights, d=D_weights),
        drrs=DRRSParams(corrM=corrM, shrink=DRRS_SHRINK, G=DRRS_G, D=DRRS_D, cross_mu_gd=CROSS_MU_GD),
        price_max=CAND_PRICE_MAX
    )

    # 1) 入力（外部I/Oと前処理）
    inp = Input(cand=cand, exist=exist, bench=bench, price_max=cfg.price_max, finnhub_api_key=FINNHUB_API_KEY)
    state = inp.prepare_data()
    tickers_bulk, info = state["tickers_bulk"], state["info"]
    ib = InputBundle(cand=state["cand"], tickers=state["tickers"], bench=bench, data=state["data"], px=state["px"], spx=state["spx"], tickers_bulk=tickers_bulk, info=info, eps_df=state["eps_df"], fcf_df=state["fcf_df"], returns=state["returns"])

    # 2) 集計（純粋）：特徴量→Z化→合成スコア
    scorer = Scorer()
    fb = scorer.aggregate_scores(ib, cfg)

    # 2.5) 選定（相関低減）
    selector = Selector()
    sb = selector.select_buckets(
        returns_df=ib.returns,
        g_score=fb.g_score,
        d_score_all=fb.d_score_all,
        cfg=cfg
    )

    selected12 = sb.top_G  # 既存ロジックのまま

    # 次点5：g_score降順から選定12を除いた上位5
    try:
        ranked_all = fb.g_score.sort_values(ascending=False)
        near5 = [t for t in ranked_all.index if t not in selected12][:5]
    except Exception:
        near5 = []

    df   = fb.df
    guni = _infer_g_universe(df, selected12, near5)

    # 直近5営業日のイベント（Gユニバース内のみ）
    try:
        fire_recent = [t for t in guni
                       if str(df.at[t,"TT_PASS"])=="True" and ((str(df.at[t,"G_BREAKOUT_recent_5d"])=="True") or (str(df.at[t,"G_PULLBACK_recent_5d"])=="True"))]
    except Exception:
        fire_recent = []

    lines = []
    lines.append("【G枠レポート｜週次モニタ（直近5営業日）】")
    lines.append("【凡例】🔥=直近5営業日内に「ブレイクアウト確定」または「押し目反発」を検知")
    lines.append(f"選定12: {', '.join(_fmt_with_fire_mark(selected12, df))}" if selected12 else "選定12: なし")
    lines.append(f"次点5: {', '.join(_fmt_with_fire_mark(near5, df))}" if near5 else "次点5: なし")

    if fire_recent:
        fire_list = ", ".join([_label_recent_event(t, df) for t in fire_recent])
        lines.append(f"過去5営業日の検知: {fire_list}")
    else:
        lines.append("過去5営業日の検知: なし")

    try:
        webhook = os.environ.get("SLACK_WEBHOOK_URL","")
        if webhook:
            requests.post(webhook, json={"text":"\n".join(lines)}, timeout=10)
    except Exception:
        pass

    # 3) 出力（表示→Slack） — 既存I/Fのまま
    out = Output(debug=debug_mode)
    out.miss_df = fb.missing_logs  # 互換表示のため display_results 内で利用
    out.display_results(exist=exist, bench=bench, df_z=fb.df_z, g_score=fb.g_score, d_score_all=fb.d_score_all,
                        init_G=sb.init_G, init_D=sb.init_D, top_G=sb.top_G, top_D=sb.top_D)
    out.notify_slack()
