"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ROLE of factor.py                                     ┃
┃  - Orchestration ONLY（外部I/O・SSOT・Slack出力）     ┃
┃  - 計算ロジック（採点/フィルタ/相関低減）は scorer.py ┃
┃  - ここでロジックを実装/変更しない                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""
# === NOTE: 機能・入出力・ログ文言・例外挙動は不変。安全な短縮（import統合/複数代入/内包表記/メソッドチェーン/一行化/空行圧縮など）のみ適用 ===
import yfinance as yf, pandas as pd, numpy as np, os, requests, time, json
from scipy.stats import zscore
from dataclasses import dataclass
from typing import Dict, List
from scorer import Scorer, ttm_div_yield_portfolio
import os
import requests
from time import perf_counter


class T:
    t = perf_counter()

    @staticmethod
    def log(tag: str):
        now = perf_counter()
        print(f"[T] {tag}: {now - T.t:.2f}s")
        T.t = now


T.log("start")

# ===== ユニバースと定数（冒頭に固定） =====
exist, cand = [pd.read_csv(f, header=None)[0].tolist() for f in ("current_tickers.csv","candidate_tickers.csv")]
T.log(f"csv loaded: exist={len(exist)} cand={len(cand)}")
CAND_PRICE_MAX, bench = 450, '^GSPC'  # 価格上限・ベンチマーク
N_G, N_D = 12, 13  # G/D枠サイズ
g_weights = {'GRW':0.40,'MOM':0.45,'VOL':-0.15}
D_BETA_MAX = float(os.environ.get("D_BETA_MAX", "0.8"))
FILTER_SPEC = {"G":{"pre_mask":["trend_template"]},"D":{"pre_filter":{"beta_max":D_BETA_MAX}}}
D_weights = {'QAL':0.15,'YLD':0.15,'VOL':-0.45,'TRD':0.25}
def _fmt_w(w): return " ".join(f"{k}{int(v*100)}" for k,v in w.items())

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
debug_mode, FINNHUB_API_KEY = True, os.environ.get("FINNHUB_API_KEY")


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

def _env_true(name: str, default=False):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() == "true"

def _slack(message, code=False):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        print("⚠️ SLACK_WEBHOOK_URL 未設定"); return
    try:
        requests.post(url, json={"text": f"```{message}```" if code else message}).raise_for_status()
    except Exception as e:
        print(f"⚠️ Slack通知エラー: {e}")

def _slack_debug(text: str, chunk=2800):
    url=os.getenv("SLACK_WEBHOOK_URL")
    if not url: print("⚠️ SLACK_WEBHOOK_URL 未設定"); return
    i=0
    while i<len(text):
        j=min(len(text), i+chunk); k=text.rfind("\n", i, j); j=k if k>i+100 else j
        blk={"type":"section","text":{"type":"mrkdwn","text":f"```{text[i:j]}```"}}
        try: requests.post(url, json={"blocks":[blk]}).raise_for_status()
        except Exception as e: print(f"⚠️ Slack通知エラー: {e}")
        i=j

def _compact_debug(fb, sb, prevG, prevD, max_rows=140):
    # ---- 列選択：既定は最小列、DEBUG_ALL_COLS=True で全列に ----
    want=["TR","EPS","REV","ROE","BETA_RAW","FCF","RS","TR_str","DIV_STREAK","DSC"]
    all_cols = _env_true("DEBUG_ALL_COLS", False)
    cols = list(fb.df_z.columns if all_cols else [c for c in want if c in fb.df_z.columns])

    # ---- 差分（入替）----
    Gp, Dp = set(prevG or []), set(prevD or [])
    g_new=[t for t in (sb.top_G or []) if t not in Gp]; g_out=[t for t in Gp if t not in (sb.top_G or [])]
    d_new=[t for t in (sb.top_D or []) if t not in Dp]; d_out=[t for t in Dp if t not in (sb.top_D or [])]

    # ---- 次点5（フラグで有無切替）----
    show_near = _env_true("DEBUG_NEAR5", True)
    gs = getattr(fb,"g_score",None); ds = getattr(fb,"d_score_all",None)
    gs = gs.sort_values(ascending=False) if show_near and hasattr(gs,"sort_values") else None
    ds = ds.sort_values(ascending=False) if show_near and hasattr(ds,"sort_values") else None
    g_miss = ([t for t in gs.index if t not in (sb.top_G or [])][:5]) if gs is not None else []
    d_excl = set((sb.top_G or [])+(sb.top_D or []))
    d_miss = ([t for t in ds.index if t not in d_excl][:5]) if ds is not None else []

    # ---- 行選択：既定は入替+採用+次点、DEBUG_ALL_ROWS=True で全銘柄 ----
    all_rows = _env_true("DEBUG_ALL_ROWS", False)
    focus = list(fb.df_z.index) if all_rows else sorted(set(g_new+g_out+d_new+d_out+(sb.top_G or [])+(sb.top_D or [])+g_miss+d_miss))
    focus = focus[:max_rows]

    # ---- ヘッダ（フィルター条件を明示）----
    def _fmt_near(lbl, ser, lst):
        if ser is None: return f"{lbl}: off"
        parts=[]
        for t in lst:
            x=ser.get(t, float("nan"))
            parts.append(f"{t}:{x:.3f}" if pd.notna(x) else f"{t}:nan")
        return f"{lbl}: "+(", ".join(parts) if parts else "-")
    head=[f"G new/out: {len(g_new)}/{len(g_out)}  D new/out: {len(d_new)}/{len(d_out)}",
          _fmt_near("G near5", gs, g_miss),
          _fmt_near("D near5", ds, d_miss),
          f"Filters: G pre_mask=['trend_template'], D pre_filter={{'beta_max': {D_BETA_MAX}}}",
          f"Cols={'ALL' if all_cols else 'MIN'}  Rows={'ALL' if all_rows else 'SUBSET'}"]

    # ---- テーブル ----
    if fb.df_z.empty or not cols:
        tbl="(df_z or columns not available)"
    else:
        idx=[t for t in focus if t in fb.df_z.index]
        tbl=fb.df_z.loc[idx, cols].round(3).to_string(max_rows=None, max_cols=None)

    # ---- 欠損ログ（フラグで有無切替）----
    miss_txt=""
    if _env_true("DEBUG_MISSING_LOGS", False):
        miss=getattr(fb,"missing_logs",None)
        if miss is not None and not miss.empty:
            miss_txt="\nMissing data (head)\n"+miss.head(10).to_string(index=False)

    return "\n".join(head+["\nChanged/Selected (+ Near Miss)", tbl])+miss_txt

def _disjoint_keepG(top_G, top_D, poolD):
    """
    Gに含まれる銘柄をDから除去し、DはpoolD（次点）で補充する。
    - 引数:
        top_G: List[str]  … G最終12銘柄
        top_D: List[str]  … D最終13銘柄（重複を含む可能性あり）
        poolD: List[str]  … D候補の順位リスト（top_Dを含む上位拡張）
    - 戻り値: (top_G, top_D_disjoint)
    - 挙動:
        1) DにG重複があれば順に置換
        2) 置換候補は poolD から、既使用(G∪D)を避けて前から採用
        3) 補充分が尽きた場合は元の銘柄を残す（安全フォールバック）
    """
    used, D = set(top_G), list(top_D)
    i = 0
    for j, t in enumerate(D):
        if t in used:
            while i < len(poolD) and (poolD[i] in used or poolD[i] in D):
                i += 1
            if i < len(poolD):
                D[j] = poolD[i]; used.add(D[j]); i += 1
    return top_G, D


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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pick, sumn, latest, aliases = self._pick_row, self._sum_last_n, self._latest, self._CF_ALIASES

        def one(t: str):
            try:
                tk = yf.Ticker(t)  # ★ セッションは渡さない（YFがcurl_cffiで管理）
                qcf = tk.quarterly_cashflow
                cfo_q, capex_q = pick(qcf, aliases["cfo"]), pick(qcf, aliases["capex"])
                fcf_q = pick(qcf, ["Free Cash Flow","FreeCashFlow","Free cash flow"])
                cfo, capex, fcf = sumn(cfo_q,4), sumn(capex_q,4), sumn(fcf_q,4)
                if any(v is None for v in (cfo, capex, fcf)):
                    acf = tk.cashflow
                    if cfo   is None: cfo   = latest(pick(acf, aliases["cfo"]))
                    if capex is None: capex = latest(pick(acf, aliases["capex"]))
                    if fcf   is None: fcf   = latest(pick(acf, ["Free Cash Flow","FreeCashFlow","Free cash flow"]))
            except Exception as e:
                print(f"[warn] yf financials error: {t}: {e}"); cfo=capex=fcf=None
            n=np.nan
            return {"ticker":t,
                    "cfo_ttm_yf":   n if cfo   is None else cfo,
                    "capex_ttm_yf": n if capex is None else capex,
                    "fcf_ttm_yf_direct": n if fcf is None else fcf}

        rows, mw = [], int(os.getenv("FIN_THREADS","8"))
        with ThreadPoolExecutor(max_workers=mw) as ex:
            for f in as_completed(ex.submit(one,t) for t in tickers): rows.append(f.result())
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
        yf_df = self.fetch_cfo_capex_ttm_yf(tickers)
        T.log("financials (yf) done")
        miss_mask = yf_df[["cfo_ttm_yf","capex_ttm_yf","fcf_ttm_yf_direct"]].isna().any(axis=1)
        need = yf_df.index[miss_mask].tolist(); print(f"[T] yf financials missing: {len(need)} {need[:10]}{'...' if len(need)>10 else ''}")
        if need:
            fh_df = self.fetch_cfo_capex_ttm_finnhub(need, api_key=finnhub_api_key)
            df = yf_df.join(fh_df, how="left")
            for col_yf, col_fh in [("cfo_ttm_yf","cfo_ttm_fh"),("capex_ttm_yf","capex_ttm_fh")]:
                df[col_yf] = df[col_yf].fillna(df[col_fh])
            print("[T] financials (finnhub) done (fallback only)")
        else:
            df = yf_df.assign(cfo_ttm_fh=np.nan, capex_ttm_fh=np.nan)
            print("[T] financials (finnhub) skipped (no missing)")
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
        T.log("price cap filter done (CAND_PRICE_MAX)")
        tickers = sorted(set(self.exist + cand_f))
        T.log(f"universe prepared: unique={len(tickers)} bench={self.bench}")
        data = yf.download(tickers + [self.bench], period="600d", auto_adjust=True, progress=False)
        T.log("yf.download done")
        px, spx = data["Close"], data["Close"][self.bench]
        clip_days = int(os.getenv("PRICE_CLIP_DAYS", "0"))   # 0なら無効（既定）
        if clip_days > 0:
            px  = px.tail(clip_days + 1)
            spx = spx.tail(clip_days + 1)
            print(f"[T] price window clipped by env: {len(px)} rows (PRICE_CLIP_DAYS={clip_days})")
        else:
            print(f"[T] price window clip skipped; rows={len(px)}")
        tickers_bulk, info = yf.Tickers(" ".join(tickers)), {}
        for t in tickers:
            try: info[t] = tickers_bulk.tickers[t].info
            except Exception as e: print(f"{t}: info fetch failed ({e})"); info[t] = {}
        eps_df = self._build_eps_df(tickers, tickers_bulk, info)
        fcf_df = self.compute_fcf_with_fallback(tickers, finnhub_api_key=self.api_key)
        T.log("eps/fcf prep done")
        returns = px[tickers].pct_change()
        T.log("price prep/returns done")
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
        d_score = d_score_all.drop(top_G, errors='ignore')
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
        self.g_formatters = self.d_formatters = {}
        # 低スコア（GSC+DSC）Top10 表示/送信用
        self.low10_table = None

    # --- 表示（元 display_results のロジックそのまま） ---
    def display_results(self, *, exist, bench, df_z, g_score, d_score_all,
                        init_G, init_D, top_G, top_D, **kwargs):
        pd.set_option('display.float_format','{:.3f}'.format)
        print("📈 ファクター分散最適化の結果")
        if self.miss_df is not None and not self.miss_df.empty:
            print("Missing Data:")
            print(self.miss_df.to_string(index=False))

        # ---- 表示用：Changes/Near-Miss のスコア源を“最終集計”に統一するプロキシ ----
        try:
            sc = getattr(self, "_sc", None)
            agg_G = getattr(sc, "_agg_G", None)
            agg_D = getattr(sc, "_agg_D", None)
        except Exception:
            sc = agg_G = agg_D = None
        class _SeriesProxy:
            __slots__ = ("primary", "fallback")
            def __init__(self, primary, fallback): self.primary, self.fallback = primary, fallback
            def get(self, key, default=None):
                try:
                    v = self.primary.get(key) if hasattr(self.primary, "get") else None
                    if v is not None and not (isinstance(v, float) and v != v):
                        return v
                except Exception:
                    pass
                try:
                    return self.fallback.get(key) if hasattr(self.fallback, "get") else default
                except Exception:
                    return default
        g_score = _SeriesProxy(agg_G, g_score)
        d_score_all = _SeriesProxy(agg_D, d_score_all)
        near_G = getattr(sc, "_near_G", []) if sc else []
        near_D = getattr(sc, "_near_D", []) if sc else []

        extra_G = [t for t in init_G if t not in top_G][:5]; G_UNI = top_G + extra_G
        gsc_series = pd.Series({t: g_score.get(t) for t in G_UNI}, name='GSC')
        self.g_table = pd.concat([df_z.loc[G_UNI,['GRW','MOM','TRD','VOL']], gsc_series], axis=1)
        self.g_table.index = [t + ("⭐️" if t in top_G else "") for t in G_UNI]
        self.g_formatters = {col:"{:.2f}".format for col in ['GRW','MOM','TRD','VOL']}; self.g_formatters['GSC'] = "{:.3f}".format
        self.g_title = (f"[G枠 / {N_G} / {_fmt_w(g_weights)} / corrM={corrM} / "
                        f"LB={DRRS_G['lookback']} nPC={DRRS_G['n_pc']} γ={DRRS_G['gamma']} λ={DRRS_G['lam']} η={DRRS_G['eta']} shrink={DRRS_SHRINK}]")
        if near_G:
            add = [t for t in near_G if t not in set(G_UNI)][:5]
            if add:
                near_tbl = pd.concat([df_z.loc[add,['GRW','MOM','TRD','VOL']], pd.Series({t: g_score.get(t) for t in add}, name='GSC')], axis=1)
                self.g_table = pd.concat([self.g_table, near_tbl], axis=0)
        print(self.g_title); print(self.g_table.to_string(formatters=self.g_formatters))

        extra_D = [t for t in init_D if t not in top_D][:5]; D_UNI = top_D + extra_D
        cols_D = ['QAL','YLD','VOL','TRD']; d_disp = pd.DataFrame(index=D_UNI)
        d_disp['QAL'], d_disp['YLD'], d_disp['VOL'], d_disp['TRD'] = df_z.loc[D_UNI,'D_QAL'], df_z.loc[D_UNI,'D_YLD'], df_z.loc[D_UNI,'D_VOL_RAW'], df_z.loc[D_UNI,'D_TRD']
        dsc_series = pd.Series({t: d_score_all.get(t) for t in D_UNI}, name='DSC')
        self.d_table = pd.concat([d_disp, dsc_series], axis=1); self.d_table.index = [t + ("⭐️" if t in top_D else "") for t in D_UNI]
        self.d_formatters = {col:"{:.2f}".format for col in cols_D}; self.d_formatters['DSC']="{:.3f}".format
        import scorer
        dw_eff = scorer.D_WEIGHTS_EFF
        self.d_title = (f"[D枠 / {N_D} / {_fmt_w(dw_eff)} / corrM={corrM} / "
                        f"LB={DRRS_D['lookback']} nPC={DRRS_D['n_pc']} γ={DRRS_D['gamma']} λ={DRRS_D['lam']} μ={CROSS_MU_GD} η={DRRS_D['eta']} shrink={DRRS_SHRINK}]")
        if near_D:
            add = [t for t in near_D if t not in set(D_UNI)][:5]
            if add:
                d_disp2 = pd.DataFrame(index=add)
                d_disp2['QAL'], d_disp2['YLD'], d_disp2['VOL'], d_disp2['TRD'] = df_z.loc[add,'D_QAL'], df_z.loc[add,'D_YLD'], df_z.loc[add,'D_VOL_RAW'], df_z.loc[add,'D_TRD']
                near_tbl = pd.concat([d_disp2, pd.Series({t: d_score_all.get(t) for t in add}, name='DSC')], axis=1)
                self.d_table = pd.concat([self.d_table, near_tbl], axis=0)
        print(self.d_title); print(self.d_table.to_string(formatters=self.d_formatters))

        # === Changes（IN の GSC/DSC を表示。OUT は銘柄名のみ） ===
        in_list = sorted(set(list(top_G)+list(top_D)) - set(exist))
        out_list = sorted(set(exist) - set(list(top_G)+list(top_D)))

        self.io_table = pd.DataFrame({
            'IN': pd.Series(in_list),
            '/ OUT': pd.Series(out_list)
        })
        g_list = [f"{g_score.get(t):.3f}" if pd.notna(g_score.get(t)) else '—' for t in out_list]
        d_list = [f"{d_score_all.get(t):.3f}" if pd.notna(d_score_all.get(t)) else '—' for t in out_list]
        self.io_table['GSC'] = pd.Series(g_list)
        self.io_table['DSC'] = pd.Series(d_list)

        print("Changes:")
        print(self.io_table.to_string(index=False))

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
            divy = ttm_div_yield_portfolio(ticks); metrics[name] = {'RET':ann_ret,'VOL':ann_vol,'SHP':sharpe,'MDD':drawdown,'RAWρ':RAW_rho,'RESIDρ':RESID_rho,'DIVY':divy}
        df_metrics = pd.DataFrame(metrics).T; df_metrics_pct = df_metrics.copy(); self.df_metrics = df_metrics
        for col in ['RET','VOL','MDD','DIVY']: df_metrics_pct[col] = df_metrics_pct[col]*100
        cols_order = ['RET','VOL','SHP','MDD','RAWρ','RESIDρ','DIVY']; df_metrics_pct = df_metrics_pct.reindex(columns=cols_order)
        def _fmt_row(s):
            return pd.Series({'RET':f"{s['RET']:.1f}%",'VOL':f"{s['VOL']:.1f}%",'SHP':f"{s['SHP']:.1f}",'MDD':f"{s['MDD']:.1f}%",'RAWρ':(f"{s['RAWρ']:.2f}" if pd.notna(s['RAWρ']) else "NaN"),'RESIDρ':(f"{s['RESIDρ']:.2f}" if pd.notna(s['RESIDρ']) else "NaN"),'DIVY':f"{s['DIVY']:.1f}%"})
        self.df_metrics_fmt = df_metrics_pct.apply(_fmt_row, axis=1); print("Performance Comparison:"); print(self.df_metrics_fmt.to_string())
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
        def _filter_suffix_from(spec: dict, group: str) -> str:
            g = spec.get(group, {})
            parts = [str(m) for m in g.get("pre_mask", [])]
            for k, v in (g.get("pre_filter", {}) or {}).items():
                base, op = (k[:-4], "<") if k.endswith("_max") else ((k[:-4], ">") if k.endswith("_min") else (k, "="))
                name = {"beta": "β"}.get(base, base)
                try: val = f"{float(v):g}"
                except: val = str(v)
                parts.append(f"{name}{op}{val}")
            return "" if not parts else " / filter:" + " & ".join(parts)
        def _inject_filter_suffix(title: str, group: str) -> str:
            suf = _filter_suffix_from(FILTER_SPEC, group)
            return f"{title[:-1]}{suf}]" if suf and title.endswith("]") else (title + suf)
        def _blk(title, tbl, fmt=None, drop=()):
            if tbl is None or getattr(tbl,'empty',False): return f"{title}\n(選定なし)\n"
            if drop and hasattr(tbl,'columns'):
                keep = [c for c in tbl.columns if c not in drop]
                tbl, fmt = tbl[keep], {k:v for k,v in (fmt or {}).items() if k in keep}
            return f"{title}\n```{tbl.to_string(formatters=fmt)}```\n"

        g_title = _inject_filter_suffix(self.g_title, "G")
        d_title = _inject_filter_suffix(self.d_title, "D")
        message  = "📈 ファクター分散最適化の結果\n"
        if self.miss_df is not None and not self.miss_df.empty:
            message += "Missing Data\n```" + self.miss_df.to_string(index=False) + "```\n"
        message += _blk(g_title, self.g_table, self.g_formatters, drop=("TRD",))
        message += _blk(d_title, self.d_table, self.d_formatters)
        message += "Changes\n" + ("(変更なし)\n" if self.io_table is None or getattr(self.io_table,'empty',False) else f"```{self.io_table.to_string(index=False)}```\n")
        message += "Performance Comparison:\n```" + self.df_metrics_fmt.to_string() + "```"
        if self.low10_table is not None:
            message += "\nLow Score Candidates (GSC+DSC bottom 10)\n```" + self.low10_table.to_string() + "```\n"
        if self.debug and self.debug_table is not None:
            message += "\nDebug Data\n```" + self.debug_table.to_string() + "```"
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
            out.append(f"{t}{' 🔥' if (br or pb) else ''}")
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


# ===== パイプライン可視化：G/D共通フロー（出力は不変） ==============================

def io_build_input_bundle() -> InputBundle:
    """
    既存の『データ取得→前処理』を実行し、InputBundle を返す。
    処理内容・列名・丸め・例外・ログ文言は現行どおり（変更禁止）。
    """
    inp = Input(cand=cand, exist=exist, bench=bench,
                price_max=CAND_PRICE_MAX, finnhub_api_key=FINNHUB_API_KEY)
    state = inp.prepare_data()
    return InputBundle(
        cand=state["cand"], tickers=state["tickers"], bench=bench,
        data=state["data"], px=state["px"], spx=state["spx"],
        tickers_bulk=state["tickers_bulk"], info=state["info"],
        eps_df=state["eps_df"], fcf_df=state["fcf_df"],
        returns=state["returns"]
    )

def run_group(sc: Scorer, group: str, inb: InputBundle, cfg: PipelineConfig,
              n_target: int, prev_json_path: str) -> tuple[list, float, float, float]:
    """
    G/Dを同一手順で処理：採点→フィルター→選定（相関低減込み）。
    戻り値：(pick, avg_res_corr, sum_score, objective)
    JSON保存は既存フォーマット（キー名・丸め桁・順序）を踏襲。
    """
    sc.cfg = cfg

    if hasattr(sc, "score_build_features"):
        feat = sc.score_build_features(inb)
        if not hasattr(sc, "_feat_logged"):
            T.log("features built (scorer)")
            sc._feat_logged = True
        agg = sc.score_aggregate(feat, group, cfg) if hasattr(sc, "score_aggregate") else feat
    else:
        fb = sc.aggregate_scores(inb, cfg)
        if not hasattr(sc, "_feat_logged"):
            T.log("features built (scorer)")
            sc._feat_logged = True
        sc._feat = fb
        agg = fb.g_score if group == "G" else fb.d_score_all
        if group == "D" and hasattr(fb, "df"):
            agg = agg[fb.df['BETA'] < D_BETA_MAX]

    if hasattr(sc, "filter_candidates"):
        mask = sc.filter_candidates(inb, agg, group, cfg)
        agg = agg[mask]

    selector = Selector()
    prev = _load_prev(prev_json_path)
    if hasattr(sc, "select_diversified"):
        pick, avg_r, sum_sc, obj = sc.select_diversified(
            agg, group, cfg, n_target,
            selector=selector, prev_tickers=prev,
            corrM=cfg.drrs.corrM, shrink=cfg.drrs.shrink,
            cross_mu=cfg.drrs.cross_mu_gd
        )
    else:
        if group == "G":
            init = agg.nlargest(min(cfg.drrs.corrM, len(agg))).index.tolist()
            res = selector.select_bucket_drrs(
                returns_df=inb.returns, score_ser=agg, pool_tickers=init, k=n_target,
                n_pc=cfg.drrs.G.get("n_pc", 3), gamma=cfg.drrs.G.get("gamma", 1.2),
                lam=cfg.drrs.G.get("lam", 0.68), eta=cfg.drrs.G.get("eta", 0.8),
                lookback=cfg.drrs.G.get("lookback", 252), prev_tickers=prev,
                shrink=cfg.drrs.shrink, g_fixed_tickers=None, mu=0.0
            )
        else:
            init = agg.nlargest(min(cfg.drrs.corrM, len(agg))).index.tolist()
            g_fixed = getattr(sc, "_top_G", None)
            res = selector.select_bucket_drrs(
                returns_df=inb.returns, score_ser=agg, pool_tickers=init, k=n_target,
                n_pc=cfg.drrs.D.get("n_pc", 4), gamma=cfg.drrs.D.get("gamma", 0.8),
                lam=cfg.drrs.D.get("lam", 0.85), eta=cfg.drrs.D.get("eta", 0.5),
                lookback=cfg.drrs.D.get("lookback", 504), prev_tickers=prev,
                shrink=cfg.drrs.shrink, g_fixed_tickers=g_fixed,
                mu=cfg.drrs.cross_mu_gd
            )
        pick = res["tickers"]; avg_r = res["avg_res_corr"]
        sum_sc = res["sum_score"]; obj = res["objective"]
        if group == "D":
            _, pick = _disjoint_keepG(getattr(sc, "_top_G", []), pick, init)
            T.log("selection finalized (G/D)")
    # --- Near-Miss: 惜しくも選ばれなかった上位5を保持（Slack表示用） ---
    # 5) Near-Miss と最終集計Seriesを保持（表示専用。計算へ影響なし）
    try:
        pool = agg.drop(index=[t for t in pick if t in agg.index], errors="ignore")
        near5 = list(pool.sort_values(ascending=False).head(5).index)
        setattr(sc, f"_near_{group}", near5)
        setattr(sc, f"_agg_{group}", agg)
    except Exception:
        pass

    _save_sel(prev_json_path, pick, avg_r, sum_sc, obj)
    if group == "D":
        T.log("save done")
    if group == "G":
        sc._top_G = pick
    return pick, avg_r, sum_sc, obj

def run_pipeline() -> SelectionBundle:
    """
    G/D共通フローの入口。I/Oはここだけで実施し、計算はScorerに委譲。
    Slack文言・丸め・順序は既存の Output を用いて変更しない。
    """
    inb = io_build_input_bundle()
    cfg = PipelineConfig(
        weights=WeightsConfig(g=g_weights, d=D_weights),
        drrs=DRRSParams(corrM=corrM, shrink=DRRS_SHRINK,
                         G=DRRS_G, D=DRRS_D, cross_mu_gd=CROSS_MU_GD),
        price_max=CAND_PRICE_MAX
    )
    sc = Scorer()
    top_G, avgG, sumG, objG = run_group(sc, "G", inb, cfg, N_G, G_PREV_JSON)
    poolG = list(getattr(sc, "_agg_G", pd.Series(dtype=float)).sort_values(ascending=False).index)
    alpha = Scorer.spx_to_alpha(inb.spx)
    sectors = {t: (inb.info.get(t, {}).get("sector") or "U") for t in poolG}
    scores = {t: Scorer.g_score.get(t, 0.0) for t in poolG}
    top_G = Scorer.pick_top_softcap(scores, sectors, N=N_G, cap=2, alpha=alpha, hard=5)
    sc._top_G = top_G
    base = sum(Scorer.g_score.get(t,0.0) for t in poolG[:N_G])
    effs = sum(Scorer.g_score.get(t,0.0) for t in top_G)
    print(f"[soft_cap2] score_cost={(base-effs)/max(1e-9,abs(base)):.2%}, alpha={alpha:.3f}")
    top_D, avgD, sumD, objD = run_group(sc, "D", inb, cfg, N_D, D_PREV_JSON)
    fb = getattr(sc, "_feat", None)
    near_G = getattr(sc, "_near_G", [])
    selected12 = list(top_G)
    df = fb.df if fb is not None else pd.DataFrame()
    guni = _infer_g_universe(df, selected12, near_G)
    try:
        fire_recent = [t for t in guni
                       if (str(df.at[t, "G_BREAKOUT_recent_5d"]) == "True") or
                          (str(df.at[t, "G_PULLBACK_recent_5d"]) == "True")]
    except Exception:
        fire_recent = []
    lines = [
        "【G枠レポート｜週次モニタ（直近5営業日）】",
        "【凡例】🔥=直近5営業日内に「ブレイクアウト確定」または「押し目反発」を検知",
        f"選定12: {', '.join(_fmt_with_fire_mark(selected12, df))}" if selected12 else "選定12: なし",
        f"次点5: {', '.join(_fmt_with_fire_mark(near_G, df))}" if near_G else "次点5: なし",
    ]
    if fire_recent:
        fire_list = ", ".join([_label_recent_event(t, df) for t in fire_recent])
        lines.append(f"過去5営業日の検知: {fire_list}")
    else:
        lines.append("過去5営業日の検知: なし")
    try:
        webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
        if webhook:
            requests.post(webhook, json={"text": "\n".join(lines)}, timeout=10)
    except Exception:
        pass

    out = Output(debug=debug_mode)
    # 表示側から選定時の集計へアクセスできるように保持（表示専用・副作用なし）
    try: out._sc = sc
    except Exception: pass
    if hasattr(sc, "_feat"):
        try:
            out.miss_df = sc._feat.missing_logs
            out.display_results(
                exist=exist, bench=bench, df_z=sc._feat.df_z,
                g_score=sc._feat.g_score, d_score_all=sc._feat.d_score_all,
                init_G=top_G, init_D=top_D, top_G=top_G, top_D=top_D
            )
        except Exception:
            pass
    out.notify_slack()
    sb = SelectionBundle(
        resG={"tickers": top_G, "avg_res_corr": avgG,
              "sum_score": sumG, "objective": objG},
        resD={"tickers": top_D, "avg_res_corr": avgD,
              "sum_score": sumD, "objective": objD},
        top_G=top_G, top_D=top_D, init_G=top_G, init_D=top_D
    )

    if debug_mode:
        prevG, prevD = _load_prev(G_PREV_JSON), _load_prev(D_PREV_JSON)
        try:
            _slack_debug(_compact_debug(fb, sb, prevG, prevD))
        except Exception as e:
            print(f"[debug skipped] {e}")

    return sb

if __name__ == "__main__":
    run_pipeline()
