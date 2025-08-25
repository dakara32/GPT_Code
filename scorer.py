# scorer.py
# =============================================================================
# Scorer: ファクター/指標の生成と合成スコア算出を担う純粋層
#
# 【このファイルだけ読めば分かるポイント】
# - 入力(InputBundle)は「価格/出来高/ベンチ/基本情報/EPS/FCF/リターン」を含むDTO
# - 出力(FeatureBundle)は「raw特徴量 df」「標準化 df_z」「G/D スコア」「欠損ログ」
# - 重み等のコンフィグ(PipelineConfig)は外部から渡されればそれを優先
#   渡されない場合は本ファイルの DEFAULT_CONFIG を使用（単体実行OK）
# - 旧カラム名は Scorer 内で自動リネームして受け入れ（後方互換）
#   例) eps_ttm -> EPS_TTM, eps_q_recent -> EPS_Q_LastQ, fcf_ttm -> FCF_TTM
#
# 【I/O契約（Scorerが参照するInputBundleフィールド）】
#   - cand: List[str]    … 候補銘柄（単体実行では未使用）
#   - tickers: List[str] … 対象銘柄リスト
#   - bench: str         … ベンチマークティッカー（例 '^GSPC'）
#   - data: pd.DataFrame … yfinance download結果 ('Close','Volume' 等の階層列)
#   - px: pd.DataFrame   … data['Close'] 相当（終値）
#   - spx: pd.Series     … ベンチマークの終値
#   - tickers_bulk: object         … yfinance.Tickers
#   - info: Dict[str, dict]        … yfinance info per ticker
#   - eps_df: pd.DataFrame         … 必須列: EPS_TTM, EPS_Q_LastQ（旧名も可）
#   - fcf_df: pd.DataFrame         … 必須列: FCF_TTM（旧名も可）
#   - returns: pd.DataFrame        … px[tickers].pct_change() 相当
#
# ※入出力の形式・例外文言は既存実装を変えません（安全な短縮のみ）
# =============================================================================

import os, requests
import numpy as np, pandas as pd, yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy.stats import zscore

# ---- 簡易ユーティリティ（安全な短縮のみ） -----------------------------------
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

# ---- DTO & Config（単体実行でも動くよう最小限を内包） -----------------------
@dataclass(frozen=True)
class InputBundle:
    cand: List[str]
    tickers: List[str]
    bench: str
    data: pd.DataFrame
    px: pd.DataFrame
    spx: pd.Series
    tickers_bulk: object
    info: Dict[str, dict]
    eps_df: pd.DataFrame
    fcf_df: pd.DataFrame
    returns: pd.DataFrame

@dataclass(frozen=True)
class FeatureBundle:
    df: pd.DataFrame
    df_z: pd.DataFrame
    g_score: pd.Series
    d_score_all: pd.Series
    missing_logs: pd.DataFrame

@dataclass(frozen=True)
class WeightsConfig:
    g: Dict[str, float]
    d: Dict[str, float]

@dataclass(frozen=True)
class DRRSParams:
    corrM: int
    shrink: float
    G: Dict[str, float]
    D: Dict[str, float]
    cross_mu_gd: float

@dataclass(frozen=True)
class PipelineConfig:
    weights: WeightsConfig
    drrs: DRRSParams
    price_max: float

# ---- デフォルト設定（外部が渡せばそれを優先） -------------------------------
DEFAULT_G_WEIGHTS = {'GRW':0.40,'MOM':0.40,'TRD':0.00,'VOL':-0.20}
DEFAULT_D_WEIGHTS = {'QAL':0.15,'YLD':0.15,'VOL':-0.45,'TRD':0.25}
DEFAULT_DRRS = DRRSParams(
    corrM=45, shrink=0.10,
    G=dict(lookback=252, n_pc=3, gamma=1.2, lam=0.68, eta=0.8),
    D=dict(lookback=504, n_pc=4, gamma=0.8, lam=0.85, eta=0.5),
    cross_mu_gd=0.40
)
DEFAULT_CONFIG = PipelineConfig(
    weights=WeightsConfig(g=DEFAULT_G_WEIGHTS, d=DEFAULT_D_WEIGHTS),
    drrs=DEFAULT_DRRS,
    price_max=400.0
)

# Slack/Debug 環境に合わせて既存変数を許容（未定義なら無視）
debug_mode = bool(os.environ.get("SCORER_DEBUG", "0") == "1")
D_BETA_MAX = float(os.environ.get("D_BETA_MAX", "0.8"))
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
D_WEIGHTS_EFF = None  # 出力表示互換のため

# ---- Scorer 本体 -------------------------------------------------------------
class Scorer:
    """
    - factor.py からは `aggregate_scores(ib, cfg)` を呼ぶだけでOK。
    - cfg を省略 or None の場合は DEFAULT_CONFIG を使用。
    - 旧カラム名を自動リネームして新スキーマに吸収します。
    """

    # === 先頭で旧→新カラム名マップ（移行用） ===
    EPS_RENAME = {"eps_ttm":"EPS_TTM", "eps_q_recent":"EPS_Q_LastQ"}
    FCF_RENAME = {"fcf_ttm":"FCF_TTM"}

    # === スキーマ簡易チェック（最低限） ===
    @staticmethod
    def _validate_ib_for_scorer(ib: InputBundle):
        must_attrs = ["tickers","bench","data","px","spx","tickers_bulk","info","eps_df","fcf_df","returns"]
        miss = [a for a in must_attrs if not hasattr(ib, a) or getattr(ib, a) is None]
        if miss: raise ValueError(f"InputBundle is missing required attributes for Scorer: {miss}")

        # 後方互換のため、まず rename を試みる
        if any(c in ib.eps_df.columns for c in Scorer.EPS_RENAME.keys()):
            ib.eps_df.rename(columns=Scorer.EPS_RENAME, inplace=True)
        if any(c in ib.fcf_df.columns for c in Scorer.FCF_RENAME.keys()):
            ib.fcf_df.rename(columns=Scorer.FCF_RENAME, inplace=True)

        # 必須列の存在確認
        need_eps = {"EPS_TTM","EPS_Q_LastQ"}
        need_fcf = {"FCF_TTM"}
        if not need_eps.issubset(set(ib.eps_df.columns)):
            raise ValueError(f"eps_df must contain columns {need_eps} (accepts old names via auto-rename). Got: {list(ib.eps_df.columns)}")
        if not need_fcf.issubset(set(ib.fcf_df.columns)):
            raise ValueError(f"fcf_df must contain columns {need_fcf} (accepts old names via auto-rename). Got: {list(ib.fcf_df.columns)}")

    # ----（Scorer専用）テクニカル・指標系 ----
    @staticmethod
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

    @staticmethod
    def rs(s, b):
        n, nb = len(s), len(b)
        if n<60 or nb<60: return np.nan
        L12 = 252 if n>=252 and nb>=252 else min(n,nb)-1; L1 = 22 if n>=22 and nb>=22 else max(5, min(n,nb)//3)
        r12, r1, br12, br1 = s.iloc[-1]/s.iloc[-L12]-1, s.iloc[-1]/s.iloc[-L1]-1, b.iloc[-1]/b.iloc[-L12]-1, b.iloc[-1]/b.iloc[-L1]-1
        return (r12 - br12)*0.7 + (r1 - br1)*0.3

    @staticmethod
    def tr_str(s):
        if len(s)<50: return np.nan
        return s.iloc[-1]/s.rolling(50).mean().iloc[-1] - 1

    @staticmethod
    def rs_line_slope(s: pd.Series, b: pd.Series, win: int) -> float:
        r = (s/b).dropna()
        if len(r)<win: return np.nan
        y, x = np.log(r.iloc[-win:]), np.arange(win, dtype=float)
        try: return float(np.polyfit(x, y, 1)[0])
        except Exception: return np.nan

    @staticmethod
    def ev_fallback(info_t: dict, tk: yf.Ticker) -> float:
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

    @staticmethod
    def dividend_status(ticker: str) -> str:
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

    @staticmethod
    def div_streak(t):
        try:
            divs = yf.Ticker(t).dividends.dropna(); ann = divs.groupby(divs.index.year).sum(); ann = ann[ann.index<pd.Timestamp.today().year]
            years, streak = sorted(ann.index), 0
            for i in range(len(years)-1,0,-1):
                if ann[years[i]] > ann[years[i-1]]: streak += 1
                else: break
            return streak
        except Exception: return 0

    @staticmethod
    def fetch_finnhub_metrics(symbol):
        if not FINNHUB_API_KEY: return {}
        url, params = "https://finnhub.io/api/v1/stock/metric", {"symbol":symbol,"metric":"all","token":FINNHUB_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10); r.raise_for_status(); m = r.json().get("metric",{})
            return {'EPS':m.get('epsGrowthTTMYoy'),'REV':m.get('revenueGrowthTTMYoy'),'ROE':m.get('roeTTM'),'BETA':m.get('beta'),'DIV':m.get('dividendYieldIndicatedAnnual'),'FCF':(m.get('freeCashFlowTTM')/m.get('enterpriseValue')) if m.get('freeCashFlowTTM') and m.get('enterpriseValue') else None}
        except Exception: return {}

    @staticmethod
    def calc_beta(series: pd.Series, market: pd.Series, lookback=252):
        r, m = series.pct_change().dropna(), market.pct_change().dropna()
        n = min(len(r), len(m), lookback)
        if n<60: return np.nan
        r, m = r.iloc[-n:], m.iloc[-n:]; cov, var = np.cov(r, m)[0,1], np.var(m)
        return np.nan if var==0 else cov/var

    # ---- スコア集計（DTO/Configを受け取り、FeatureBundleを返す） ----
    def aggregate_scores(self, ib: InputBundle, cfg: Optional[PipelineConfig]=None) -> FeatureBundle:
        self._validate_ib_for_scorer(ib)
        cfg = cfg or DEFAULT_CONFIG  # 外部優先だが、無ければデフォルトで自走

        px, spx, tickers = ib.px, ib.spx, ib.tickers
        tickers_bulk, info, eps_df, fcf_df = ib.tickers_bulk, ib.info, ib.eps_df, ib.fcf_df

        df, missing_logs = pd.DataFrame(index=tickers), []
        for t in tickers:
            d, s = info[t], px[t]; ev = self.ev_fallback(d, tickers_bulk.tickers[t])
            # --- 基本特徴 ---
            df.loc[t,'TR']   = self.trend(s)
            df.loc[t,'EPS']  = eps_df.loc[t,'EPS_TTM'] if t in eps_df.index else np.nan
            df.loc[t,'REV']  = d.get('revenueGrowth',np.nan)
            df.loc[t,'ROE']  = d.get('returnOnEquity',np.nan)
            df.loc[t,'BETA'] = self.calc_beta(s, spx, lookback=252)

            # --- 配当（欠損補完含む） ---
            div = d.get('dividendYield') if d.get('dividendYield') is not None else d.get('trailingAnnualDividendYield')
            if div is None or pd.isna(div):
                try:
                    divs = yf.Ticker(t).dividends
                    if divs is not None and not divs.empty:
                        last_close = s.iloc[-1]; div_1y = divs[divs.index >= (divs.index.max() - pd.Timedelta(days=365))].sum()
                        if last_close and last_close>0: div = float(div_1y/last_close)
                except Exception: pass
            df.loc[t,'DIV'] = 0.0 if (div is None or pd.isna(div)) else float(div)

            # --- FCF/EV ---
            fcf_val = fcf_df.loc[t,'FCF_TTM'] if t in fcf_df.index else np.nan
            df.loc[t,'FCF'] = (fcf_val/ev) if (pd.notna(fcf_val) and pd.notna(ev) and ev>0) else np.nan

            # --- モメンタム・ボラ関連 ---
            df.loc[t,'RS'], df.loc[t,'TR_str'] = self.rs(s, spx), self.tr_str(s)
            r, rm = s.pct_change().dropna(), spx.pct_change().dropna()
            n = int(min(len(r), len(rm)))

            DOWNSIDE_DEV = np.nan
            if n>=60:
                r6 = r.iloc[-min(len(r),126):]; neg = r6[r6<0]
                if len(neg)>=10: DOWNSIDE_DEV = float(neg.std(ddof=0)*np.sqrt(252))
            df.loc[t,'DOWNSIDE_DEV'] = DOWNSIDE_DEV

            MDD_1Y = np.nan
            try:
                w = s.iloc[-min(len(s),252):].dropna()
                if len(w)>=30:
                    roll_max = w.cummax(); MDD_1Y = float((w/roll_max - 1.0).min())
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

            # --- 長期移動平均/位置 ---
            sma200 = s.rolling(200).mean(); df.loc[t,'EXT_200'] = np.nan
            if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1]!=0: df.loc[t,'EXT_200'] = abs(float(s.iloc[-1]/sma200.iloc[-1]-1.0))

            # --- 配当の詳細系 ---
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
                so = d.get('sharesOutstanding',None)
                if so and pd.notna(DIV_TTM_PS) and pd.notna(fcf_val) and fcf_val!=0:
                    DIV_FCF_COVER = float((fcf_val)/(DIV_TTM_PS*float(so)))
            except Exception: pass
            df.loc[t,'DIV_TTM_PS'], df.loc[t,'DIV_VAR5'], df.loc[t,'DIV_YOY'], df.loc[t,'DIV_FCF_COVER'] = DIV_TTM_PS, DIV_VAR5, DIV_YOY, DIV_FCF_COVER

            # --- 財務安定性 ---
            df.loc[t,'DEBT2EQ'], df.loc[t,'CURR_RATIO'] = d.get('debtToEquity',np.nan), d.get('currentRatio',np.nan)

            # --- EPS 変動 ---
            EPS_VAR_8Q = np.nan
            try:
                qe, so = tickers_bulk.tickers[t].quarterly_earnings, d.get('sharesOutstanding',None)
                if qe is not None and not qe.empty and so:
                    eps_q = (qe['Earnings'].dropna().astype(float)/float(so)).replace([np.inf,-np.inf],np.nan)
                    if len(eps_q)>=4: EPS_VAR_8Q = float(eps_q.iloc[-min(8,len(eps_q)):].std(ddof=1))
            except Exception: pass
            df.loc[t,'EPS_VAR_8Q'] = EPS_VAR_8Q

            # --- サイズ/流動性 ---
            df.loc[t,'MARKET_CAP'] = d.get('marketCap',np.nan); adv60 = np.nan
            try:
                vol_series = ib.data['Volume'][t].dropna()
                if len(vol_series)>=5 and len(s)==len(vol_series):
                    dv = (vol_series*s).rolling(60).mean(); adv60 = float(dv.iloc[-1])
            except Exception: pass
            df.loc[t,'ADV60_USD'] = adv60

            # --- 売上/利益の加速度等 ---
            REV_Q_YOY=EPS_Q_YOY=REV_YOY_ACC=REV_YOY_VAR=np.nan
            REV_ANNUAL_STREAK = np.nan
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
                        # NEW: 年次の持続性（直近から遡って前年比プラスが何年連続か、四半期4本揃う完全年のみ）
                        try:
                            g = rev.groupby(rev.index.year)
                            ann_sum, cnt = g.sum(), g.count()
                            ann_sum = ann_sum[cnt >= 4]
                            if len(ann_sum) >= 3:
                                yoy = ann_sum.pct_change().dropna()
                                streak = 0
                                for v in yoy.iloc[::-1]:
                                    if pd.isna(v) or v <= 0:
                                        break
                                    streak += 1
                                REV_ANNUAL_STREAK = float(streak)
                        except Exception:
                            pass
                    if 'Earnings' in qe.columns and so:
                        eps_series = (qe['Earnings'].dropna().astype(float)/float(so)).replace([np.inf,-np.inf],np.nan)
                        if len(eps_series)>=5 and pd.notna(eps_series.iloc[-5]) and eps_series.iloc[-5]!=0:
                            EPS_Q_YOY = _safe_div(eps_series.iloc[-1]-eps_series.iloc[-5], eps_series.iloc[-5])
            except Exception: pass
            df.loc[t,'REV_Q_YOY'], df.loc[t,'EPS_Q_YOY'], df.loc[t,'REV_YOY_ACC'], df.loc[t,'REV_YOY_VAR'] = REV_Q_YOY, EPS_Q_YOY, REV_YOY_ACC, REV_YOY_VAR
            df.loc[t,'REV_ANN_STREAK'] = REV_ANNUAL_STREAK

            # --- Rule of 40 や周辺 ---
            total_rev_ttm = d.get('totalRevenue',np.nan)
            FCF_MGN = _safe_div(fcf_val, total_rev_ttm)
            df.loc[t,'FCF_MGN'] = FCF_MGN
            rule40 = np.nan
            try:
                r = df.loc[t,'REV']; rule40 = (r if pd.notna(r) else np.nan) + (FCF_MGN if pd.notna(FCF_MGN) else np.nan)
            except Exception: pass
            df.loc[t,'RULE40'] = rule40

            # --- トレンド補助 ---
            sma50  = s.rolling(50).mean()
            sma150 = s.rolling(150).mean()
            sma200 = s.rolling(200).mean()
            p = _safe_last(s)

            df.loc[t,'MA50_OVER_150'] = (
                _safe_last(sma50)/_safe_last(sma150) - 1
                if pd.notna(_safe_last(sma50)) and pd.notna(_safe_last(sma150)) and _safe_last(sma150)!=0 else np.nan
            )
            df.loc[t,'MA150_OVER_200'] = (
                _safe_last(sma150)/_safe_last(sma200) - 1
                if pd.notna(_safe_last(sma150)) and pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan
            )

            lo52 = s[-252:].min() if len(s)>=252 else s.min()
            df.loc[t,'P_OVER_LOW52'] = (p/lo52 - 1) if (lo52 and lo52>0 and pd.notna(p)) else np.nan

            df.loc[t,'MA200_SLOPE_1M'] = np.nan
            if len(sma200.dropna()) >= 21:
                cur200 = _safe_last(sma200)
                old2001 = float(sma200.iloc[-21])
                if old2001:
                    df.loc[t,'MA200_SLOPE_1M'] = cur200/old2001 - 1

            df.loc[t,'P_OVER_150'] = p/_safe_last(sma150)-1 if pd.notna(_safe_last(sma150)) and _safe_last(sma150)!=0 else np.nan
            df.loc[t,'P_OVER_200'] = p/_safe_last(sma200)-1 if pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan
            df.loc[t,'MA50_OVER_200'] = _safe_last(sma50)/_safe_last(sma200)-1 if pd.notna(_safe_last(sma50)) and pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan
            df.loc[t,'MA200_SLOPE_5M'] = np.nan
            if len(sma200.dropna())>=105:
                cur200, old200 = _safe_last(sma200), float(sma200.iloc[-105])
                if old200 and old200!=0: df.loc[t,'MA200_SLOPE_5M'] = cur200/old200 - 1
            # NEW: 200日線が連続で上向きの「日数」
            df.loc[t,'MA200_UP_STREAK_D'] = np.nan
            try:
                s200 = sma200.dropna()
                if len(s200) >= 2:
                    diff200 = s200.diff()
                    up = 0
                    for v in diff200.iloc[::-1]:
                        if pd.isna(v) or v <= 0:
                            break
                        up += 1
                    df.loc[t,'MA200_UP_STREAK_D'] = float(up)
            except Exception:
                pass
            df.loc[t,'LOW52PCT25_EXCESS'] = np.nan if (lo52 is None or lo52<=0 or pd.isna(p)) else (p/(lo52*1.25)-1)
            hi52 = s[-252:].max() if len(s)>=252 else s.max(); df.loc[t,'NEAR_52W_HIGH'] = np.nan
            if hi52 and hi52>0 and pd.notna(p):
                d_hi = (p/hi52)-1.0; df.loc[t,'NEAR_52W_HIGH'] = -abs(min(0.0, d_hi))
            df.loc[t,'RS_SLOPE_6W'] = self.rs_line_slope(s, ib.spx, 30)
            df.loc[t,'RS_SLOPE_13W'] = self.rs_line_slope(s, ib.spx, 65)

            df.loc[t,'DIV_STREAK'] = self.div_streak(t)

            # --- 欠損メモ ---
            fin_cols = ['REV','ROE','BETA','DIV','FCF']
            need_finnhub = [col for col in fin_cols if pd.isna(df.loc[t,col])]
            if need_finnhub:
                fin_data = self.fetch_finnhub_metrics(t)
                for col in need_finnhub:
                    val = fin_data.get(col)
                    if val is not None and not pd.isna(val): df.loc[t,col] = val
            for col in fin_cols + ['EPS','RS','TR_str','DIV_STREAK']:
                if pd.isna(df.loc[t,col]):
                    if col=='DIV':
                        status = self.dividend_status(t)
                        if status!='none_confident': missing_logs.append({'Ticker':t,'Column':col,'Status':status})
                    else:
                        missing_logs.append({'Ticker':t,'Column':col})

        # === Z化と合成 ===
        for col in ['ROE','FCF','REV','EPS']: df[f'{col}_W'] = winsorize_s(df[col], 0.02)

        df_z = pd.DataFrame(index=df.index)
        for col in ['EPS','REV','ROE','FCF','RS','TR_str','BETA','DIV','DIV_STREAK']: df_z[col] = robust_z(df[col])
        df_z['REV'], df_z['EPS'], df_z['TR'] = robust_z(df['REV_W']), robust_z(df['EPS_W']), robust_z(df['TR'])
        for col in ['P_OVER_150','P_OVER_200','MA50_OVER_200','MA200_SLOPE_5M','LOW52PCT25_EXCESS','NEAR_52W_HIGH','RS_SLOPE_6W','RS_SLOPE_13W','MA200_UP_STREAK_D']: df_z[col] = robust_z(df[col])
        for col in ['REV_Q_YOY','EPS_Q_YOY','REV_YOY_ACC','REV_YOY_VAR','FCF_MGN','RULE40','REV_ANN_STREAK']: df_z[col] = robust_z(df[col])
        for col in ['DOWNSIDE_DEV','MDD_1Y','RESID_VOL','DOWN_OUTPERF','EXT_200','DIV_TTM_PS','DIV_VAR5','DIV_YOY','DIV_FCF_COVER','DEBT2EQ','CURR_RATIO','EPS_VAR_8Q','MARKET_CAP','ADV60_USD']: df_z[col] = robust_z(df[col])

        df_z['SIZE'], df_z['LIQ'] = robust_z(np.log1p(df['MARKET_CAP'])), robust_z(np.log1p(df['ADV60_USD']))
        df_z['QUALITY_F'] = robust_z(0.6*df['FCF_W'] + 0.4*df['ROE_W']).clip(-3.0,3.0)
        df_z['YIELD_F']   = 0.3*df_z['DIV'] + 0.7*df_z['DIV_STREAK']
        df_z['GROWTH_F']  = robust_z(
              0.30*df_z['REV']
            + 0.20*df_z['EPS_Q_YOY']
            + 0.15*df_z['REV_Q_YOY']
            + 0.15*df_z['REV_YOY_ACC']
            + 0.10*df_z['RULE40']
            + 0.10*df_z['FCF_MGN']
            + 0.10*df_z['REV_ANN_STREAK']
            - 0.05*df_z['REV_YOY_VAR']
        ).clip(-3.0,3.0)
        df_z['MOM_F'] = robust_z(
              0.40*df_z['RS']
            + 0.15*df_z['TR_str']
            + 0.15*df_z['RS_SLOPE_6W']
            + 0.15*df_z['RS_SLOPE_13W']
            + 0.10*df_z['MA200_SLOPE_5M']
            + 0.10*df_z['MA200_UP_STREAK_D']
        ).clip(-3.0,3.0)
        df_z['VOL'] = robust_z(df['BETA'])
        df_z.rename(columns={'GROWTH_F':'GRW','MOM_F':'MOM','QUALITY_F':'QAL','YIELD_F':'YLD'}, inplace=True)
        df_z['TRD'] = 0.0  # TRDはスコア寄与から外し、テンプレ判定はフィルタで行う（列は表示互換のため残す）
        if 'BETA' not in df_z.columns: df_z['BETA'] = robust_z(df['BETA'])

        df_z['D_VOL_RAW'] = robust_z(0.40*df_z['DOWNSIDE_DEV'] + 0.22*df_z['RESID_VOL'] + 0.18*df_z['MDD_1Y'] - 0.10*df_z['DOWN_OUTPERF'] - 0.05*df_z['EXT_200'] - 0.08*df_z['SIZE'] - 0.10*df_z['LIQ'] + 0.10*df_z['BETA'])
        df_z['D_QAL']     = robust_z(0.35*df_z['QAL'] + 0.20*df_z['FCF'] + 0.15*df_z['CURR_RATIO'] - 0.15*df_z['DEBT2EQ'] - 0.15*df_z['EPS_VAR_8Q'])
        df_z['D_YLD']     = robust_z(0.45*df_z['DIV'] + 0.25*df_z['DIV_STREAK'] + 0.20*df_z['DIV_FCF_COVER'] - 0.10*df_z['DIV_VAR5'])
        df_z['D_TRD']     = robust_z(0.40*df_z.get('MA200_SLOPE_5M',0) - 0.30*df_z.get('EXT_200',0) + 0.15*df_z.get('NEAR_52W_HIGH',0) + 0.15*df_z['TR'])

        # --- 重みは cfg を優先（外部があればそれを使用） ---
        # ① 全銘柄で G/D スコアを算出（unmasked）
        g_score_all = df_z.mul(pd.Series(cfg.weights.g)).sum(axis=1)

        d_comp = pd.concat({
            'QAL': df_z['D_QAL'],
            'YLD': df_z['D_YLD'],
            'VOL': df_z['D_VOL_RAW'],
            'TRD': df_z['D_TRD']
        }, axis=1)
        dw = pd.Series(cfg.weights.d, dtype=float).reindex(['QAL','YLD','VOL','TRD']).fillna(0.0)
        globals()['D_WEIGHTS_EFF'] = dw.copy()
        d_score_all = d_comp.mul(dw, axis=1).sum(axis=1)

        # ② テンプレ判定（既存ロジックそのまま）
        def _trend_template_pass(row, rs_alpha_thresh=0.10):
            c1 = (row.get('P_OVER_150', np.nan) > 0) and (row.get('P_OVER_200', np.nan) > 0)
            c2 = (row.get('MA150_OVER_200', np.nan) > 0)
            c3 = (row.get('MA200_SLOPE_1M', np.nan) > 0)
            c4 = (row.get('MA50_OVER_150', np.nan) > 0) and (row.get('MA50_OVER_200', np.nan) > 0)
            c5 = (row.get('TR_str', np.nan) > 0)
            c6 = (row.get('P_OVER_LOW52', np.nan) >= 0.30)
            c7 = (row.get('NEAR_52W_HIGH', np.nan) >= -0.25)
            c8 = (row.get('RS', np.nan) >= 0.10)
            return bool(c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8)

        mask = df.apply(_trend_template_pass, axis=1).fillna(False)

        if not bool(mask.any()):
            mask = (
                (df.get('P_OVER_LOW52', np.nan) >= 0.25) &
                (df.get('NEAR_52W_HIGH', np.nan) >= -0.30) &
                (df.get('RS', np.nan) >= 0.08) &
                (df.get('MA200_SLOPE_1M', np.nan) > 0) &
                (df.get('P_OVER_150', np.nan) > 0) & (df.get('P_OVER_200', np.nan) > 0) &
                (df.get('MA150_OVER_200', np.nan) > 0) &
                (df.get('MA50_OVER_150', np.nan) > 0) & (df.get('MA50_OVER_200', np.nan) > 0) &
                (df.get('TR_str', np.nan) > 0)
            ).fillna(False)

        df['TT_PASS'] = mask

        # ③ 採用用は mask、表示/分析用は列で全銘柄保存
        g_score = g_score_all.loc[mask]
        df_z['GSC'] = g_score_all
        df_z['DSC'] = d_score_all

        # --- D枠のβフィルタ（採用可視化） ---
        # removed: beta gate moved to factor.py
        # d_pass_beta = df['BETA'] < D_BETA_MAX
        # df_z['D_PASS_BETA'] = d_pass_beta.astype(float)
        # df_z['DSC_DPASS']   = d_score_all.where(d_pass_beta, np.nan)

        if debug_mode:
            eps = 0.1
            _base = d_comp.mul(dw, axis=1).sum(axis=1)
            _test = d_comp.assign(VOL=d_comp['VOL'] + eps).mul(dw, axis=1).sum(axis=1)
            print("VOL増→d_score低下の比率:", ((_test <= _base) | _test.isna() | _base.isna()).mean())
        try:
            df = _apply_growth_entry_flags(df, ib, self, win_breakout=5, win_pullback=5)
        except Exception:
            pass

        return FeatureBundle(
            df=df,
            df_z=df_z,
            g_score=g_score,
            d_score_all=d_score_all,
            missing_logs=pd.DataFrame(missing_logs)
        )


def _apply_growth_entry_flags(feature_df, bundle, self_obj, win_breakout=5, win_pullback=5):
    """
    G枠ユニバースに対し、ブレイクアウト確定/押し目反発の「直近N営業日内の発火」を判定し、
    次の列を feature_df に追加する（index=ticker）。
      - G_BREAKOUT_recent_5d : bool
      - G_BREAKOUT_last_date : str "YYYY-MM-DD"
      - G_PULLBACK_recent_5d : bool
      - G_PULLBACK_last_date : str "YYYY-MM-DD"
      - G_PIVOT_price        : float
    失敗しても例外は握り潰し、既存処理を阻害しない。
    """
    try:
        px   = bundle.px                      # 終値 DataFrame
        hi   = bundle.data['High']
        lo   = bundle.data['Low']
        vol  = bundle.data['Volume']
        bench= bundle.spx                     # ベンチマーク Series

        # Gユニバース推定：self.g_universe 優先 → feature_df['group']=='G' → 全銘柄
        g_universe = getattr(self_obj, "g_universe", None)
        if g_universe is None:
            try:
                g_universe = feature_df.index[feature_df['group'].astype(str).str.upper().eq('G')].tolist()
            except Exception:
                g_universe = list(feature_df.index)
        if not g_universe:
            return feature_df

        # 指標
        ema21 = px[g_universe].ewm(span=21, adjust=False).mean()
        atr20 = (hi[g_universe] - lo[g_universe]).rolling(20).mean()
        vol20 = vol[g_universe].rolling(20).mean()
        vol50 = vol[g_universe].rolling(50).mean()

        # 汎用ピボット：直近65営業日の高値（当日除外）
        pivot_price = hi[g_universe].rolling(65).max().shift(1)

        # 相対力：年内高値更新
        bench_aligned = bench.reindex(px.index).ffill()
        rs = px[g_universe].div(bench_aligned, axis=0)
        rs_high = rs.rolling(252).max().shift(1)

        # ブレイクアウト「発生日」：条件立ち上がり
        breakout_today = (px[g_universe] > pivot_price) \
                         & (vol[g_universe] >= 1.5 * vol50) & (rs > rs_high)
        breakout_event = breakout_today & ~breakout_today.shift(1).fillna(False)

        # 押し目反発「発生日」：EMA21帯×出来高ドライアップ×前日高値越え×終値EMA21上
        near_ema21_band = px[g_universe].between(ema21 - atr20, ema21 + atr20)
        volume_dryup = (vol20 / vol50) <= 1.0
        pullback_bounce_confirmed = (px[g_universe] > hi[g_universe].shift(1)) & (px[g_universe] > ema21)
        pullback_today = near_ema21_band & volume_dryup & pullback_bounce_confirmed
        pullback_event = pullback_today & ~pullback_today.shift(1).fillna(False)

        # 直近N営業日内の発火 / 最終発生日
        rows = []
        for t in g_universe:
            def _recent_and_date(s, win):
                sw = s[t].iloc[-win:]
                if sw.any():
                    d = sw[sw].index[-1]
                    return True, d.strftime("%Y-%m-%d")
                return False, ""
            br_recent, br_date = _recent_and_date(breakout_event, win_breakout)
            pb_recent, pb_date = _recent_and_date(pullback_event, win_pullback)
            rows.append((t, {
                "G_BREAKOUT_recent_5d": br_recent,
                "G_BREAKOUT_last_date": br_date,
                "G_PULLBACK_recent_5d": pb_recent,
                "G_PULLBACK_last_date": pb_date,
                "G_PIVOT_price": float(pivot_price[t].iloc[-1]) if t in pivot_price.columns else float('nan'),
            }))
        flags = pd.DataFrame({k: v for k, v in rows}).T

        # 列を作成・上書き
        cols = ["G_BREAKOUT_recent_5d","G_BREAKOUT_last_date","G_PULLBACK_recent_5d","G_PULLBACK_last_date","G_PIVOT_price"]
        for c in cols:
            if c not in feature_df.columns:
                feature_df[c] = np.nan
        feature_df.loc[flags.index, flags.columns] = flags

    except Exception:
        pass
    return feature_df


# === 単体実行サンプル（最小） =================================================
# 使い方: `python scorer.py` を叩くと current_tickers.csv / candidate_tickers.csv を読み込み
# デフォルト設定でスコア計算だけを行い、上位を表示します。（セレクタは別ファイル）
if __name__ == "__main__":
    # 入力CSV（存在しなければサンプルで停止）
    cur_csv, cand_csv = "current_tickers.csv", "candidate_tickers.csv"
    if not (os.path.exists(cur_csv) and os.path.exists(cand_csv)):
        raise SystemExit("current_tickers.csv / candidate_tickers.csv が見つかりません。単体実行の事前準備をしてください。")

    exist = pd.read_csv(cur_csv, header=None)[0].tolist()
    cand  = pd.read_csv(cand_csv, header=None)[0].tolist()
    bench = '^GSPC'

    # 価格上限フィルタ（単体実行でも体験を合わせる）
    info_cand = yf.Tickers(" ".join(cand))
    last_prices = {}
    for t in cand:
        try: last_prices[t] = info_cand.tickers[t].fast_info.get("lastPrice", np.inf)
        except Exception as e:
            print(f"{t}: price fetch failed ({e})"); last_prices[t] = np.inf
    cand_f = [t for t,p in last_prices.items() if p <= DEFAULT_CONFIG.price_max]

    # ユニバース作成 & データ取得
    tickers = sorted(set(exist + cand_f))
    data = yf.download(tickers + [bench], period="600d", auto_adjust=True, progress=False)
    px, spx = data["Close"], data["Close"][bench]
    tickers_bulk, info = yf.Tickers(" ".join(tickers)), {}
    for t in tickers:
        try: info[t] = tickers_bulk.tickers[t].info
        except Exception as e: print(f"{t}: info fetch failed ({e})"); info[t] = {}

    # EPS/FCF（簡易）：既存パイプライン互換の最低限のみ
    # EPS_TTM / EPS_Q_LastQ
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
        eps_rows.append({"ticker":t,"EPS_TTM":eps_ttm,"EPS_Q_LastQ":eps_q})
    eps_df = pd.DataFrame(eps_rows).set_index("ticker").copy()
    # FCF_TTM（最低限：yfinanceから CFO/Capex の TTM を近似）
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
    CF_ALIASES = {"cfo":["Operating Cash Flow","Total Cash From Operating Activities"], "capex":["Capital Expenditure","Capital Expenditures"]}
    rows=[]
    for t in tickers:
        tk = yf.Ticker(t); qcf = tk.quarterly_cashflow
        cfo_q = _pick_row(qcf, CF_ALIASES["cfo"]); capex_q = _pick_row(qcf, CF_ALIASES["capex"])
        cfo_ttm = _sum_last_n(cfo_q,4); capex_ttm = _sum_last_n(capex_q,4)
        if cfo_ttm is None or capex_ttm is None:
            acf = tk.cashflow
            if cfo_ttm is None:
                s = _pick_row(acf, CF_ALIASES["cfo"]); cfo_ttm = float(s.iloc[0]) if s is not None and not s.dropna().empty else np.nan
            if capex_ttm is None:
                s = _pick_row(acf, CF_ALIASES["capex"]); capex_ttm = float(s.iloc[0]) if s is not None and not s.dropna().empty else np.nan
        fcf_ttm = np.nan
        if pd.notna(cfo_ttm) and pd.notna(capex_ttm): fcf_ttm = float(cfo_ttm) - float(abs(capex_ttm))
        rows.append({"ticker":t,"FCF_TTM":fcf_ttm})
    fcf_df = pd.DataFrame(rows).set_index("ticker")

    returns = px[tickers].pct_change()
    ib = InputBundle(cand=cand_f, tickers=tickers, bench=bench, data=data, px=px, spx=spx,
                     tickers_bulk=tickers_bulk, info=info, eps_df=eps_df, fcf_df=fcf_df, returns=returns)

    scorer = Scorer()
    fb = scorer.aggregate_scores(ib, DEFAULT_CONFIG)

    # 単体実行の出力（簡易）
    print("=== G-score (top 15) ===")
    print(fb.g_score.sort_values(ascending=False).head(15).round(3).to_string())
    print("\n=== D-score (top 15) ===")
    print(fb.d_score_all.sort_values(ascending=False).head(15).round(3).to_string())
    print("\n=== Missing Data Logs ===")
    if fb.missing_logs is not None and not fb.missing_logs.empty:
        print(fb.missing_logs.to_string(index=False))
    else:
        print("(none)")
