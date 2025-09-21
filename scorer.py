# scorer.py
# kawatest
# =============================================================================
# Scorer: ファクター/指標の生成と合成スコア算出を担う純粋層
#
# 【このファイルだけ読めば分かるポイント】
# - 入力(InputBundle)は「価格/出来高/ベンチ/基本情報/EPS/FCF/リターン」を含むDTO
# - 出力(FeatureBundle)は「raw特徴量 df」「標準化 df_z」「G/D スコア」「欠損ログ」
# - 重み等のコンフィグ(PipelineConfig)は factor から渡す（cfg 必須）
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

import json, logging, os, requests, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any, TYPE_CHECKING
from scipy.stats import zscore
from datetime import datetime as _dt

if TYPE_CHECKING:
    from factor import PipelineConfig  # type: ignore  # 実行時importなし（循環回避）

logger = logging.getLogger(__name__)


def _log(stage, msg):
    try:
        print(f"[DBG][{_dt.utcnow().isoformat(timespec='seconds')}Z][{stage}] {msg}")
    except Exception:
        print(f"[DBG][{stage}] {msg}")


# ---- Dividend Helpers -------------------------------------------------------
def _last_close(t, price_map=None):
    if price_map and (c := price_map.get(t)) is not None: return float(c)
    try:
        h = yf.Ticker(t).history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) if len(h) else np.nan
    except Exception:
        return np.nan

def _ttm_div_sum(t, lookback_days=400):
    try:
        div = yf.Ticker(t).dividends
        if div is None or len(div) == 0: return 0.0
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
        ttm = float(div[div.index.tz_localize(None) >= cutoff].sum())
        return ttm if ttm > 0 else float(div.tail(4).sum())
    except Exception:
        return 0.0

def ttm_div_yield_portfolio(tickers, price_map=None):
    ys = [(lambda c, s: (s/c) if (np.isfinite(c) and c>0 and s>0) else 0.0)(_last_close(t, price_map), _ttm_div_sum(t)) for t in tickers]
    return float(np.mean(ys)) if ys else 0.0

# ---- 簡易ユーティリティ（安全な短縮のみ） -----------------------------------
def _as_numeric_series(s: pd.Series) -> pd.Series:
    """Series を float dtype に強制変換し、index を保持する。"""
    if s is None:
        return pd.Series(dtype=float)
    v = pd.to_numeric(s, errors="coerce")
    return pd.Series(v.values, index=getattr(s, "index", None), dtype=float, name=getattr(s, "name", None))


def _scalar(x):
    """
    入力を安全に float スカラへ変換する。

    許容する入力パターン:
      - pandas.Series: 非NaNの最後の値を採用
      - numpy スカラ/配列: 最後の要素を採用
      - その他の数値っぽい値: float へ変換

    変換できない場合は np.nan を返す。
    """

    if x is None:
        return np.nan

    # pandas.Series の場合は非NaNの最後の値を採用
    if isinstance(x, pd.Series):
        s = pd.to_numeric(x, errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else np.nan

    # numpy スカラ (item() を持つ) ※文字列は除外
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return float(x.item())
        except Exception:
            pass

    # 配列様のオブジェクト
    try:
        arr = np.asarray(x, dtype=float).ravel()
        return float(arr[-1]) if arr.size else np.nan
    except Exception:
        pass

    # 最後に素直に float 変換を試す
    try:
        return float(x)
    except Exception:
        return np.nan


def winsorize_s(s: pd.Series, p=0.02):
    if s is None or s.dropna().empty: return s
    lo, hi = np.nanpercentile(s.astype(float), [100*p, 100*(1-p)]); return s.clip(lo, hi)

def robust_z(s: pd.Series, p=0.02):
    s2 = winsorize_s(s,p); return np.nan_to_num(zscore(s2.fillna(s2.mean())))

def robust_z_keepnan(s: pd.Series) -> pd.Series:
    """robust_z variant that preserves NaNs and falls back to rank-z when needed."""
    if s is None:
        return pd.Series(dtype=float)
    v = pd.to_numeric(s, errors="coerce")
    m = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - m))
    z = (v - m) / (1.4826 * mad + 1e-9)
    if np.nanstd(z) < 1e-9:
        r = v.rank(method="average", na_option="keep")
        z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-9)
    return pd.Series(z, index=v.index, dtype=float)


def _safe_div(a, b):
    try: return np.nan if (b is None or float(b)==0 or pd.isna(b)) else float(a)/float(b)
    except Exception: return np.nan

def _safe_last(series: pd.Series, default=np.nan):
    try: return float(series.iloc[-1])
    except Exception: return default


def _ensure_series(x):
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.dropna()
    if isinstance(x, (list, tuple)):
        if len(x) and isinstance(x[0], (tuple, list)) and len(x[0]) == 2:
            dt = pd.to_datetime([d for d, _ in x], errors="coerce")
            v = pd.to_numeric([_v for _, _v in x], errors="coerce")
            return pd.Series(v, index=dt).dropna()
        return pd.Series(pd.to_numeric(list(x), errors="coerce")).dropna()
    try:
        return pd.Series(x).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _to_quarterly(s: pd.Series) -> pd.Series:
    if s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return s
    return s.resample("Q").last().dropna()


def _ttm_yoy_from_quarterly(qs: pd.Series) -> pd.Series:
    if qs is None or qs.empty:
        return pd.Series(dtype=float)
    ttm = qs.rolling(4, min_periods=2).sum()
    yoy = ttm.pct_change(4)
    return yoy




class Scorer:
    """
    - factor.py からは `aggregate_scores(ib, cfg)` を呼ぶだけでOK。
    - cfg は必須（factor.PipelineConfig を渡す）。
    - 旧カラム名を自動リネームして新スキーマに吸収します。
    """

    # === 先頭で旧→新カラム名マップ（移行用） ===
    EPS_RENAME = {"eps_ttm":"EPS_TTM", "eps_q_recent":"EPS_Q_LastQ"}
    FCF_RENAME = {"fcf_ttm":"FCF_TTM"}

    # === スキーマ簡易チェック（最低限） ===
    @staticmethod
    def _validate_ib_for_scorer(ib: Any):
        miss = [a for a in ["tickers","bench","data","px","spx","tickers_bulk","info","eps_df","fcf_df","returns"] if not hasattr(ib,a) or getattr(ib,a) is None]
        if miss: raise ValueError(f"InputBundle is missing required attributes for Scorer: {miss}")
        if any(c in ib.eps_df.columns for c in Scorer.EPS_RENAME): ib.eps_df.rename(columns=Scorer.EPS_RENAME, inplace=True)
        if any(c in ib.fcf_df.columns for c in Scorer.FCF_RENAME): ib.fcf_df.rename(columns=Scorer.FCF_RENAME, inplace=True)
        need_eps, need_fcf = {"EPS_TTM","EPS_Q_LastQ"},{"FCF_TTM"}
        if not need_eps.issubset(ib.eps_df.columns): raise ValueError(f"eps_df must contain columns {need_eps} (accepts old names via auto-rename). Got: {list(ib.eps_df.columns)}")
        if not need_fcf.issubset(ib.fcf_df.columns): raise ValueError(f"fcf_df must contain columns {need_fcf} (accepts old names via auto-rename). Got: {list(ib.fcf_df.columns)}")

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
        if s is None:
            return np.nan
        s = s.ffill(limit=2).dropna()
        if len(s) < 50:
            return np.nan
        ma50 = s.rolling(50, min_periods=50).mean()
        last_ma = ma50.iloc[-1]
        last_px = s.iloc[-1]
        return float(last_px/last_ma - 1.0) if pd.notna(last_ma) and pd.notna(last_px) else np.nan

    @staticmethod
    def rs_line_slope(s: pd.Series, b: pd.Series, win: int) -> float:
        r = (s/b).dropna()
        if len(r) < win: return np.nan
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
            if (a is not None and not a.empty and "Stock Splits" in a.columns and a["Stock Splits"].abs().sum()>0): return "none_confident"
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
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key: return {}
        url, params = "https://finnhub.io/api/v1/stock/metric", {"symbol":symbol,"metric":"all","token":api_key}
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

    @staticmethod
    def spx_to_alpha(spx: pd.Series, bands=(0.03,0.10), w=(0.6,0.4),
                     span=5, q=(0.20,0.40), alphas=(0.05,0.08,0.10)) -> float:
        """
        S&P500指数のみから擬似breadthを作り、履歴分位でαを段階決定。
        bands=(±3%, ±10%), w=(50DMA,200DMA), 分位q=(20%,40%), alphas=(低,中,高)
        """
        ma50, ma200 = spx.rolling(50).mean(), spx.rolling(200).mean()
        b50, b200 = ((spx/ma50 - 1)+bands[0])/(2*bands[0]), ((spx/ma200 - 1)+bands[1])/(2*bands[1])
        hist = (w[0]*b50 + w[1]*b200).clip(0,1).ewm(span=span).mean()
        b, (lo, mid) = float(hist.iloc[-1]), (float(hist.quantile(q[0])), float(hist.quantile(q[1])))
        return alphas[0] if b < lo else alphas[1] if b < mid else alphas[2]

    @staticmethod
    def soft_cap_effective_scores(scores: pd.Series|dict, sectors: dict, cap=2, alpha=0.08) -> pd.Series:
        """
        同一セクターcap超過（3本目以降）に α×段階減点を課した“有効スコア”Seriesを返す。
        戻り値は降順ソート済み。
        """
        s = pd.Series(scores, dtype=float); order = s.sort_values(ascending=False).index
        cnt, pen = {}, {}
        for t in order:
            sec = sectors.get(t, "U"); cnt[sec] = cnt.get(sec,0) + 1; pen[t] = alpha*max(0, cnt[sec]-cap)
        return (s - pd.Series(pen)).sort_values(ascending=False)

    @staticmethod
    def pick_top_softcap(scores: pd.Series|dict, sectors: dict, N: int, cap=2, alpha=0.08, hard: int|None=5) -> list[str]:
        """
        soft-cap適用後の上位Nティッカーを返す。hard>0なら非常用ハード上限で同一セクター超過を間引く（既定=5）。
        """
        eff = Scorer.soft_cap_effective_scores(scores, sectors, cap, alpha)
        if not hard:
            return list(eff.head(N).index)
        pick, used = [], {}
        for t in eff.index:
            s = sectors.get(t, "U")
            if used.get(s,0) < hard:
                pick.append(t); used[s] = used.get(s,0) + 1
            if len(pick) == N: break
        return pick

    @staticmethod
    def trend_template_breadth_series(px: pd.DataFrame, spx: pd.Series, win_days: int | None = None) -> pd.Series:
        """
        各営業日の trend_template 合格本数（合格“本数”=C）を返す。
        - px: 列=ticker（ベンチは含めない）
        - spx: ベンチマーク Series（px.index に整列）
        - win_days: 末尾の計算対象営業日数（None→全体、既定600は呼び出し側指定）
        ベクトル化＆rollingのみで軽量。欠損は False 扱い。
        """
        import numpy as np, pandas as pd
        if px is None or px.empty:
            return pd.Series(dtype=int)
        px = px.dropna(how="all", axis=1)
        if win_days and win_days > 0:
            px = px.tail(win_days)
        if px.empty:
            return pd.Series(dtype=int)
        spx = spx.reindex(px.index).ffill()

        ma50  = px.rolling(50).mean()
        ma150 = px.rolling(150).mean()
        ma200 = px.rolling(200).mean()

        tt = (px > ma150)
        tt &= (px > ma200)
        tt &= (ma150 > ma200)
        tt &= (ma200 - ma200.shift(21) > 0)
        tt &= (ma50  > ma150)
        tt &= (ma50  > ma200)
        tt &= (px    > ma50)

        lo252 = px.rolling(252).min()
        hi252 = px.rolling(252).max()
        tt &= (px.divide(lo252).sub(1.0) >= 0.30)   # P_OVER_LOW52 >= 0.30
        tt &= (px >= (0.75 * hi252))                # NEAR_52W_HIGH >= -0.25

        r12  = px.divide(px.shift(252)).sub(1.0)
        br12 = spx.divide(spx.shift(252)).sub(1.0)
        r1   = px.divide(px.shift(22)).sub(1.0)
        br1  = spx.divide(spx.shift(22)).sub(1.0)
        rs   = 0.7*(r12.sub(br12, axis=0)) + 0.3*(r1.sub(br1, axis=0))
        tt &= (rs >= 0.10)

        return tt.fillna(False).sum(axis=1).astype(int)

    # ---- スコア集計（DTO/Configを受け取り、FeatureBundleを返す） ----
    def aggregate_scores(self, ib: Any, cfg):
        if cfg is None:
            raise ValueError("cfg is required; pass factor.PipelineConfig")
        self._validate_ib_for_scorer(ib)

        px, spx, tickers = ib.px, ib.spx, ib.tickers
        tickers_bulk, info, eps_df, fcf_df = ib.tickers_bulk, ib.info, ib.eps_df, ib.fcf_df

        df, missing_logs = pd.DataFrame(index=tickers), []
        df['EPS_SERIES'] = pd.Series([None] * len(df), index=df.index, dtype=object)
        debug_mode = bool(getattr(cfg, "debug_mode", False))
        eps_cols = set(getattr(eps_df, "columns", []))
        for t in tickers:
            d, s = info[t], px[t]; ev = self.ev_fallback(d, tickers_bulk.tickers[t])
            try:
                volume_series_full = ib.data['Volume'][t]
            except Exception:
                volume_series_full = None

            # --- 基本特徴 ---
            df.loc[t,'TR']   = self.trend(s)

            def _eps_value(col: str) -> float:
                if col not in eps_cols:
                    return np.nan
                try:
                    return _scalar(eps_df[col].get(t, np.nan))
                except Exception:
                    return np.nan

            df.loc[t,'EPS']  = _eps_value('EPS_TTM')
            df.loc[t,'EPS_Q'] = _eps_value('EPS_Q_LastQ')
            df.loc[t,'REV_TTM'] = _eps_value('REV_TTM')
            df.loc[t,'REV_Q']   = _eps_value('REV_Q_LastQ')
            df.loc[t,'EPS_TTM_PREV'] = _eps_value('EPS_TTM_PREV')
            df.loc[t,'REV_TTM_PREV'] = _eps_value('REV_TTM_PREV')
            df.loc[t,'EPS_Q_PREV'] = _eps_value('EPS_Q_Prev')
            df.loc[t,'REV_Q_PREV'] = _eps_value('REV_Q_Prev')
            df.loc[t,'EPS_A_LATEST'] = _eps_value('EPS_A_LATEST')
            df.loc[t,'EPS_A_PREV'] = _eps_value('EPS_A_PREV')
            df.loc[t,'REV_A_LATEST'] = _eps_value('REV_A_LATEST')
            df.loc[t,'REV_A_PREV'] = _eps_value('REV_A_PREV')
            df.loc[t,'EPS_A_CAGR3'] = _eps_value('EPS_A_CAGR3')
            df.loc[t,'REV_A_CAGR3'] = _eps_value('REV_A_CAGR3')
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
                if isinstance(volume_series_full, pd.Series):
                    vol_series = volume_series_full.reindex(s.index).dropna()
                    if len(vol_series) >= 5:
                        aligned_px = s.reindex(vol_series.index).dropna()
                        if len(aligned_px) == len(vol_series):
                            dv = (vol_series*aligned_px).rolling(60).mean()
                            if not dv.dropna().empty:
                                adv60 = float(dv.dropna().iloc[-1])
            except Exception:
                pass
            df.loc[t,'ADV60_USD'] = adv60

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

            df.loc[t,'MA50_OVER_150'] = (_safe_last(sma50)/_safe_last(sma150) - 1
                if pd.notna(_safe_last(sma50)) and pd.notna(_safe_last(sma150)) and _safe_last(sma150)!=0 else np.nan)
            df.loc[t,'MA150_OVER_200'] = (_safe_last(sma150)/_safe_last(sma200) - 1
                if pd.notna(_safe_last(sma150)) and pd.notna(_safe_last(sma200)) and _safe_last(sma200)!=0 else np.nan)

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

        def _pick_series(entry: dict, keys: list[str]):
            for k in keys:
                val = entry.get(k) if isinstance(entry, dict) else None
                if val is None:
                    continue
                try:
                    if hasattr(val, "empty") and getattr(val, "empty"):
                        continue
                except Exception:
                    pass
                if isinstance(val, (list, tuple)) and len(val) == 0:
                    continue
                return val
            return None

        def _has_sec_series(val) -> bool:
            try:
                if isinstance(val, pd.Series):
                    return not val.dropna().empty
                if isinstance(val, (list, tuple)):
                    return any(pd.notna(v) for v in val)
                return bool(val)
            except Exception:
                return False

        def _series_len(val) -> int:
            try:
                if isinstance(val, pd.Series):
                    return int(val.dropna().size)
                if isinstance(val, (list, tuple)):
                    return len(val)
                return int(bool(val))
            except Exception:
                return 0

        for t in tickers:
            try:
                d = info.get(t, {}) or {}
                rev_series = d.get("SEC_REV_Q_SERIES")
                eps_series = d.get("SEC_EPS_Q_SERIES")
                fallback_qearn = False
                try:
                    qe = tickers_bulk.tickers[t].quarterly_earnings
                    fallback_qearn = bool(qe is not None and not getattr(qe, "empty", True))
                except Exception:
                    qe = None

                r_src = _pick_series(d, ["SEC_REV_Q_SERIES", "rev_q_series_pairs", "rev_q_series"])
                e_src = _pick_series(d, ["SEC_EPS_Q_SERIES", "eps_q_series_pairs", "eps_q_series"])
                r_raw = _ensure_series(r_src)
                e_raw = _ensure_series(e_src)

                r_q = _to_quarterly(r_raw)
                e_q = _to_quarterly(e_raw)

                df.at[t, "EPS_SERIES"] = e_q

                r_yoy_ttm = _ttm_yoy_from_quarterly(r_q)
                e_yoy_ttm = _ttm_yoy_from_quarterly(e_q)

                def _q_yoy(qs):
                    return np.nan if qs is None or len(qs) < 5 else float(qs.iloc[-1] / qs.iloc[-5] - 1.0)

                rev_q_yoy = _q_yoy(r_q)
                eps_q_yoy = _q_yoy(e_q)

                def _annual_from(qs: pd.Series, yoy_ttm: pd.Series):
                    if isinstance(qs.index, pd.DatetimeIndex) and len(qs) >= 8:
                        ann = qs.groupby(qs.index.year).last().pct_change()
                        ann_dn = ann.dropna()
                        if not ann_dn.empty:
                            y = float(ann_dn.iloc[-1])
                            acc = float(ann_dn.tail(3).mean()) if ann_dn.size >= 3 else np.nan
                            var = float(ann_dn.tail(4).var()) if ann_dn.size >= 4 else np.nan
                            return y, acc, var
                    yoy_dn = yoy_ttm.dropna()
                    if yoy_dn.empty:
                        return np.nan, np.nan, np.nan
                    return (
                        float(yoy_dn.iloc[-1]),
                        float(yoy_dn.tail(3).mean() if yoy_dn.size >= 3 else np.nan),
                        float(yoy_dn.tail(4).var() if yoy_dn.size >= 4 else np.nan),
                    )

                rev_yoy, rev_acc, rev_var = _annual_from(r_q, r_yoy_ttm)
                eps_yoy, _, _ = _annual_from(e_q, e_yoy_ttm)

                def _pos_streak(s: pd.Series):
                    s = s.dropna()
                    if s.empty:
                        return np.nan
                    b = (s > 0).astype(int).to_numpy()[::-1]
                    k = 0
                    for v in b:
                        if v == 1:
                            k += 1
                        else:
                            break
                    return float(k)

                rev_ann_streak = _pos_streak(r_yoy_ttm)

                df.loc[t, "REV_Q_YOY"] = rev_q_yoy
                df.loc[t, "EPS_Q_YOY"] = eps_q_yoy
                df.loc[t, "REV_YOY"] = rev_yoy
                df.loc[t, "EPS_YOY"] = eps_yoy
                df.loc[t, "REV_YOY_ACC"] = rev_acc
                df.loc[t, "REV_YOY_VAR"] = rev_var
                df.loc[t, "REV_ANN_STREAK"] = rev_ann_streak

            except Exception as e:
                logger.warning("growth-derivatives failed: %s: %s", t, e)

        def _pct_change(new, old):
            try:
                if np.isfinite(new) and np.isfinite(old) and float(old) != 0:
                    return float((new - old) / abs(old))
            except Exception:
                pass
            return np.nan

        def _pct_series(a: pd.Series, b: pd.Series) -> list[float]:
            a_vals = pd.to_numeric(a, errors="coerce") if a is not None else pd.Series(np.nan, index=df.index)
            b_vals = pd.to_numeric(b, errors="coerce") if b is not None else pd.Series(np.nan, index=df.index)
            return [_pct_change(x, y) for x, y in zip(a_vals.reindex(df.index), b_vals.reindex(df.index))]

        def _mean_valid(vals: list[float]) -> float:
            arr = [float(v) for v in vals if np.isfinite(v)]
            return float(np.mean(arr)) if arr else np.nan

        grw_q_eps_last = _pct_series(df['EPS_Q'], df.get('EPS_Q_PREV', pd.Series(np.nan, index=df.index)))
        grw_q_rev_last = _pct_series(df['REV_Q'], df.get('REV_Q_PREV', pd.Series(np.nan, index=df.index)))
        grw_q_eps_ttm = _pct_series(df['EPS'], df.get('EPS_TTM_PREV', pd.Series(np.nan, index=df.index)))
        grw_q_rev_ttm = _pct_series(df['REV_TTM'], df.get('REV_TTM_PREV', pd.Series(np.nan, index=df.index)))

        grw_a_eps_yoy = _pct_series(df.get('EPS_A_LATEST', pd.Series(np.nan, index=df.index)), df.get('EPS_A_PREV', pd.Series(np.nan, index=df.index)))
        grw_a_rev_yoy = _pct_series(df.get('REV_A_LATEST', pd.Series(np.nan, index=df.index)), df.get('REV_A_PREV', pd.Series(np.nan, index=df.index)))
        grw_a_eps_cagr = pd.to_numeric(df.get('EPS_A_CAGR3', pd.Series(np.nan, index=df.index)), errors="coerce").reindex(df.index).tolist()
        grw_a_rev_cagr = pd.to_numeric(df.get('REV_A_CAGR3', pd.Series(np.nan, index=df.index)), errors="coerce").reindex(df.index).tolist()

        grw_q_combined = [
            _mean_valid([a, b, c, d])
            for a, b, c, d in zip(grw_q_eps_last, grw_q_rev_last, grw_q_eps_ttm, grw_q_rev_ttm)
        ]
        grw_a_combined = [
            _mean_valid([a, b, c, d])
            for a, b, c, d in zip(grw_a_eps_yoy, grw_a_rev_yoy, grw_a_eps_cagr, grw_a_rev_cagr)
        ]

        df['GRW_Q_RAW'] = pd.Series(grw_q_combined, index=df.index, dtype=float)
        df['GRW_A_RAW'] = pd.Series(grw_a_combined, index=df.index, dtype=float)

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

        if 'trend_template' not in df.columns: df['trend_template'] = df.apply(_trend_template_pass, axis=1).fillna(False)
        assert 'trend_template' in df.columns

        def _calc_eps_abs_slope(eps_series, n=12):
            try:
                if isinstance(eps_series, pd.Series):
                    series = pd.to_numeric(eps_series, errors="coerce").dropna()
                elif isinstance(eps_series, (list, tuple, np.ndarray)):
                    series = pd.Series(eps_series, dtype=float).dropna()
                else:
                    return 0.0
            except Exception:
                return 0.0

            if series.empty:
                return 0.0

            tail = series.tail(n).to_numpy(dtype=float)
            if tail.size < 2:
                return 0.0

            x = np.arange(tail.size, dtype=float)
            x = x - x.mean()
            y = tail - tail.mean()
            denom = np.dot(x, x)
            if denom == 0:
                return 0.0
            slope = float(np.dot(x, y) / denom)
            return slope

        df['EPS_ABS_SLOPE'] = df['EPS_SERIES'].apply(_calc_eps_abs_slope).astype(float)
        df.drop(columns=['EPS_SERIES'], inplace=True)

        # === Z化と合成 ===
        for col in ['ROE','FCF','REV','EPS']: df[f'{col}_W'] = winsorize_s(df[col], 0.02)

        df_z = pd.DataFrame(index=df.index)
        for col in ['EPS','REV','ROE','FCF','RS','TR_str','BETA','DIV','DIV_STREAK']: df_z[col] = robust_z(df[col])
        df_z['REV'], df_z['EPS'], df_z['TR'] = robust_z(df['REV_W']), robust_z(df['EPS_W']), robust_z(df['TR'])
        for col in ['P_OVER_150','P_OVER_200','MA200_SLOPE_5M','NEAR_52W_HIGH','RS_SLOPE_6W','RS_SLOPE_13W','MA200_UP_STREAK_D']: df_z[col] = robust_z(df[col])

        df_z['EPS_ABS_SLOPE'] = robust_z(df['EPS_ABS_SLOPE']).clip(-3.0, 3.0)

        # === Growth深掘り系（欠損保持z + RAW併載） ===
        grw_cols = ['REV_Q_YOY','EPS_Q_YOY','REV_YOY','EPS_YOY','REV_YOY_ACC','REV_YOY_VAR','FCF_MGN','RULE40','REV_ANN_STREAK']
        for col in grw_cols:
            if col in df.columns:
                raw = pd.to_numeric(df[col], errors="coerce")
                df_z[col] = robust_z_keepnan(raw)
        for k in ("TREND_SLOPE_EPS", "TREND_SLOPE_REV"):
            if k in df.columns and k not in df_z.columns:
                raw = pd.to_numeric(df[k], errors="coerce")
                df_z[k] = robust_z_keepnan(raw)
        for col in ['DOWNSIDE_DEV','MDD_1Y','RESID_VOL','DOWN_OUTPERF','EXT_200','DIV_VAR5','DIV_FCF_COVER','DEBT2EQ','CURR_RATIO','EPS_VAR_8Q','MARKET_CAP','ADV60_USD']: df_z[col] = robust_z(df[col])

        df_z['SIZE'], df_z['LIQ'] = robust_z(np.log1p(df['MARKET_CAP'])), robust_z(np.log1p(df['ADV60_USD']))
        df_z['QUALITY_F'] = robust_z(0.6*df['FCF_W'] + 0.4*df['ROE_W']).clip(-3.0,3.0)
        df_z['YIELD_F']   = 0.3*df_z['DIV'] + 0.7*df_z['DIV_STREAK']

        # EPSが赤字でもFCFが黒字なら実質黒字とみなす
        eps_pos_mask = (df['EPS'] > 0) | (df['FCF_MGN'] > 0)
        df_z['EPS_POS'] = df_z['EPS'].where(eps_pos_mask, 0.0)

        # ===== トレンドスロープ算出 =====
        def zpos(x):
            arr = robust_z(x)
            idx = getattr(x, 'index', df_z.index)
            return pd.Series(arr, index=idx).fillna(0.0)

        def relu(x):
            ser = x if isinstance(x, pd.Series) else pd.Series(x, index=df_z.index)
            return ser.clip(lower=0).fillna(0.0)

        # 売上トレンドスロープ（四半期）
        slope_rev = 0.70*zpos(df_z['REV_Q_YOY']) + 0.30*zpos(df_z['REV_YOY_ACC'])
        noise_rev = relu(robust_z(df_z['REV_YOY_VAR']) - 0.8)
        slope_rev_combo = slope_rev - 0.25*noise_rev
        df_z['TREND_SLOPE_REV'] = slope_rev_combo.clip(-3.0, 3.0)

        # EPSトレンドスロープ（四半期）
        slope_eps = (
            0.40*zpos(df_z['EPS_Q_YOY']) +
            0.20*zpos(df_z['EPS_POS']) +
            0.40*zpos(df_z['EPS_ABS_SLOPE'])
        )
        df_z['TREND_SLOPE_EPS'] = slope_eps.clip(-3.0, 3.0)

        # 年次トレンド（サブ）
        slope_rev_yr = zpos(df_z['REV_YOY'])
        slope_eps_yr = zpos(df_z.get('EPS_YOY', pd.Series(0.0, index=df.index)))
        streak_base = df['REV_ANN_STREAK'].clip(lower=0).fillna(0)
        streak_yr = streak_base / (streak_base.abs() + 1.0)
        slope_rev_yr_combo = 0.7*slope_rev_yr + 0.3*streak_yr
        df_z['TREND_SLOPE_REV_YR'] = slope_rev_yr_combo.clip(-3.0, 3.0)
        df_z['TREND_SLOPE_EPS_YR'] = slope_eps_yr.clip(-3.0, 3.0)

        grw_q_z = robust_z_keepnan(df['GRW_Q_RAW']).clip(-3.0, 3.0)
        grw_a_z = robust_z_keepnan(df['GRW_A_RAW']).clip(-3.0, 3.0)
        df_z['GRW_Q'] = grw_q_z
        df_z['GRW_A'] = grw_a_z

        try:
            mix = float(os.environ.get("GRW_Q_ANNUAL_MIX", "0.7"))
        except Exception:
            mix = 0.7
        if not np.isfinite(mix):
            mix = 0.7
        mix = float(np.clip(mix, 0.0, 1.0))

        weights_q: list[float] = []
        weights_a: list[float] = []
        grw_mix: list[float] = []
        for idx in df.index:
            q_val = grw_q_z.get(idx, np.nan)
            a_val = grw_a_z.get(idx, np.nan)
            q_ok = np.isfinite(q_val)
            a_ok = np.isfinite(a_val)
            if q_ok and a_ok:
                wq, wa = mix, 1.0 - mix
            elif q_ok:
                wq, wa = 1.0, 0.0
            elif a_ok:
                wq, wa = 0.0, 1.0
            else:
                wq = wa = np.nan
                grw_mix.append(np.nan)
                weights_q.append(wq)
                weights_a.append(wa)
                continue
            weights_q.append(wq)
            weights_a.append(wa)
            grw_mix.append(q_val * wq + a_val * wa)

        wq_series = pd.Series(weights_q, index=df.index, dtype=float)
        wa_series = pd.Series(weights_a, index=df.index, dtype=float)
        grw_series = pd.Series(grw_mix, index=df.index, dtype=float).clip(-3.0, 3.0)

        df_z['GROWTH_F'] = grw_series
        df_z['GRW_FLEX_WEIGHT'] = 1.0  # 現状は固定（SECの可用性に依らず）

        if str(os.environ.get("GRW_DBG_DETAIL", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            df_z['GRW_Q_DBG'] = pd.Series(df['GRW_Q_RAW'], index=df.index, dtype=float)
            df_z['GRW_A_DBG'] = pd.Series(df['GRW_A_RAW'], index=df.index, dtype=float)
            df_z['GRW_WQ_DBG'] = wq_series
            df_z['GRW_WA_DBG'] = wa_series

        df_z['MOM_F'] = robust_z(0.40*df_z['RS']
            + 0.15*df_z['TR_str']
            + 0.15*df_z['RS_SLOPE_6W']
            + 0.15*df_z['RS_SLOPE_13W']
            + 0.10*df_z['MA200_SLOPE_5M']
            + 0.10*df_z['MA200_UP_STREAK_D']).clip(-3.0,3.0)
        df_z['VOL'] = robust_z(df['BETA'])
        df_z['QAL'], df_z['YLD'], df_z['MOM'] = df_z['QUALITY_F'], df_z['YIELD_F'], df_z['MOM_F']
        df_z.drop(columns=['QUALITY_F','YIELD_F','MOM_F'], inplace=True, errors='ignore')

        # df_z 全明細をページングしてログ出力（最小版）
        if getattr(cfg, "debug_mode", False):
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_colwidth", None)
            pd.set_option("display.width", None)
            page = int(getattr(cfg, "debug_dfz_page", 50))  # デフォルト50行単位
            n = len(df_z)
            logger.info("=== df_z FULL DUMP start === rows=%d cols=%d page=%d", n, df_z.shape[1], page)
            for i in range(0, n, page):
                j = min(i + page, n)
                try:
                    chunk_str = df_z.iloc[i:j].to_string()
                except Exception:
                    chunk_str = df_z.iloc[i:j].astype(str).to_string()
                logger.info("--- df_z rows %d..%d ---\n%s", i, j-1, chunk_str)
            logger.info("=== df_z FULL DUMP end ===")

        # === begin: BIO LOSS PENALTY =====================================
        try:
            penalty_z = float(os.getenv("BIO_LOSS_PENALTY_Z", "0.8"))
        except Exception:
            penalty_z = 0.8

        def _is_bio_like(t: str) -> bool:
            inf = info.get(t, {}) if isinstance(info, dict) else {}
            sec = str(inf.get("sector", "")).lower()
            ind = str(inf.get("industry", "")).lower()
            if "health" not in sec:
                return False
            keys = ("biotech", "biopharma", "pharma")
            return any(k in ind for k in keys)

        tickers_s = pd.Index(df_z.index)
        is_bio = pd.Series({t: _is_bio_like(t) for t in tickers_s})
        is_loss = pd.Series({t: (pd.notna(df.loc[t,"EPS"]) and df.loc[t,"EPS"] <= 0) for t in tickers_s})
        mask_bio_loss = (is_bio & is_loss).reindex(df_z.index).fillna(False)

        if bool(mask_bio_loss.any()) and penalty_z > 0:
            df_z.loc[mask_bio_loss, "GROWTH_F"] = df_z.loc[mask_bio_loss, "GROWTH_F"] - penalty_z
            df_z["GROWTH_F"] = df_z["GROWTH_F"].clip(-3.0, 3.0)
        # === end: BIO LOSS PENALTY =======================================

        _debug_only_cols = [c for c in df_z.columns if c.endswith("_RAW")]
        _no_score_cols = ["DIV_TTM_PS", "DIV_YOY", "LOW52PCT25_EXCESS", "MA50_OVER_200"]
        _drop_cols = [c for c in (_debug_only_cols + _no_score_cols) if c in df_z.columns]
        if _drop_cols:
            df_z = df_z.drop(columns=_drop_cols, errors="ignore")

        assert not any(c.endswith("_RAW") for c in df_z.columns)
        for c in ["DIV_TTM_PS","DIV_YOY","LOW52PCT25_EXCESS","MA50_OVER_200"]:
            assert c not in df_z.columns

        df_z['TRD'] = 0.0  # TRDはスコア寄与から外し、テンプレ判定はフィルタで行う（列は表示互換のため残す）
        if 'BETA' not in df_z.columns: df_z['BETA'] = robust_z(df['BETA'])

        df_z['D_VOL_RAW'] = robust_z(0.40*df_z['DOWNSIDE_DEV'] + 0.22*df_z['RESID_VOL'] + 0.18*df_z['MDD_1Y'] - 0.10*df_z['DOWN_OUTPERF'] - 0.05*df_z['EXT_200'] - 0.08*df_z['SIZE'] - 0.10*df_z['LIQ'] + 0.10*df_z['BETA'])
        df_z['D_QAL']     = robust_z(0.35*df_z['QAL'] + 0.20*df_z['FCF'] + 0.15*df_z['CURR_RATIO'] - 0.15*df_z['DEBT2EQ'] - 0.15*df_z['EPS_VAR_8Q'])
        df_z['D_YLD']     = robust_z(0.45*df_z['DIV'] + 0.25*df_z['DIV_STREAK'] + 0.20*df_z['DIV_FCF_COVER'] - 0.10*df_z['DIV_VAR5'])
        df_z['D_TRD']     = robust_z(0.40*df_z.get('MA200_SLOPE_5M',0) - 0.30*df_z.get('EXT_200',0) + 0.15*df_z.get('NEAR_52W_HIGH',0) + 0.15*df_z['TR'])

        # --- 重みは cfg を優先（外部があればそれを使用） ---
        # ① 全銘柄で G/D スコアを算出（unmasked）
        g_score_all = _as_numeric_series(df_z.mul(pd.Series(cfg.weights.g)).sum(axis=1))

        d_comp = pd.concat({
            'QAL': df_z['D_QAL'],
            'YLD': df_z['D_YLD'],
            'VOL': df_z['D_VOL_RAW'],
            'TRD': df_z['D_TRD']
        }, axis=1)
        dw = pd.Series(cfg.weights.d, dtype=float).reindex(['QAL','YLD','VOL','TRD']).fillna(0.0)
        globals()['D_WEIGHTS_EFF'] = dw.copy()
        d_score_all = _as_numeric_series(d_comp.mul(dw, axis=1).sum(axis=1))

        # ② テンプレ判定（既存ロジックそのまま）
        mask = df['trend_template']
        if not bool(mask.any()):
            mask = ((df.get('P_OVER_LOW52', np.nan) >= 0.25) &
                (df.get('NEAR_52W_HIGH', np.nan) >= -0.30) &
                (df.get('RS', np.nan) >= 0.08) &
                (df.get('MA200_SLOPE_1M', np.nan) > 0) &
                (df.get('P_OVER_150', np.nan) > 0) & (df.get('P_OVER_200', np.nan) > 0) &
                (df.get('MA150_OVER_200', np.nan) > 0) &
                (df.get('MA50_OVER_150', np.nan) > 0) & (df.get('MA50_OVER_200', np.nan) > 0) &
                (df.get('TR_str', np.nan) > 0)).fillna(False)
            df['trend_template'] = mask

        # ③ 採用用は mask、表示/分析用は列で全銘柄保存
        g_score = _as_numeric_series(g_score_all.loc[mask])
        Scorer.g_score = g_score
        df_z['GSC'] = g_score_all
        df_z['DSC'] = d_score_all

        try:
            current = (pd.read_csv("current_tickers.csv")
                  .iloc[:, 0]
                  .str.upper()
                  .tolist())
        except FileNotFoundError:
            warnings.warn("current_tickers.csv not found — bonus skipped")
            current = []

        mask_bonus = g_score.index.isin(current)
        if mask_bonus.any():
            # 1) factor.BONUS_COEFF から k を決め、無ければ 0.4
            k = float(getattr(sys.modules.get("factor"), "BONUS_COEFF", 0.4))
            # 2) g 側の σ を取り、NaN なら 0 に丸める
            sigma_g = g_score.std()
            if pd.isna(sigma_g):
                sigma_g = 0.0
            bonus_g = round(k * sigma_g, 3)
            g_score.loc[mask_bonus] += bonus_g
            Scorer.g_score = g_score
            # 3) D 側も同様に σ の NaN をケア
            sigma_d = d_score_all.std()
            if pd.isna(sigma_d):
                sigma_d = 0.0
            bonus_d = round(k * sigma_d, 3)
            d_score_all.loc[d_score_all.index.isin(current)] += bonus_d

        try:
            df = _apply_growth_entry_flags(df, ib, self, win_breakout=5, win_pullback=5)
        except Exception:
            pass

        df_full = df.copy()
        df_full_z = df_z.copy()

        from factor import FeatureBundle  # type: ignore  # 実行時importなし（循環回避）
        return FeatureBundle(df=df,
            df_z=df_z,
            g_score=g_score,
            d_score_all=d_score_all,
            missing_logs=pd.DataFrame(missing_logs),
            df_full=df_full,
            df_full_z=df_full_z,
            scaler=None)

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
        px = px.ffill(limit=2)
        ema21 = px[g_universe].ewm(span=21, adjust=False).mean()
        ma50  = px[g_universe].rolling(50).mean()
        ma150 = px[g_universe].rolling(150).mean()
        ma200 = px[g_universe].rolling(200).mean()
        atr20 = (hi[g_universe] - lo[g_universe]).rolling(20).mean()
        vol20 = vol[g_universe].rolling(20).mean()
        vol50 = vol[g_universe].rolling(50).mean()

        # トレンドテンプレート合格
        trend_template_ok = (px[g_universe] > ma50) & (px[g_universe] > ma150) & (px[g_universe] > ma200) \
                            & (ma150 > ma200) & (ma200.diff() > 0)

        # 汎用ピボット：直近65営業日の高値（当日除外）
        pivot_price = hi[g_universe].rolling(65).max().shift(1)

        # 相対力：年内高値更新
        bench_aligned = bench.reindex(px.index).ffill()
        rs = px[g_universe].div(bench_aligned, axis=0)
        rs_high = rs.rolling(252).max().shift(1)

        # ブレイクアウト「発生日」：条件立ち上がり
        breakout_today = trend_template_ok & (px[g_universe] > pivot_price) \
                         & (vol[g_universe] >= 1.5 * vol50) & (rs > rs_high)
        breakout_event = breakout_today & ~breakout_today.shift(1).fillna(False)

        # 押し目反発「発生日」：EMA21帯×出来高ドライアップ×前日高値越え×終値EMA21上
        near_ema21_band = px[g_universe].between(ema21 - atr20, ema21 + atr20)
        volume_dryup = (vol20 / vol50) <= 1.0
        pullback_bounce_confirmed = (px[g_universe] > hi[g_universe].shift(1)) & (px[g_universe] > ema21)
        pullback_today = trend_template_ok & near_ema21_band & volume_dryup & pullback_bounce_confirmed
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

