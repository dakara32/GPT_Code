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

DEBUG_SCOPE_STRICT = (
    os.getenv("DEBUG_SCOPE_STRICT", "false").strip().lower() == "true"
)

FACTOR_COLUMNS = {
    "GRW": [
        "GROWTH_F",
        "GRW_FLEX_SCORE",
        "GRW_REV_YOY_Q",
        "GRW_REV_ACC_Q",
        "GRW_REV_QOQ",
        "GRW_REV_TTM2",
        "GRW_REV_YOY_Y",
        "GRW_PRICE_PROXY",
        "GRW_PATH",
    ],
    "MOM": ["MOM", "MOM_RAW", "MOM_P12M", "MOM_P6M", "MOM_P1M"],
    "VOL": ["VOL", "VOL_SD", "VOL_BETA"],
    "QUAL": ["QUAL", "ROE", "ROA", "FCF_MGN"],
    "VAL": ["VAL", "PE", "PB", "PS", "EVEBITDA"],
}

if TYPE_CHECKING:
    from factor import PipelineConfig  # type: ignore  # 実行時importなし（循環回避）

logger = logging.getLogger(__name__)


def _log(stage, msg):
    try:
        print(f"[DBG][{_dt.utcnow().isoformat(timespec='seconds')}Z][{stage}] {msg}")
    except Exception:
        print(f"[DBG][{stage}] {msg}")


def _detect_used_cols(df, df_z):
    used = set()

    try:
        import factor as _f

        for k in getattr(_f, "g_weights", {}).keys():
            if k in df_z.columns:
                used.add(k)
    except Exception:
        pass

    try:
        import scorer as _sc

        for k in getattr(_sc, "D_WEIGHTS_EFF", {}).keys():
            if k in df_z.columns:
                used.add(k)
    except Exception:
        pass

    for k in [
        "GROWTH_F",
        "MOM",
        "TRD",
        "VOL",
        "D_QAL",
        "D_YLD",
        "D_VOL_RAW",
        "D_TRD",
    ]:
        if k in df_z.columns:
            used.add(k)

    grw_cols = [
        "GRW_PATH",
        "GRW_FLEX_SCORE",
        "GROWTH_F",
        "GRW_REV_YOY_Q",
        "GRW_REV_ACC_Q",
        "GRW_REV_QOQ",
        "GRW_REV_TTM2",
        "GRW_REV_YOY_Y",
        "GRW_PRICE_PROXY",
    ]
    for k in grw_cols:
        if (k in df.columns) or (k in df_z.columns):
            used.add(k)

    for c in df_z.columns:
        if isinstance(c, str) and c.startswith("D_"):
            used.add(c)

    num = df_z.select_dtypes(include=["number"])
    if not num.empty:
        var_top = num.var().sort_values(ascending=False).head(20).index.tolist()
        used.update(var_top)

    return sorted(used)


def _reorder_for_debug(df, df_z, factor_cols=FACTOR_COLUMNS):
    cols: list[str] = []
    for fac in ["GRW", "MOM", "VOL", "QUAL", "VAL"]:
        for c in factor_cols.get(fac, []):
            if c in getattr(df_z, "columns", []) or c in getattr(df, "columns", []):
                cols.append(c)
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def dump_dfz_scoped(df, df_z, *, topk=20, logger=None):
    import numpy as np, pandas as pd, logging

    lg = logger or logging.getLogger(__name__)

    if not DEBUG_SCOPE_STRICT:
        dfz = df_z.copy()
        lg.info("DEBUG scope: disabled (showing ALL %d columns).", dfz.shape[1])
    else:
        rel = _detect_used_cols(df, df_z)
        dfz = df_z[[c for c in rel if c in df_z.columns]].copy()
        if dfz.shape[1] < 15:
            num = df_z.select_dtypes(include=["number"])
            add = []
            if not num.empty:
                add = [
                    c
                    for c in num.var().sort_values(ascending=False).head(30).index
                    if c not in dfz.columns
                ]
                if dfz.shape[1] < 15:
                    add = add[: 15 - dfz.shape[1]]
                else:
                    add = []
            dfz = pd.concat([dfz, df_z[add]], axis=1)
            lg.info("DEBUG scope too small → fallback add %d cols", len(add))
        excluded = [c for c in df_z.columns if c not in dfz.columns]
        lg.info(
            "DEBUG scope: %d relevant cols kept, %d excluded.",
            dfz.shape[1],
            len(excluded),
        )

    nan_top = dfz.isna().sum().sort_values(ascending=False).head(topk)
    lg.info("scorer:NaN columns (top%d):", topk)
    for c, n in nan_top.items():
        lg.info("%s\t%d", c, int(n))

    num_dfz = dfz.select_dtypes(include=["number"])
    if not num_dfz.empty:
        ztop = (num_dfz == 0).mean().sort_values(ascending=False).head(topk)
        lg.info("scorer:Zero-dominated columns (top%d):", topk)
        for c, r in ztop.items():
            lg.info("%s\t%.2f%%", c, 100.0 * float(r))

    return dfz


def save_factor_debug_csv(df, df_z, path="out/factor_debug_latest.csv"):
    import os, pandas as pd, logging

    lg = logging.getLogger(__name__)
    try:
        cols = _reorder_for_debug(df, df_z)
        dump = pd.DataFrame(index=df.index)
        for c in cols:
            if c in getattr(df, "columns", []):
                dump[c] = df[c]
            if c in getattr(df_z, "columns", []):
                dump[c] = df_z[c]
        dump.reset_index(names=["symbol"], inplace=True)
        if path:
            dirpath = os.path.dirname(path) or "."
            os.makedirs(dirpath, exist_ok=True)
            dump.to_csv(path, index=False)
        lg.info(
            "factor debug CSV saved: %s (cols=%d rows=%d)",
            path,
            dump.shape[1],
            dump.shape[0],
        )
    except Exception as e:
        lg.warning("factor debug CSV failed: %s", e)


def log_grw_stats(df, df_z, logger):
    import numpy as np, pandas as pd

    try:
        s = pd.to_numeric(df.get("GRW_FLEX_SCORE", pd.Series(dtype=float)), errors="coerce")
        z = pd.to_numeric(df_z.get("GROWTH_F", pd.Series(dtype=float)), errors="coerce")
        if s.size:
            logger.info(
                "GRW raw stats: n=%d, median=%.3f, mad=%.3f, std=%.3f",
                s.count(),
                np.nanmedian(s),
                np.nanmedian(np.abs(s - np.nanmedian(s))),
                np.nanstd(s),
            )
        if z.size and not z.dropna().empty:
            clip_hi = float((z >= 2.95).mean() * 100.0)
            clip_lo = float((z <= -2.95).mean() * 100.0)
            logger.info(
                "GRW z stats: min=%.2f, p25=%.2f, med=%.2f, p75=%.2f, max=%.2f, clipped_hi=%.1f%%, clipped_lo=%.1f%%",
                np.nanmin(z),
                np.nanpercentile(z.dropna(), 25),
                np.nanmedian(z),
                np.nanpercentile(z.dropna(), 75),
                np.nanmax(z),
                clip_hi,
                clip_lo,
            )
        if "GRW_PATH" in getattr(df, "columns", []):
            logger.info(
                "GRW path breakdown: %s",
                df["GRW_PATH"].value_counts(dropna=False).to_dict(),
            )
    except Exception as e:
        logger.warning("GRW stats log failed: %s", e)


def _grw_record_to_df(t: str, info_t: dict, df):
    if not isinstance(df, pd.DataFrame):
        return
    raw_parts = info_t.get("DEBUG_GRW_PARTS") if isinstance(info_t, dict) else None
    parts: dict[str, Any] = {}
    if isinstance(raw_parts, str):
        try:
            parts = json.loads(raw_parts)
        except Exception:
            parts = {}
    elif isinstance(raw_parts, dict):
        parts = raw_parts
    path = info_t.get("DEBUG_GRW_PATH") if isinstance(info_t, dict) else None
    score = info_t.get("GRW_SCORE") if isinstance(info_t, dict) else None

    def _part_value(key: str):
        value = parts.get(key) if isinstance(parts, dict) else None
        if value is None:
            return np.nan
        try:
            return float(value)
        except Exception:
            return np.nan

    df.loc[t, "GRW_PATH"] = path
    df.loc[t, "GRW_FLEX_SCORE"] = np.nan if score is None else float(score)
    df.loc[t, "GRW_REV_YOY_Q"] = _part_value("rev_yoy_q")
    df.loc[t, "GRW_REV_ACC_Q"] = _part_value("rev_acc_q")
    df.loc[t, "GRW_REV_QOQ"] = _part_value("rev_qoq")
    df.loc[t, "GRW_REV_TTM2"] = _part_value("rev_ttm2")
    df.loc[t, "GRW_REV_YOY_Y"] = _part_value("rev_yoy_y")
    df.loc[t, "GRW_PRICE_PROXY"] = _part_value("price_proxy")

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


def _dump_dfz(
    df: pd.DataFrame,
    df_z: pd.DataFrame,
    debug_mode: bool,
    max_rows: int = 400,
    ndigits: int = 3,
) -> None:
    """df_z を System log(INFO) へダンプする簡潔なユーティリティ."""

    if not debug_mode:
        return
    try:
        dfz_scoped = dump_dfz_scoped(df, df_z, topk=20, logger=logger)
        ordered = _reorder_for_debug(df, df_z)
        rel_set = set(dfz_scoped.columns)
        view_cols = [c for c in ordered if c in rel_set]
        if not view_cols:
            view_cols = list(dfz_scoped.columns)
        view = dfz_scoped[view_cols].copy()
        view = view.apply(
            lambda s: s.round(ndigits)
            if getattr(getattr(s, "dtype", None), "kind", "") in ("f", "i")
            else s
        )
        if len(view) > max_rows:
            view = view.iloc[:max_rows]

        logger.info("===== DF_Z DUMP START =====")
        logger.info("\n%s", view.to_string(max_rows=None, max_cols=None))
        logger.info("===== DF_Z DUMP END =====")
    except Exception as exc:
        logger.warning("df_z dump failed: %s", exc)

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


def _nz(x) -> float:
    if x is None:
        return 0.0
    try:
        value = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def _winsor(x, lo=-2.0, hi=2.0) -> float:
    v = _nz(x)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _round_debug(x, ndigits: int = 4):
    try:
        value = float(x)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def _calc_grw_flexible(
    ticker: str,
    info_entry: dict | None,
    close_series: pd.Series | None,
    volume_series: pd.Series | None,
):
    info_entry = info_entry if isinstance(info_entry, dict) else {}

    s_rev_q = _ensure_series(info_entry.get("SEC_REV_Q_SERIES"))
    s_eps_q = _ensure_series(info_entry.get("SEC_EPS_Q_SERIES"))
    s_rev_y = _ensure_series(info_entry.get("SEC_REV_Y_SERIES"))

    nQ = int(getattr(s_rev_q, "size", 0))
    nY = int(getattr(s_rev_y, "size", 0))

    parts: dict[str, Any] = {"nQ": nQ, "nY": nY}
    path = "NONE"
    w = 0.0

    def _valid_ratio(a, b):
        try:
            na, nb = float(a), float(b)
        except Exception:
            return None
        if not np.isfinite(na) or not np.isfinite(nb) or nb == 0:
            return None
        return na, nb

    def yoy_q(series: pd.Series) -> float | None:
        s = _ensure_series(series)
        if s.empty:
            return None
        s = s.sort_index()
        if isinstance(s.index, pd.DatetimeIndex):
            last_idx = s.index[-1]
            window_start = last_idx - pd.DateOffset(months=15)
            window_end = last_idx - pd.DateOffset(months=9)
            candidates = s.loc[(s.index >= window_start) & (s.index <= window_end)]
            if candidates.empty:
                candidates = s.loc[s.index <= window_end]
            if candidates.empty:
                return None
            v1 = candidates.iloc[-1]
            v0 = s.iloc[-1]
        else:
            if s.size < 5:
                return None
            v0 = s.iloc[-1]
            v1 = s.iloc[-5]
        pair = _valid_ratio(v0, v1)
        if pair is None:
            return None
        a, b = pair
        return float(a / b - 1.0)

    def qoq(series: pd.Series) -> float | None:
        s = _ensure_series(series)
        if s.size < 2:
            return None
        s = s.sort_index()
        v0, v1 = s.iloc[-1], s.iloc[-2]
        pair = _valid_ratio(v0, v1)
        if pair is None:
            return None
        a, b = pair
        return float(a / b - 1.0)

    def ttm_delta(series: pd.Series) -> float | None:
        s = _ensure_series(series)
        if s.size < 2:
            return None
        s = s.sort_index()
        k = int(min(4, s.size))
        cur_slice = s.iloc[-k:]
        prev_slice = s.iloc[:-k]
        if prev_slice.empty:
            return None
        prev_k = int(min(k, prev_slice.size))
        cur_sum = float(cur_slice.sum())
        prev_sum = float(prev_slice.iloc[-prev_k:].sum())
        pair = _valid_ratio(cur_sum, prev_sum)
        if pair is None:
            return None
        a, b = pair
        return float(a / b - 1.0)

    def yoy_y(series: pd.Series) -> float | None:
        s = _ensure_series(series)
        if s.size < 2:
            return None
        s = s.sort_index()
        v0, v1 = s.iloc[-1], s.iloc[-2]
        pair = _valid_ratio(v0, v1)
        if pair is None:
            return None
        a, b = pair
        return float(a / b - 1.0)

    def price_proxy_growth() -> float | None:
        if not isinstance(close_series, pd.Series):
            return None
        close = close_series.sort_index().dropna()
        if close.empty:
            return None
        hh_window = int(min(126, len(close)))
        if hh_window < 20:
            return None
        hh = close.rolling(hh_window).max().iloc[-1]
        prox = None
        if np.isfinite(hh) and hh > 0:
            prox = float(close.iloc[-1] / hh)
        rs6 = None
        if len(close) >= 63:
            rs6 = float(close.pct_change(63).iloc[-1])
        rs12 = None
        if len(close) >= 126:
            rs12 = float(close.pct_change(126).iloc[-1])
        vexp = None
        if isinstance(volume_series, pd.Series):
            vol = volume_series.reindex(close.index).dropna()
            if len(vol) >= 50:
                v20 = vol.rolling(20).mean().iloc[-1]
                v50 = vol.rolling(50).mean().iloc[-1]
                if np.isfinite(v20) and np.isfinite(v50) and v50 > 0:
                    vexp = float(v20 / v50 - 1.0)
        prox = 0.0 if prox is None or not np.isfinite(prox) else prox
        rs6 = 0.0 if rs6 is None or not np.isfinite(rs6) else rs6
        rs12 = 0.0 if rs12 is None or not np.isfinite(rs12) else rs12
        vexp = 0.0 if vexp is None or not np.isfinite(vexp) else vexp
        return 0.5 * prox + 0.3 * rs6 + 0.2 * rs12 + 0.2 * vexp

    price_alt = price_proxy_growth() or 0.0
    core = 0.0
    core_raw = 0.0
    price_raw = price_alt

    if nQ >= 5:
        path = "P5"
        yq = yoy_q(s_rev_q)
        parts["rev_yoy_q"] = yq
        tmp_prev = s_rev_q.iloc[:-1] if s_rev_q.size > 1 else s_rev_q
        acc = None
        if tmp_prev.size >= 5 and yq is not None:
            yq_prev = yoy_q(tmp_prev)
            if yq_prev is not None:
                acc = float(yq - yq_prev)
        parts["rev_acc_q"] = acc
        eps_yoy = yoy_q(s_eps_q) if s_eps_q.size >= 5 else None
        parts["eps_yoy_q"] = eps_yoy
        eps_acc = None
        if eps_yoy is not None and s_eps_q.size > 5:
            eps_prev = s_eps_q.iloc[:-1]
            if eps_prev.size >= 5:
                eps_prev_yoy = yoy_q(eps_prev)
                if eps_prev_yoy is not None:
                    eps_acc = float(eps_yoy - eps_prev_yoy)
        parts["eps_acc_q"] = eps_acc
        w = 1.0
        core_raw = (
            0.60 * _nz(yq)
            + 0.20 * _nz(acc)
            + 0.15 * _nz(eps_yoy)
            + 0.05 * _nz(eps_acc)
        )
        price_alt = 0.0
    elif 2 <= nQ <= 4:
        path = "P24"
        rev_qoq = qoq(s_rev_q)
        rev_ttm2 = ttm_delta(s_rev_q)
        parts["rev_qoq"] = rev_qoq
        parts["rev_ttm2"] = rev_ttm2
        eps_qoq = qoq(s_eps_q) if s_eps_q.size >= 2 else None
        parts["eps_qoq"] = eps_qoq
        w = min(1.0, nQ / 5.0)
        core_raw = 0.6 * _nz(rev_qoq) + 0.3 * _nz(rev_ttm2) + 0.1 * _nz(eps_qoq)
    else:
        path = "P1Y"
        rev_yoy_y = yoy_y(s_rev_y) if nY >= 2 else None
        parts["rev_yoy_y"] = rev_yoy_y
        w = 0.6 * min(1.0, nY / 3.0) if nY >= 2 else 0.4
        core_raw = _nz(rev_yoy_y)
        if nQ <= 1 and nY < 2 and price_alt == 0.0:
            price_alt = price_proxy_growth() or 0.0

    core = _winsor(core_raw, lo=-1.5, hi=1.5)
    price_alt = _winsor(price_alt, lo=-1.5, hi=1.5)
    grw = _winsor(w * core + (1.0 - w) * (0.5 * _nz(price_alt)), lo=-2.0, hi=2.0)

    parts.update(
        {
            "core_raw": core_raw,
            "core": core,
            "price_proxy_raw": price_raw,
            "price_proxy": price_alt,
            "weight": w,
            "score": grw,
        }
    )

    parts_out: dict[str, Any] = {
        "nQ": nQ,
        "nY": nY,
    }
    for key, value in parts.items():
        if key in ("nQ", "nY"):
            continue
        rounded = _round_debug(value)
        parts_out[key] = rounded

    info_entry["DEBUG_GRW_PATH"] = path
    info_entry["DEBUG_GRW_PARTS"] = json.dumps(parts_out, ensure_ascii=False, sort_keys=True)
    info_entry["GRW_SCORE"] = grw
    info_entry["GRW_WEIGHT"] = w
    info_entry["GRW_CORE"] = core
    info_entry["GRW_PRICE_PROXY"] = price_alt

    return {
        "score": grw,
        "path": path,
        "parts": info_entry["DEBUG_GRW_PARTS"],
        "weight": w,
        "core": core,
        "price_proxy": price_alt,
    }


D_WEIGHTS_EFF = None  # 出力表示互換のため


def _scalar(v):
    """単一セル代入用に値をスカラーへ正規化する。

    - pandas Series -> .iloc[-1]（最後を採用）
    - list/tuple/ndarray -> 最後の要素
    - それ以外          -> そのまま
    取得失敗時は np.nan を返す。
    """
    import numpy as _np
    import pandas as _pd
    try:
        if isinstance(v, _pd.Series):
            return v.iloc[-1] if len(v) else _np.nan
        if isinstance(v, (list, tuple, _np.ndarray)):
            return v[-1] if len(v) else _np.nan
        return v
    except Exception:
        return _np.nan


# ---- Scorer 本体 -------------------------------------------------------------
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
        for t in tickers:
            d, s = info[t], px[t]; ev = self.ev_fallback(d, tickers_bulk.tickers[t])
            try:
                volume_series_full = ib.data['Volume'][t]
            except Exception:
                volume_series_full = None

            grw_result = _calc_grw_flexible(t, d, s, volume_series_full)
            _grw_record_to_df(t, d, df)
            df.loc[t,'GRW_FLEX_SCORE'] = grw_result.get('score')
            df.loc[t,'GRW_FLEX_WEIGHT'] = grw_result.get('weight')
            df.loc[t,'GRW_FLEX_CORE'] = grw_result.get('core')
            df.loc[t,'GRW_FLEX_PRICE'] = grw_result.get('price_proxy')
            df.loc[t,'DEBUG_GRW_PATH'] = grw_result.get('path')
            df.loc[t,'DEBUG_GRW_PARTS'] = grw_result.get('parts')

            # --- 基本特徴 ---
            df.loc[t,'TR']   = self.trend(s)
            df.loc[t,'EPS']  = _scalar(eps_df.loc[t,'EPS_TTM']) if t in eps_df.index else np.nan
            df.loc[t,'EPS_Q'] = _scalar(eps_df.loc[t,'EPS_Q_LastQ']) if t in eps_df.index else np.nan
            df.loc[t,'REV_TTM'] = _scalar(eps_df.loc[t,'REV_TTM']) if t in eps_df.index else np.nan
            df.loc[t,'REV_Q']   = _scalar(eps_df.loc[t,'REV_Q_LastQ']) if t in eps_df.index else np.nan
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

        cnt_rev_series = sum(1 for _t, d in info.items() if _has_sec_series(d.get("SEC_REV_Q_SERIES")))
        cnt_eps_series = sum(1 for _t, d in info.items() if _has_sec_series(d.get("SEC_EPS_Q_SERIES")))
        logger.info(
            "[DERIV] SEC series presence: REV_Q=%d, EPS_Q=%d (universe=%d)",
            cnt_rev_series,
            cnt_eps_series,
            len(info),
        )

        rev_q_ge5 = 0
        ttm_yoy_avail = 0
        wrote_growth = 0

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
                logger.debug(
                    "[DERIV] %s: rev_q_len=%s eps_q_len=%s fallback_qearn=%s",
                    t,
                    _series_len(rev_series),
                    _series_len(eps_series),
                    fallback_qearn,
                )

                r_src = _pick_series(d, ["SEC_REV_Q_SERIES", "rev_q_series_pairs", "rev_q_series"])
                e_src = _pick_series(d, ["SEC_EPS_Q_SERIES", "eps_q_series_pairs", "eps_q_series"])
                r_raw = _ensure_series(r_src)
                e_raw = _ensure_series(e_src)
                _log("DERIV_SRC", f"{t} rev_raw_len={r_raw.size} eps_raw_len={e_raw.size}")

                r_q = _to_quarterly(r_raw)
                e_q = _to_quarterly(e_raw)
                _log("DERIV_Q", f"{t} rev_q_len={r_q.size} eps_q_len={e_q.size}")
                if r_q.size >= 5:
                    rev_q_ge5 += 1

                r_yoy_ttm = _ttm_yoy_from_quarterly(r_q)
                e_yoy_ttm = _ttm_yoy_from_quarterly(e_q)
                has_ttm = int(not r_yoy_ttm.dropna().empty)
                ttm_yoy_avail += has_ttm
                _log("DERIV_TTM", f"{t} rev_ttm_yoy_len={r_yoy_ttm.dropna().size} eps_ttm_yoy_len={e_yoy_ttm.dropna().size}")

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

                wrote_growth += 1
                _log(
                    "DERIV_WRITE",
                    f"{t} wrote: Q_YOY(rev={rev_q_yoy}, eps={eps_q_yoy}) ANN(rev_yoy={rev_yoy}, acc={rev_acc}, var={rev_var}) streak={rev_ann_streak}",
                )

            except Exception as e:
                logger.warning("[DERIV_WARN] %s growth-derivatives failed: %s", t, e)
                _log("DERIV_WARN", f"{t} {type(e).__name__}: {e}")

        _log("DERIV_SUMMARY", f"rev_q_len>=5: {rev_q_ge5}/{len(tickers)}  ttm_yoy_available: {ttm_yoy_avail}  wrote_growth_for: {wrote_growth}")

        try:
            cols = [
                "REV_Q_YOY",
                "EPS_Q_YOY",
                "REV_YOY",
                "EPS_YOY",
                "REV_YOY_ACC",
                "REV_YOY_VAR",
                "REV_ANN_STREAK",
            ]
            cnt = {c: int(df[c].count()) for c in cols if c in df.columns}
            _log("DERIV_NONNAN_COUNTS", str(cnt))
        except Exception as e:
            _log("DERIV_NONNAN_COUNTS", f"error: {e}")

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

        # === Z化と合成 ===
        for col in ['ROE','FCF','REV','EPS']: df[f'{col}_W'] = winsorize_s(df[col], 0.02)

        df_z = pd.DataFrame(index=df.index)
        for col in ['EPS','REV','ROE','FCF','RS','TR_str','BETA','DIV','DIV_STREAK']: df_z[col] = robust_z(df[col])
        df_z['REV'], df_z['EPS'], df_z['TR'] = robust_z(df['REV_W']), robust_z(df['EPS_W']), robust_z(df['TR'])
        for col in ['P_OVER_150','P_OVER_200','MA50_OVER_200','MA200_SLOPE_5M','LOW52PCT25_EXCESS','NEAR_52W_HIGH','RS_SLOPE_6W','RS_SLOPE_13W','MA200_UP_STREAK_D']: df_z[col] = robust_z(df[col])

        # === Growth深掘り系（欠損保持z + RAW併載） ===
        grw_cols = ['REV_Q_YOY','EPS_Q_YOY','REV_YOY','EPS_YOY','REV_YOY_ACC','REV_YOY_VAR','FCF_MGN','RULE40','REV_ANN_STREAK']
        for col in grw_cols:
            if col in df.columns:
                raw = pd.to_numeric(df[col], errors="coerce")
                df_z[col] = robust_z_keepnan(raw)
                df_z[f'{col}_RAW'] = raw
        for k in ("TREND_SLOPE_EPS", "TREND_SLOPE_REV"):
            if k in df.columns and k not in df_z.columns:
                raw = pd.to_numeric(df[k], errors="coerce")
                df_z[k] = robust_z_keepnan(raw)
                df_z[f'{k}_RAW'] = raw
        for col in ['DOWNSIDE_DEV','MDD_1Y','RESID_VOL','DOWN_OUTPERF','EXT_200','DIV_TTM_PS','DIV_VAR5','DIV_YOY','DIV_FCF_COVER','DEBT2EQ','CURR_RATIO','EPS_VAR_8Q','MARKET_CAP','ADV60_USD']: df_z[col] = robust_z(df[col])

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
        df_z['TREND_SLOPE_REV_RAW'] = slope_rev_combo
        df_z['TREND_SLOPE_REV'] = slope_rev_combo.clip(-3.0, 3.0)

        # EPSトレンドスロープ（四半期）
        slope_eps = 0.60*zpos(df_z['EPS_Q_YOY']) + 0.40*zpos(df_z['EPS_POS'])
        df_z['TREND_SLOPE_EPS_RAW'] = slope_eps
        df_z['TREND_SLOPE_EPS'] = slope_eps.clip(-3.0, 3.0)

        # 年次トレンド（サブ）
        slope_rev_yr = zpos(df_z['REV_YOY'])
        slope_eps_yr = zpos(df_z.get('EPS_YOY', pd.Series(0.0, index=df.index)))
        streak_base = df['REV_ANN_STREAK'].clip(lower=0).fillna(0)
        streak_yr = streak_base / (streak_base.abs() + 1.0)
        slope_rev_yr_combo = 0.7*slope_rev_yr + 0.3*streak_yr
        df_z['TREND_SLOPE_REV_YR_RAW'] = slope_rev_yr_combo
        df_z['TREND_SLOPE_REV_YR'] = slope_rev_yr_combo.clip(-3.0, 3.0)
        df_z['TREND_SLOPE_EPS_YR_RAW'] = slope_eps_yr
        df_z['TREND_SLOPE_EPS_YR'] = slope_eps_yr.clip(-3.0, 3.0)

        # ===== GRW flexible score (variable data paths) =====
        grw_raw = pd.to_numeric(df.get('GRW_FLEX_SCORE'), errors="coerce")
        df_z['GRW_FLEX_SCORE_RAW'] = grw_raw
        df_z['GROWTH_F_RAW'] = grw_raw
        df_z['GROWTH_F'] = robust_z_keepnan(grw_raw).clip(-3.0, 3.0)
        df_z['GRW_FLEX_WEIGHT'] = pd.to_numeric(df.get('GRW_FLEX_WEIGHT'), errors="coerce")
        df_z['GRW_FLEX_CORE_RAW'] = pd.to_numeric(df.get('GRW_FLEX_CORE'), errors="coerce")
        df_z['GRW_FLEX_PRICE_RAW'] = pd.to_numeric(df.get('GRW_FLEX_PRICE'), errors="coerce")

        # Debug dump for GRW composition (console OFF by default; enable only with env)
        if bool(os.getenv("GRW_CONSOLE_DEBUG")):
            try:
                cols = ['GROWTH_F', 'GROWTH_F_RAW', 'GRW_FLEX_WEIGHT']
                use_cols = [c for c in cols if c in df_z.columns]
                i = df_z[use_cols].copy() if use_cols else pd.DataFrame(index=df_z.index)
                i.sort_values('GROWTH_F', ascending=False, inplace=True)
                limit = max(0, min(40, len(i)))
                print("[DEBUG: GRW]")
                for t in i.index[:limit]:
                    row = i.loc[t]
                    parts = []
                    if pd.notna(row.get('GROWTH_F')):
                        parts.append(f"GROWTH_F={row.get('GROWTH_F'):.3f}")
                    raw_val = row.get('GROWTH_F_RAW')
                    if pd.notna(raw_val):
                        parts.append(f"GROWTH_F_RAW={raw_val:.3f}")
                    weight_val = row.get('GRW_FLEX_WEIGHT')
                    if pd.notna(weight_val):
                        parts.append(f"w={weight_val:.2f}")
                    path_val = None
                    try:
                        path_val = info.get(t, {}).get('DEBUG_GRW_PATH')
                    except Exception:
                        path_val = None
                    if not path_val and 'DEBUG_GRW_PATH' in df.columns:
                        path_val = df.at[t, 'DEBUG_GRW_PATH']
                    if path_val:
                        parts.append(f"PATH={path_val}")
                    parts_json = None
                    try:
                        parts_json = info.get(t, {}).get('DEBUG_GRW_PARTS')
                    except Exception:
                        parts_json = None
                    if not parts_json and 'DEBUG_GRW_PARTS' in df.columns:
                        parts_json = df.at[t, 'DEBUG_GRW_PARTS']
                    if parts_json:
                        parts.append(f"PARTS={parts_json}")
                    if not parts:
                        parts.append('no-data')
                    print(f"Ticker: {t} | " + " ".join(parts))
                print()
            except Exception as exc:
                print(f"[ERR] GRW debug dump failed: {exc}")

        df_z['MOM_F'] = robust_z(0.40*df_z['RS']
            + 0.15*df_z['TR_str']
            + 0.15*df_z['RS_SLOPE_6W']
            + 0.15*df_z['RS_SLOPE_13W']
            + 0.10*df_z['MA200_SLOPE_5M']
            + 0.10*df_z['MA200_UP_STREAK_D']).clip(-3.0,3.0)
        df_z['VOL'] = robust_z(df['BETA'])
        df_z['QAL'], df_z['YLD'], df_z['MOM'] = df_z['QUALITY_F'], df_z['YIELD_F'], df_z['MOM_F']
        df_z.drop(columns=['QUALITY_F','YIELD_F','MOM_F'], inplace=True, errors='ignore')

        _dump_dfz(
            df=df,
            df_z=df_z,
            debug_mode=getattr(cfg, "debug_mode", False),
        )
        if getattr(cfg, "debug_mode", False):
            log_grw_stats(df, df_z, logger)
        save_factor_debug_csv(df, df_z)

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
        g_score = g_score_all.loc[mask]
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

