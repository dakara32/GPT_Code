import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import requests
import time

# ========= EPS補完 & FCF算出ユーティリティ =========

def impute_eps_ttm(df: pd.DataFrame,
                   ttm_col: str = "eps_ttm",
                   q_col: str = "eps_q_recent",
                   out_col: str | None = None) -> pd.DataFrame:
    if out_col is None:
        out_col = ttm_col
    df = df.copy()
    df["eps_imputed"] = False
    cand = df[q_col] * 4
    ok = df[ttm_col].isna() & cand.replace([np.inf, -np.inf], np.nan).notna()
    df.loc[ok, out_col] = cand[ok]
    df.loc[ok, "eps_imputed"] = True
    return df

_CF_ALIASES = {
    "cfo": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "Capital Expenditures",
    ],
}

def _pick_row(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for name in names:
        key = name.lower()
        if key in idx_lower:
            return df.loc[idx_lower[key]]
    return None

def _sum_last_n(s: pd.Series | None, n: int) -> float | None:
    if s is None or s.empty:
        return None
    vals = s.dropna().astype(float)
    if vals.empty:
        return None
    return vals.iloc[:n].sum()

def _latest(s: pd.Series | None) -> float | None:
    if s is None or s.empty:
        return None
    vals = s.dropna().astype(float)
    return vals.iloc[0] if not vals.empty else None

def fetch_cfo_capex_ttm_yf(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        qcf = tk.quarterly_cashflow
        cfo_q = _pick_row(qcf, _CF_ALIASES["cfo"])
        capex_q = _pick_row(qcf, _CF_ALIASES["capex"])
        cfo_ttm = _sum_last_n(cfo_q, 4)
        capex_ttm = _sum_last_n(capex_q, 4)

        if cfo_ttm is None or capex_ttm is None:
            acf = tk.cashflow
            cfo_a = _pick_row(acf, _CF_ALIASES["cfo"])
            capex_a = _pick_row(acf, _CF_ALIASES["capex"])
            if cfo_ttm is None:
                cfo_ttm = _latest(cfo_a)
            if capex_ttm is None:
                capex_ttm = _latest(capex_a)

        rows.append({
            "ticker": t,
            "cfo_ttm_yf": cfo_ttm if cfo_ttm is not None else np.nan,
            "capex_ttm_yf": capex_ttm if capex_ttm is not None else np.nan,
        })
    return pd.DataFrame(rows).set_index("ticker")

_FINN_CFO_KEYS = [
    "netCashProvidedByOperatingActivities",
    "netCashFromOperatingActivities",
    "cashFlowFromOperatingActivities",
    "operatingCashFlow",
]
_FINN_CAPEX_KEYS = [
    "capitalExpenditure",
    "capitalExpenditures",
    "purchaseOfPPE",
    "investmentsInPropertyPlantAndEquipment",
]

def _first_key(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _finn_get(session: requests.Session, url: str, params: dict,
              retries: int = 3, sleep_s: float = 0.5):
    for i in range(retries):
        r = session.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(min(2**i * sleep_s, 4.0))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def fetch_cfo_capex_ttm_finnhub(tickers: list[str], api_key: str | None = None) -> pd.DataFrame:
    api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Finnhub API key not provided. Set FINNHUB_API_KEY or pass api_key=")

    base = "https://finnhub.io/api/v1"
    s = requests.Session()
    rows = []
    for sym in tickers:
        cfo_ttm = None
        capex_ttm = None
        try:
            j = _finn_get(s, f"{base}/stock/cash-flow", {
                "symbol": sym, "frequency": "quarterly", "limit": 8, "token": api_key
            })
            arr = j.get("cashFlow") or []
            cfo_vals, capex_vals = [], []
            for item in arr[:4]:
                cfo_vals.append(_first_key(item, _FINN_CFO_KEYS))
                capex_vals.append(_first_key(item, _FINN_CAPEX_KEYS))
            if any(v is not None for v in cfo_vals):
                cfo_ttm = float(np.nansum([np.nan if v is None else float(v) for v in cfo_vals]))
            if any(v is not None for v in capex_vals):
                capex_ttm = float(np.nansum([np.nan if v is None else float(v) for v in capex_vals]))
        except Exception:
            pass
        if cfo_ttm is None or capex_ttm is None:
            try:
                j = _finn_get(s, f"{base}/stock/cash-flow", {
                    "symbol": sym, "frequency": "annual", "limit": 1, "token": api_key
                })
                arr = j.get("cashFlow") or []
                if arr:
                    item0 = arr[0]
                    if cfo_ttm is None:
                        v = _first_key(item0, _FINN_CFO_KEYS)
                        if v is not None:
                            cfo_ttm = float(v)
                    if capex_ttm is None:
                        v = _first_key(item0, _FINN_CAPEX_KEYS)
                        if v is not None:
                            capex_ttm = float(v)
            except Exception:
                pass

        rows.append({
            "ticker": sym,
            "cfo_ttm_fh": np.nan if cfo_ttm is None else cfo_ttm,
            "capex_ttm_fh": np.nan if capex_ttm is None else capex_ttm,
        })
    return pd.DataFrame(rows).set_index("ticker")

def compute_fcf_with_fallback(tickers: list[str], finnhub_api_key: str | None = None) -> pd.DataFrame:
    yf_df = fetch_cfo_capex_ttm_yf(tickers)
    fh_df = fetch_cfo_capex_ttm_finnhub(tickers, api_key=finnhub_api_key)
    df = yf_df.join(fh_df, how="outer")
    df["cfo_ttm"] = df["cfo_ttm_yf"].where(df["cfo_ttm_yf"].notna(), df["cfo_ttm_fh"])
    df["capex_ttm"] = df["capex_ttm_yf"].where(df["capex_ttm_yf"].notna(), df["capex_ttm_fh"])

    df["cfo_source"] = pd.Series(index=df.index, dtype="object")
    df.loc[df["cfo_ttm_yf"].notna(), "cfo_source"] = "yfinance"
    df.loc[df["cfo_ttm_yf"].isna() & df["cfo_ttm_fh"].notna(), "cfo_source"] = "finnhub"

    df["capex_source"] = pd.Series(index=df.index, dtype="object")
    df.loc[df["capex_ttm_yf"].notna(), "capex_source"] = "yfinance"
    df.loc[df["capex_ttm_yf"].isna() & df["capex_ttm_fh"].notna(), "capex_source"] = "finnhub"

    cfo = pd.to_numeric(df["cfo_ttm"], errors="coerce")
    capex = pd.to_numeric(df["capex_ttm"], errors="coerce").abs()
    df["fcf_ttm"] = cfo - capex
    df["fcf_imputed"] = df[["cfo_ttm_yf", "capex_ttm_yf"]].isna().any(axis=1) & df[["cfo_ttm", "capex_ttm"]].notna().all(axis=1)
    cols = ["cfo_ttm_yf", "capex_ttm_yf", "cfo_ttm_fh", "capex_ttm_fh",
            "cfo_ttm", "capex_ttm", "fcf_ttm", "cfo_source", "capex_source", "fcf_imputed"]
    return df[cols].sort_index()

# ----- ユニバースと定数 -----
exist = pd.read_csv("current_tickers.csv", header=None)[0].tolist()
cand = pd.read_csv("candidate_tickers.csv", header=None)[0].tolist()
# 候補銘柄の価格上限（調整可能）
CAND_PRICE_MAX = 400
# ベンチマークsp500
bench = '^GSPC'
# G枠とD枠の保持数
N_G, N_D = 12, 13
# 枠別のファクター重み
g_weights = {'GRW': 0.2, 'MOM': 0.35, 'TRD': 0.45}
D_weights = {'QAL': 0.15, 'YLD': 0.35, 'VOL': -0.5}
# 貪欲選定の相関閾値（0.3 → 強く分散、0.5 → 適度に分散、0.8 → ほぼ分散条件なし）
corr_thresh_G = 0.45   # Growth側
corr_thresh_D = 0.45   # Defense側
corrM = 45
# デバッグモード（Trueで詳細情報を表示）
debug_mode = False
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# ----- データ取得 -----
cand_info = yf.Tickers(" ".join(cand))
cand_prices = {
    t: cand_info.tickers[t].fast_info.get('lastPrice', np.inf)
    for t in cand
}
cand = [t for t, p in cand_prices.items() if p <= CAND_PRICE_MAX]
tickers = sorted(set(exist + cand))
data = yf.download(tickers + [bench], period='400d', auto_adjust=True, progress=False)
px = data['Close']
spx = px[bench]
tickers_bulk = yf.Tickers(" ".join(tickers))
info = {t: tickers_bulk.tickers[t].info for t in tickers}

# EPSとFCFの補完データを用意
eps_rows = []
for t in tickers:
    info_t = info[t]
    eps_ttm = info_t.get("trailingEps", np.nan)
    eps_q = np.nan
    try:
        qearn = tickers_bulk.tickers[t].quarterly_earnings
        so = info_t.get("sharesOutstanding")
        if so and qearn is not None and not qearn.empty and "Earnings" in qearn.columns:
            eps_q = qearn["Earnings"].iloc[-1] / so
    except Exception:
        pass
    eps_rows.append({"ticker": t, "eps_ttm": eps_ttm, "eps_q_recent": eps_q})
eps_df = pd.DataFrame(eps_rows).set_index("ticker")
eps_df = impute_eps_ttm(eps_df, ttm_col="eps_ttm", q_col="eps_q_recent")

fcf_df = compute_fcf_with_fallback(tickers, finnhub_api_key=FINNHUB_API_KEY)

# ----- 相関行列 ----
returns = px[tickers].pct_change().dropna()
corr = returns.corr()

# ----- ファクター計算関数 -----
def trend(s):
    """移動平均線と52週レンジで強い上昇トレンドを判定。
    全条件を満たせば1、そうでなければ-1を返す。"""
    if len(s) < 252:
        return -1
    sma50 = s.rolling(50).mean().iloc[-1]
    sma150 = s.rolling(150).mean().iloc[-1]
    sma200 = s.rolling(200).mean().iloc[-1]
    prev200 = s.rolling(200).mean().iloc[-21]
    p = s.iloc[-1]
    hi, lo = s[-252:].max(), s[-252:].min()
    return 1 if all([p > sma50 > sma150 > sma200, sma150 > sma200, sma200 > prev200, p > 0.75 * hi, p > 1.3 * lo]) else -1


def rs(s, b):
    """12ヶ月と1ヶ月のリターンからベンチマークに対する相対強度を算出。
    正の値はベンチマーク超過を示す。"""
    r12 = s.iloc[-1] / s.iloc[-252] - 1
    r1 = s.iloc[-1] / s.iloc[-22] - 1
    br12 = b.iloc[-1] / b.iloc[-252] - 1
    br1 = b.iloc[-1] / b.iloc[-22] - 1
    return (r12 - r1) - (br12 - br1)


def tr_str(s):
    """終値が50日移動平均からどれだけ乖離しているかで短期トレンドの強さを測定。"""
    if len(s) < 50:
        return np.nan
    return s.iloc[-1] / s.rolling(50).mean().iloc[-1] - 1


def dividend_status(ticker: str) -> str:
    """銘柄の配当状況を簡易判定する。

    has            : 配当イベントが存在（=配当あり）
    none_confident : 分割イベントは見えるが配当はなし（=無配と判断）
    maybe_missing  : fast_info に配当痕跡がありデータ欠損の可能性
    unknown        : 情報が得られない
    """
    t = yf.Ticker(ticker)
    # 1) 配当イベントがあれば配当あり
    if not t.dividends.empty:
        return "has"

    # 2) 分割イベントがあればフィードは生きている → 無配と判断
    a = t.actions
    if (
        a is not None
        and not a.empty
        and "Stock Splits" in a.columns
        and a["Stock Splits"].abs().sum() > 0
    ):
        return "none_confident"

    # 3) fast_info に配当の痕跡があれば取りこぼし疑い
    fi = t.fast_info
    if any(getattr(fi, k, None) for k in ("last_dividend_date", "dividend_rate", "dividend_yield")):
        return "maybe_missing"

    # 4) それ以外は情報不足
    return "unknown"


def div_streak(t):
    """企業が何年連続で配当を増やしているかを求める。"""
    try:
        divs = yf.Ticker(t).dividends.dropna()
        ann = divs.groupby(divs.index.year).sum()
        ann = ann[ann.index < pd.Timestamp.today().year]
        years = sorted(ann.index)
        streak = 0
        for i in range(len(years) - 1, 0, -1):
            if ann[years[i]] > ann[years[i - 1]]:
                streak += 1
            else:
                break
        return streak
    except Exception:
        return 0


def fetch_finnhub_metrics(symbol):
    """finnhub API から不足データを取得"""
    if not FINNHUB_API_KEY:
        return {}
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": symbol, "metric": "all", "token": FINNHUB_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        m = r.json().get("metric", {})
        return {
            'EPS': m.get('epsGrowthTTMYoy'),
            'REV': m.get('revenueGrowthTTMYoy'),
            'ROE': m.get('roeTTM'),
            'BETA': m.get('beta'),
            'DIV': m.get('dividendYieldIndicatedAnnual'),
            'FCF': (m.get('freeCashFlowTTM') / m.get('enterpriseValue')) if m.get('freeCashFlowTTM') and m.get('enterpriseValue') else None,
        }
    except Exception:
        return {}


# ----- ベースファクター計算 -----
df = pd.DataFrame(index=tickers)
missing_logs = []
for t in tickers:
    d = info[t]
    s = px[t]
    ev = d.get('enterpriseValue', np.nan)
    df.loc[t, 'TR'] = trend(s)
    df.loc[t, 'EPS'] = eps_df.loc[t, 'eps_ttm']
    df.loc[t, 'REV'] = d.get('revenueGrowth', np.nan)
    df.loc[t, 'ROE'] = d.get('returnOnEquity', np.nan)
    df.loc[t, 'BETA'] = d.get('beta', np.nan)
    div = d.get('dividendYield')
    if div is None or pd.isna(div):
        div = d.get('trailingAnnualDividendYield')
    df.loc[t, 'DIV'] = div if div is not None else np.nan
    fcf_val = fcf_df.loc[t, 'fcf_ttm'] if t in fcf_df.index else np.nan
    df.loc[t, 'FCF'] = (fcf_val / ev) if ev and not pd.isna(fcf_val) else np.nan
    df.loc[t, 'RS'] = rs(s, spx)
    df.loc[t, 'TR_str'] = tr_str(s)
    df.loc[t, 'DIV_STREAK'] = div_streak(t)
    fin_cols = ['REV', 'ROE', 'BETA', 'DIV', 'FCF']
    need_finnhub = [col for col in fin_cols if pd.isna(df.loc[t, col])]
    if need_finnhub:
        fin_data = fetch_finnhub_metrics(t)
        for col in need_finnhub:
            val = fin_data.get(col)
            if val is not None and not pd.isna(val):
                df.loc[t, col] = val
    for col in fin_cols + ['EPS', 'RS', 'TR_str', 'DIV_STREAK']:
        if pd.isna(df.loc[t, col]):
            if col == 'DIV':
                status = dividend_status(t)
                if status != 'none_confident':
                    missing_logs.append({'Ticker': t, 'Column': col, 'Status': status})
            else:
                missing_logs.append({'Ticker': t, 'Column': col})


# ----- 正規化 (Zスコア) -----
z = lambda x: np.nan_to_num(zscore(x.fillna(x.mean())))
df_z = df.apply(z)
df_z['DIV'] = z(df['DIV'])
df_z['TR'] = df['TR']
df_z['DIV_STREAK'] = z(df['DIV_STREAK'])


# ----- 6ファクター合成 -----
df_z['GROWTH_F'] = 0.5 * df_z['REV'] + 0.3 * df_z['EPS'] + 0.2 * df_z['ROE']
df_z['MOM_F'] = 0.7 * df_z['RS'] + 0.3 * df_z['TR_str']
df_z['QUALITY_F'] = (df_z['FCF'] + df_z['ROE']) / 2
df_z['YIELD_F'] = 0.3 * df_z['DIV'] + 0.7 * df_z['DIV_STREAK']
df_z['VOL'] = df_z['BETA']
df_z['TREND'] = df_z['TR']


# ----- Compositeファクターの再標準化 -----
df_z['GROWTH_F'] = z(df_z['GROWTH_F'])
df_z['MOM_F'] = z(df_z['MOM_F'])
df_z['QUALITY_F'] = z(df_z['QUALITY_F'])
df_z['YIELD_F'] = z(df_z['YIELD_F'])
df_z['VOL'] = z(df_z['VOL'])

# ----- カラム名を短縮 -----
df_z.rename(columns={
    'GROWTH_F': 'GRW',
    'MOM_F': 'MOM',
    'TREND': 'TRD',
    'QUALITY_F': 'QAL',
    'YIELD_F': 'YLD',
    'VOL': 'VOL'
}, inplace=True)


# ----- スコアリング -----
g_score = df_z.mul(pd.Series(g_weights)).sum(axis=1)
d_score_all = df_z.mul(pd.Series(D_weights)).sum(axis=1)

# 相関抑制ロジック
def greedy_select(candidates, corr, target_n, thresh):
    selected = []
    for t in candidates:
        if all(abs(corr.loc[t, s]) < thresh for s in selected):
            selected.append(t)
        if len(selected) == target_n:
            break
    return selected

init_G = g_score.nlargest(corrM).index.tolist()
thresh_G = corr_thresh_G
chosen_G = greedy_select(init_G, corr, N_G, thresh_G)
while len(chosen_G) < N_G and thresh_G < corr_thresh_G:
    thresh_G += 0.02
    chosen_G = greedy_select(init_G, corr, N_G, thresh_G)
top_G = chosen_G

D_pool = df_z.drop(top_G)
d_score = d_score_all.drop(top_G)
init_D = d_score.nlargest(corrM).index.tolist()
thresh_D = corr_thresh_D
chosen_D = greedy_select(init_D, corr, N_D, thresh_D)
while len(chosen_D) < N_D and thresh_D < corr_thresh_D:
    thresh_D += 0.02
    chosen_D = greedy_select(init_D, corr, N_D, thresh_D)
top_D = chosen_D


# ----- 出力 -----
pd.set_option('display.float_format', '{:.3f}'.format)
print("📈 ファクター分散最適化の結果")
miss_df = pd.DataFrame(missing_logs)
if not miss_df.empty:
    print("Missing Data:")
    print(miss_df.to_string(index=False))

extra_G = [t for t in init_G if t not in top_G][:5]
G_UNI = top_G + extra_G
g_table = pd.concat([
    df_z.loc[G_UNI, ['GRW', 'MOM', 'TRD']],
    g_score[G_UNI].rename('GSC')
], axis=1)
g_table.index = [t + ("⭐️" if t in top_G else "") for t in G_UNI]
g_title = f"[G枠 / {N_G} / GRW{int(g_weights['GRW']*100)} MOM{int(g_weights['MOM']*100)} TRD{int(g_weights['TRD']*100)} / corr{int(corr_thresh_G*100)}]"
print(g_title)
print(g_table)

extra_D = [t for t in init_D if t not in top_D][:5]
D_UNI = top_D + extra_D
d_table = pd.concat([
    df_z.loc[D_UNI, ['QAL', 'YLD', 'VOL']],
    d_score_all[D_UNI].rename('DSC')
], axis=1)
d_table.index = [t + ("⭐️" if t in top_D else "") for t in D_UNI]
d_title = f"[D枠 / {N_D} / QAL{int(D_weights['QAL']*100)} YLD{int(D_weights['YLD']*100)} VOL{int(D_weights['VOL']*100)} / corr{int(corr_thresh_D*100)}]"
print(d_title)
print(d_table)

in_list = sorted(set(list(top_G) + list(top_D)) - set(exist))
out_list = sorted(set(exist) - set(list(top_G) + list(top_D)))

# IN/OUTの組み合わせがずれないようにインデックスをリセットして連結
in_df = pd.DataFrame({'IN': in_list}).reset_index(drop=True)
out_df = pd.DataFrame({
    '/ OUT': out_list,
    'GSC': g_score.reindex(out_list).round(3).to_list(),
    'DSC': d_score_all.reindex(out_list).round(3).to_list()
}).reset_index(drop=True)
io_table = pd.concat([in_df, out_df], axis=1)
print("Changes:")
print(io_table.to_string(index=False))


# ----- パフォーマンス比較 -----
all_tickers = list(set(exist + list(top_G) + list(top_D) + [bench]))
prices = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False)['Close']
ret = prices.pct_change().dropna()
portfolios = {'CUR': exist, 'NEW': list(top_G) + list(top_D)}
metrics = {}
for name, ticks in portfolios.items():
    pr = ret[ticks].mean(axis=1)
    cum = (1 + pr).cumprod() - 1
    ann_ret = (1 + cum.iloc[-1]) ** (252 / len(cum)) - 1
    ann_vol = pr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    drawdown = (cum - cum.cummax()).min()
    metrics[name] = {
        'RET': ann_ret,
        'VOL': ann_vol,
        'SHP': sharpe,
        'MDD': drawdown
    }

df_metrics = pd.DataFrame(metrics).T
df_metrics_pct = df_metrics.copy()
for col in ['RET', 'VOL', 'MDD']:
    df_metrics_pct[col] = df_metrics_pct[col] * 100
df_metrics_pct = df_metrics_pct.rename(columns={'RET': 'RET%', 'VOL': 'VOL%', 'MDD': 'MDD%'})
df_metrics_fmt = df_metrics_pct.applymap(lambda x: f"{x:.1f}")
print("Performance Comparison:")
print(df_metrics_fmt)

debug_table = None
if debug_mode:
    debug_table = pd.concat([
        df[['TR', 'EPS', 'REV', 'ROE', 'BETA', 'DIV', 'FCF', 'RS', 'TR_str', 'DIV_STREAK']],
        g_score.rename('GSC'),
        d_score_all.rename('DSC')
    ], axis=1).round(3)
    print("Debug Data:")
    print(debug_table.to_string())


# ----- Slack送信 -----
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")

message = "📈 ファクター分散最適化の結果\n"
if not miss_df.empty:
    message += "Missing Data\n```" + miss_df.to_string(index=False) + "```\n"
message += g_title + "\n```" + g_table.to_string() + "```\n"
message += d_title + "\n```" + d_table.to_string() + "```\n"
message += "Changes\n```" + io_table.to_string(index=False) + "```\n"
message += "Performance Comparison:\n```" + df_metrics_fmt.to_string() + "```"
if debug_mode and debug_table is not None:
    message += "\nDebug Data\n```" + debug_table.to_string() + "```"

payload = {"text": message}
try:
    resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
    resp.raise_for_status()
    print("✅ Slack（Webhook）へ送信しました")
except Exception as e:
    print(f"⚠️ Slack通知エラー: {e}")
