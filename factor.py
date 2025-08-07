import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import requests

# ----- ユニバースと定数 -----
exist = pd.read_csv("current_tickers.csv", header=None)[0].tolist()
cand = pd.read_csv("candidate_tickers.csv", header=None)[0].tolist()
tickers = sorted(set(exist + cand))
bench = '^GSPC'
N_G, N_D = 12, 13
g_weights = {'GRW': 0.3, 'MOM': 0.7}
D_weights = {'QAL': 0.2, 'YLD': 0.4, 'VOL': -0.4}
corr_thresh_G = 0.5   # Growth側
corr_thresh_D = 0.5   # Defense側
corrM = 45

# ----- データ取得 -----
data = yf.download(tickers + [bench], period='400d', auto_adjust=True, progress=False)
px = data['Close']
spx = px[bench]
tickers_bulk = yf.Tickers(" ".join(tickers))
info = {t: tickers_bulk.tickers[t].info for t in tickers}

# ----- 相関行列 ----
returns = px[tickers].pct_change().dropna()
corr = returns.corr()

# ----- ファクター計算関数 -----
def trend(s):
    """移動平均線と52週レンジで強い上昇トレンドを判定。
    全条件を満たせば1、そうでなければ0を返す。"""
    if len(s) < 252:
        return 0
    sma50 = s.rolling(50).mean().iloc[-1]
    sma150 = s.rolling(150).mean().iloc[-1]
    sma200 = s.rolling(200).mean().iloc[-1]
    prev200 = s.rolling(200).mean().iloc[-21]
    p = s.iloc[-1]
    hi, lo = s[-252:].max(), s[-252:].min()
    return int(all([p > sma50 > sma150 > sma200, sma150 > sma200, sma200 > prev200, p > 0.75 * hi, p > 1.3 * lo]))


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


# ----- ベースファクター計算 -----
df = pd.DataFrame(index=tickers)
for t in tickers:
    d = info[t]
    s = px[t]
    ev = d.get('enterpriseValue', np.nan)
    df.loc[t, 'TR'] = trend(s)
    df.loc[t, 'EPS'] = d.get('earningsQuarterlyGrowth', np.nan)
    df.loc[t, 'REV'] = d.get('revenueGrowth', np.nan)
    df.loc[t, 'ROE'] = d.get('returnOnEquity', np.nan)
    df.loc[t, 'BETA'] = d.get('beta', np.nan)
    df.loc[t, 'DIV'] = d.get('dividendYield') or d.get('trailingAnnualDividendYield') or 0
    df.loc[t, 'FCF'] = (d.get('freeCashflow', np.nan) / ev) if ev else np.nan
    df.loc[t, 'RS'] = rs(s, spx)
    df.loc[t, 'TR_str'] = tr_str(s)
    df.loc[t, 'DIV_STREAK'] = div_streak(t)


# ----- 正規化 (Zスコア) -----
z = lambda x: np.nan_to_num(zscore(x.fillna(x.mean())))
df_z = df.apply(z)
df_z['DIV'] = z(df['DIV'])
df_z['TR_pm1'] = df['TR'].replace({0: -1, 1: 1})
df_z['DIV_STREAK'] = z(df['DIV_STREAK'])


# ----- 6ファクター合成 -----
df_z['GROWTH_F'] = 0.5 * df_z['REV'] + 0.3 * df_z['EPS'] + 0.2 * df_z['ROE']
df_z['MOM_F'] = 0.7 * df_z['RS'] + 0.3 * df_z['TR_str']
df_z['QUALITY_F'] = (df_z['FCF'] + df_z['ROE']) / 2
df_z['YIELD_F'] = 0.3 * df_z['DIV'] + 0.7 * df_z['DIV_STREAK']
df_z['VOL'] = df_z['BETA']
df_z['TREND'] = df_z['TR_pm1']


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
G_pool = df_z[df['TR'] == 1]
g_score = G_pool.mul(pd.Series(g_weights)).sum(axis=1)

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
d_score = D_pool.mul(pd.Series(D_weights)).sum(axis=1)
init_D = d_score.nlargest(corrM).index.tolist()
thresh_D = corr_thresh_D
chosen_D = greedy_select(init_D, corr, N_D, thresh_D)
while len(chosen_D) < N_D and thresh_D < corr_thresh_D:
    thresh_D += 0.02
    chosen_D = greedy_select(init_D, corr, N_D, thresh_D)
top_D = chosen_D


# ----- 出力 -----
pd.set_option('display.float_format', '{:.3f}'.format)
# Growth枠
g_table = pd.concat([
    df_z.loc[top_G, ['GRW', 'MOM', 'TRD']],
    g_score[top_G].rename('GSC')
], axis=1)
print("[G枠]")
print(g_table)
# Defense枠
d_table = pd.concat([
    df_z.loc[top_D, ['QAL', 'YLD', 'VOL']],
    d_score[top_D].rename('DSC')
], axis=1)
print("[D枠]")
print(d_table)
# IN / OUT
in_list = sorted(set(list(top_G) + list(top_D)) - set(exist))
out_list = sorted(set(exist) - set(list(top_G) + list(top_D)))
io_table = pd.DataFrame({'IN': pd.Series(in_list), 'OUT': pd.Series(out_list)})
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
df_metrics_fmt = df_metrics.apply(lambda col: col.map(lambda x: f"{x:.2%}"))
print("Performance Comparison:")
print(df_metrics_fmt)


# ----- Slack送信 -----
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    raise ValueError("SLACK_WEBHOOK_URL not set (環境変数が未設定です)")

message = (
    "ファクター分散最適化の結果\n"
    "[G枠]\n```" + g_table.to_string() + "```\n"
    "[D枠]\n```" + d_table.to_string() + "```\n"
    "Changes\n```" + io_table.to_string(index=False) + "```\n"
    "Performance Comparison:\n```" + df_metrics_fmt.to_string() + "```"
)

payload = {"text": message}
try:
    resp = requests.post(SLACK_WEBHOOK_URL, json=payload)
    resp.raise_for_status()
    print("✅ Slack（Webhook）へ送信しました")
except Exception as e:
    print(f"⚠️ Slack通知エラー: {e}")
