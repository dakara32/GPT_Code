import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore

# ----- ユニバースと定数 -----
exist = [
    'AAPL','MRVL','NET','TTD','AMZN','NVDA','UBER',
    'TSM','AMAT','GOOGL','VEEV','ENB','APO','ICE',
    'TJX','CME','ACGL','VRT','WMB','SYY','TSLA','ABBV','CEG','CLS','IBKR'
]
cand = [
    'NKE','DOCS','DASH','U','GLW','GRBK','HON','GDDY','LRCX',
    'LVMUY','CPNG','PSX','PAC','JD','SPOT','BMO','FUJHY','NTR',
    'LTH','VZ','SIRI','LH','BN','GEHC','META','CASY','ELF',
    'SBUX','WELL','SYM','ARM','CYBR','PLTR','ASML','RXRX',
    'NXST','TMDX','GRAB','ANET','PSTG','RACE','OXY','YOU',
    'FI','GKOS','AMCR','SMCI','CRDO','QRVO','IBM','NVO',
    'AAOI','ALAB','IONQ','CRSP','HWM','UNP','SO',
    'VICI','BR','WM','KO'
]
tickers = sorted(set(exist + cand))
bench = '^GSPC'
N_G, N_D = 12, 13

# ----- データ取得 -----
data = yf.download(tickers + [bench], period='400d', auto_adjust=True, progress=False)
px = data['Close']
spx  = px[bench]
tickers_bulk = yf.Tickers(" ".join(tickers))
info = {t: tickers_bulk.tickers[t].info for t in tickers}

# ----- ファクター計算関数 -----
def trend(s):
    if len(s) < 252: return 0
    sma50 = s.rolling(50).mean().iloc[-1]
    sma150 = s.rolling(150).mean().iloc[-1]
    sma200 = s.rolling(200).mean().iloc[-1]
    prev200 = s.rolling(200).mean().iloc[-21]
    p = s.iloc[-1]
    hi, lo = s[-252:].max(), s[-252:].min()
    return int(all([p>sma50>sma150>sma200, sma150>sma200, sma200>prev200, p>0.75*hi, p>1.3*lo]))

def rs(s, b):
    r12 = s.iloc[-1]/s.iloc[-252] - 1
    r1 = s.iloc[-1]/s.iloc[-22] - 1
    br12 = b.iloc[-1]/b.iloc[-252] - 1
    br1 = b.iloc[-1]/b.iloc[-22] - 1
    return (r12-r1) - (br12-br1)

def tr_str(s):
    if len(s) < 50: return np.nan
    return s.iloc[-1]/s.rolling(50).mean().iloc[-1] - 1

def div_streak(t):
    try:
        divs = yf.Ticker(t).dividends.dropna()
        ann = divs.groupby(divs.index.year).sum()
        ann = ann[ann.index < pd.Timestamp.today().year]
        years = sorted(ann.index)
        streak = 0
        for i in range(len(years)-1,0,-1):
            if ann[years[i]]>ann[years[i-1]]: streak+=1
            else: break
        return streak
    except: return 0

# ----- ベースファクター計算 -----
df = pd.DataFrame(index=tickers)
for t in tickers:
    d = info[t]; s = px[t]; ev=d.get('enterpriseValue',np.nan)
    df.loc[t,'TR']=trend(s)
    df.loc[t,'EPS']=d.get('earningsQuarterlyGrowth',np.nan)
    df.loc[t,'REV']=d.get('revenueGrowth',np.nan)
    df.loc[t,'ROE']=d.get('returnOnEquity',np.nan)
    df.loc[t,'BETA']=d.get('beta',np.nan)
    df.loc[t,'DIV']=d.get('dividendYield') or d.get('trailingAnnualDividendYield') or 0
    df.loc[t,'FCF']=(d.get('freeCashflow',np.nan)/ev) if ev else np.nan
    df.loc[t,'RS']=rs(s,spx)
    df.loc[t,'TR_str']=tr_str(s)
    df.loc[t,'DIV_STREAK']=div_streak(t)

# ----- 正規化 (Zスコア) -----
z=lambda x: np.nan_to_num(zscore(x.fillna(x.mean())))
df_z=df.apply(z)
df_z['DIV']=z(df['DIV'])
df_z['TR_pm1']=df['TR'].replace({0:-1,1:1})
df_z['DIV_STREAK']=z(df['DIV_STREAK'])

# ----- 6ファクター合成 -----
df_z['GROWTH_F'] = 0.5 * df_z['REV'] + 0.3 * df_z['EPS'] + 0.2 * df_z['ROE']
df_z['MOM_F']    = 0.7 * df_z['RS'] + 0.3 * df_z['TR_str']
df_z['QUALITY_F'] = (df_z['FCF'] + df_z['ROE']) / 2
df_z['YIELD_F']   = 0.3 * df_z['DIV'] + 0.7 * df_z['DIV_STREAK']
df_z['VOL']       = df_z['BETA']
df_z['TREND']     = df_z['TR_pm1']

# ----- Compositeファクターの再標準化 -----
df_z['GROWTH_F']  = z(df_z['GROWTH_F'])
df_z['MOM_F']     = z(df_z['MOM_F'])
df_z['QUALITY_F'] = z(df_z['QUALITY_F'])
df_z['YIELD_F']   = z(df_z['YIELD_F'])
df_z['VOL']       = z(df_z['VOL'])

# ----- スコアリング -----
G_pool = df_z[df['TR'] == 1]
g_weights = {'GROWTH_F': 0.5, 'MOM_F': 0.5}
g_score = G_pool.mul(pd.Series(g_weights)).sum(axis=1)
top_G = g_score.nlargest(N_G).index

D_pool = df_z.drop(top_G)
D_weights = {'QUALITY_F': 0.25, 'YIELD_F': 0.35, 'VOL': -0.4}
d_score = D_pool.mul(pd.Series(D_weights)).sum(axis=1)
top_D = d_score.nlargest(N_D).index

# ----- 出力 -----
pd.set_option('display.float_format', '{:.3f}'.format)
# Growth枠
print("[G枠]")
print(pd.concat([
    df_z.loc[top_G, ['GROWTH_F','MOM_F','TREND']],
    g_score[top_G].rename('G_score')
], axis=1))
# Defense枠
print("[D枠]")
print(pd.concat([
    df_z.loc[top_D, ['QUALITY_F','YIELD_F','VOL']],
    d_score[top_D].rename('D_score')
], axis=1))
# IN / OUT
print("IN  :", sorted(set(list(top_G) + list(top_D)) - set(exist)))
print("OUT :", sorted(set(exist) - set(list(top_G) + list(top_D))))

# ----- パフォーマンス比較 -----
all_tickers = list(set(exist + list(top_G) + list(top_D) + [bench]))
prices = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False)['Close']
ret = prices.pct_change().dropna()
portfolios = {'Current': exist, 'New': list(top_G) + list(top_D)}
metrics = {}
for name, ticks in portfolios.items():
    pr = ret[ticks].mean(axis=1)
    cum = (1 + pr).cumprod() - 1
    ann_ret = (1 + cum.iloc[-1]) ** (252 / len(cum)) - 1
    ann_vol = pr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    drawdown = (cum - cum.cummax()).min()
    metrics[name] = {
        'Annual Return': ann_ret,
        'Annual Vol': ann_vol,
        'Sharpe': sharpe,
        'Max DD': drawdown
    }

df_metrics = pd.DataFrame(metrics).T
print("Performance Comparison:")
print(df_metrics.apply(lambda col: col.map(lambda x: f"{x:.2%}")))