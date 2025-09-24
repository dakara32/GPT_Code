#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_health_probe.py — 選定プログラム依存API（Yahoo Finance / SEC / Finnhub）の総合ヘルスチェック
Usage:
  export SLACK_WEBHOOK_URL=...
  export FINNHUB_API_KEY=...            # 任意（無ければ FinnhubはSKIPPED）
  export SEC_EMAIL=you@example.com      # 推奨（SEC User-Agent に使用）
  python tools/api_health_probe.py
Env (optional):
  CSV_CURRENT=./current.csv
  CSV_CANDIDATE=./candidate.csv
  YF_PERIOD=1y
  YF_MIN_LEN=120
  TIMEOUT_MS_WARN=5000
  MAX_WORKERS=8
  SOFT_FAIL=0   # 1なら常にexit 0
Exit codes:
  HEALTHY=0, DEGRADED=10, DOWN=20 （SOFT_FAIL=1なら常に0）
"""
import os, sys, time, json, math, csv, re, concurrent.futures as cf
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ---- Settings
CSV_CURRENT = os.getenv("CSV_CURRENT","./current.csv")
CSV_CANDIDATE= os.getenv("CSV_CANDIDATE","./candidate.csv")
YF_PERIOD   = os.getenv("YF_PERIOD","1y")
YF_MIN_LEN  = int(os.getenv("YF_MIN_LEN","120"))
TIMEOUT_MS_WARN = int(os.getenv("TIMEOUT_MS_WARN","5000"))
SOFT_FAIL   = os.getenv("SOFT_FAIL","0") == "1"
FINN_KEY    = os.getenv("FINNHUB_API_KEY")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK")
SEC_EMAIL   = os.getenv("SEC_EMAIL","")
MAX_WORKERS = int(os.getenv("MAX_WORKERS","8"))
# “任意API”の扱い：ここに列挙されたAPIがDOWNでも全体は最大DEGRADED止まり
OPTIONAL_APIS = set([x.strip().upper() for x in os.getenv("OPTIONAL_APIS","FINNHUB").split(",") if x.strip()])
# 退出条件（既定: DEGRADED）。DOWNにすれば「DOWNの時だけ失敗」
EXIT_ON_LEVEL = os.getenv("EXIT_ON_LEVEL","DEGRADED").upper()

# ---- Utils
def _now_ms() -> int: return int(time.time()*1000)

def _post_slack(text: str):
    if not SLACK_WEBHOOK:
        print("[SLACK] webhook missing; print only\n"+text); return
    try:
        r = requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=5)
        print(f"[SLACK] status={r.status_code}"); r.raise_for_status()
    except Exception as e: print(f"[SLACK] send error: {e}")

def _read_tickers(path: str) -> List[str]:
    if not os.path.exists(path): return []
    # 'ticker','symbol','Symbol','Ticker' の列に対応。無ければ1列CSVも許容。
    try:
        df = pd.read_csv(path)
        for c in ["ticker","symbol","Symbol","Ticker"]:
            if c in df.columns:
                col = df[c].astype(str).str.strip()
                return [t for t in col if t and t.lower()!="nan"]
        with open(path, newline="") as f:
            rd = csv.reader(f)
            vals = [row[0].strip() for row in rd if row]
            if vals and vals[0].lower() in ("ticker","symbol"): vals = vals[1:]
            return [v for v in vals if v]
    except Exception:
        return []

def _autodiscover_csv() -> tuple[str|None, str|None]:
    """
    リポジトリ内から current*.csv / candidate*.csv を再帰探索し、最初に見つけたものを返す。
    明示指定（ENV）があればそれを優先。見つからなければ None。
    """
    cur = CSV_CURRENT if os.path.exists(CSV_CURRENT) else None
    cand = CSV_CANDIDATE if os.path.exists(CSV_CANDIDATE) else None
    if cur and cand:
        return cur, cand

    for root, _, files in os.walk(".", topdown=True):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(root, fn)
            name = fn.lower()
            if not cur and "current" in name:
                cur = path
            if not cand and "candidate" in name:
                cand = path
        if cur and cand:
            break
    return cur, cand

def _fmt_ms(ms: int) -> str:
    return f"{ms}ms" if ms < 1000 else f"{ms/1000:.2f}s"

# ---- Ticker 正規化（YF用）
def _yf_variants(sym: str):
    s = (sym or "").upper()
    cands = []
    def add(x):
        if x and x not in cands: cands.append(x)
    add(s)
    add(s.replace(".","-"))   # BRK.B -> BRK-B, PBR.A -> PBR-A
    add(re.sub(r"[.\-^]", "", s))  # 記号除去
    return cands

# ================================================================
# Yahoo Finance: price series ヘルス
# ================================================================
def yf_price_health(tickers: List[str]) -> Tuple[str, Dict]:
    t0 = _now_ms()
    data = yf.download(tickers, period=YF_PERIOD, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data else pd.DataFrame()
    per_ticker_missing = {}
    nf=[]          # 一括でも別名再試行でも取得できず
    missing=[]     # 列はあるがNaN/不足
    ok=[]          # 問題なし
    alias_fixed=[] # (orig, alias) 別名で回復
    for t in tickers:
        if t not in close.columns:
            # 簡易ノーマライズ後、個別で5dだけ再取得して最低限の生存確認
            recovered = False
            for alias in _yf_variants(t):
                try:
                    s = yf.Ticker(alias).history(period="5d", auto_adjust=True)["Close"]
                    if isinstance(s, pd.Series) and s.notna().sum() > 0:
                        recovered = True
                        alias_fixed.append((t, alias))
                        break
                except Exception:
                    pass
            if not recovered:
                nf.append(t); per_ticker_missing[t]={"dates":set(),"max_gap":0}; continue
            # 再取得で回復した場合はOK扱い（dates/max_gapは空のまま）
            ok.append(t); per_ticker_missing.setdefault(t, {"dates":set(),"max_gap":0}); continue
        s = close[t]; n = s.shape[0]; nn = int(s.notna().sum())
        isna = s.isna().values; idx = s.index
        total_nan = int(isna.sum()); cur=max_gap=0
        dates = set(str(d.date()) for d,v in zip(idx,isna) if v)
        for v in isna:
            if v: cur+=1
            else:
                if cur>0: max_gap=max(max_gap,cur); cur=0
        if cur>0: max_gap=max(max_gap,cur)
        per_ticker_missing[t] = {"dates":dates,"max_gap":max_gap}
        if nn==0 or total_nan>0 or n<YF_MIN_LEN: missing.append(t)
        else: ok.append(t)
    ms = _now_ms()-t0
    level = "HEALTHY" if len(ok)==len(tickers) else ("DEGRADED" if len(ok)>=len(tickers)//2 else "DOWN")
    slow = " SLOW" if ms>=TIMEOUT_MS_WARN else ""
    return f"YF_PRICE:{level} ok={len(ok)}/{len(tickers)} latency={_fmt_ms(ms)}{slow}", {
        "level":level,"latency_ms":ms,"ok":ok,"nf":nf,"missing":missing,
        "per_ticker_missing":per_ticker_missing,
        "alias_fixed": alias_fixed
    }

# ================================================================
# Yahoo Finance: fast_info.lastPrice ヘルス
# ================================================================
def yf_fastinfo_health(tickers: List[str]) -> Tuple[str, Dict]:
    t0 = _now_ms(); tk = yf.Tickers(" ".join(tickers)); bad=[]
    for t in tickers:
        try:
            v = tk.tickers[t].fast_info.get("lastPrice", None)
            if v is None or (isinstance(v,float) and math.isnan(v)): bad.append(t)
        except Exception: bad.append(t)
    ms=_now_ms()-t0
    level = "HEALTHY" if not bad else ("DEGRADED" if len(bad)<=len(tickers)//2 else "DOWN")
    slow = " SLOW" if ms>=TIMEOUT_MS_WARN else ""
    return f"YF_INFO:{level} bad={len(bad)}/{len(tickers)} latency={_fmt_ms(ms)}{slow}", {
        "level":level,"latency_ms":ms,"bad":bad
    }

# ================================================================
# Yahoo Finance: financials（CFO/Capex/FCF）ヘルス
# ================================================================
_CF_ALIASES = {"cfo":["Operating Cash Flow","Total Cash From Operating Activities"],
               "capex":["Capital Expenditure","Capital Expenditures"]}
def _pick_row(df: pd.DataFrame, names: List[str]) -> pd.Series|None:
    if df is None or df.empty: return None
    idx_lower = {str(i).lower():i for i in df.index}
    for n in names:
        k = n.lower()
        if k in idx_lower: return df.loc[idx_lower[k]]
    return None
def _sum_last_n(s: pd.Series|None, n:int) -> float|None:
    if s is None or s.empty: return None
    v = s.dropna().astype(float); return None if v.empty else v.iloc[:n].sum()
def _latest(s: pd.Series|None) -> float|None:
    if s is None or s.empty: return None
    v = s.dropna().astype(float); return v.iloc[0] if not v.empty else None

def yf_financials_health(tickers: List[str]) -> Tuple[str, Dict]:
    t0=_now_ms(); bad=[]
    def one(t):
        try:
            tk = yf.Ticker(t)
            qcf = tk.quarterly_cashflow
            cfo_q = _pick_row(qcf, _CF_ALIASES["cfo"])
            cap_q = _pick_row(qcf, _CF_ALIASES["capex"])
            fcf_q = _pick_row(qcf, ["Free Cash Flow","FreeCashFlow","Free cash flow"])
            cfo = _sum_last_n(cfo_q,4); cap = _sum_last_n(cap_q,4); fcf = _sum_last_n(fcf_q,4)
            if any(v is None for v in (cfo,cap,fcf)):
                acf = tk.cashflow
                if cfo is None: cfo=_latest(_pick_row(acf,_CF_ALIASES["cfo"]))
                if cap is None: cap=_latest(_pick_row(acf,_CF_ALIASES["capex"]))
                if fcf is None: fcf=_latest(_pick_row(acf,["Free Cash Flow","FreeCashFlow","Free cash flow"]))
            return None if all(v is not None for v in (cfo,cap,fcf)) else t
        except Exception: return t
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in ex.map(one, tickers):
            if r: bad.append(r)
    ms=_now_ms()-t0
    level = "HEALTHY" if not bad else ("DEGRADED" if len(bad)<=len(tickers)//2 else "DOWN")
    slow = " SLOW" if ms>=TIMEOUT_MS_WARN else ""
    return f"YF_FIN:{level} bad={len(bad)}/{len(tickers)} latency={_fmt_ms(ms)}{slow}", {
        "level":level,"latency_ms":ms,"bad":bad
    }

# ================================================================
# Finnhub: cash-flow（CFO/Capex）ヘルス（フォールバック）
# ================================================================
_FINN_CFO_KEYS   = ["netCashProvidedByOperatingActivities","netCashFromOperatingActivities","cashFlowFromOperatingActivities","operatingCashFlow"]
_FINN_CAPEX_KEYS = ["capitalExpenditure","capitalExpenditures","purchaseOfPPE","investmentsInPropertyPlantAndEquipment"]

def _finn_get(session: requests.Session, url: str, params: dict, retries: int=3, sleep_s: float=0.5):
    for i in range(retries):
        r = session.get(url, params=params, timeout=15)
        if r.status_code==429:
            time.sleep(min(2**i*sleep_s, 4.0)); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

def finnhub_health(tickers: List[str]) -> Tuple[str, Dict]:
    if not FINN_KEY:
        return "FINNHUB:SKIPPED (no key)", dict(level="SKIPPED",bad=[])
    t0=_now_ms(); base="https://finnhub.io/api/v1"; s=requests.Session(); bad=[]
    for sym in tickers:
        try:
            j=_finn_get(s,f"{base}/stock/cash-flow",{"symbol":sym,"frequency":"quarterly","limit":8,"token":FINN_KEY})
            arr=j.get("cashFlow") or []
            def pick(item,keys):
                for k in keys:
                    if k in item and item[k] is not None: return item[k]
            cfo_vals=[pick(x,_FINN_CFO_KEYS) for x in arr[:4]]
            cap_vals=[pick(x,_FINN_CAPEX_KEYS) for x in arr[:4]]
            cfo_ttm = np.nansum([np.nan if v is None else float(v) for v in cfo_vals]) if any(v is not None for v in cfo_vals) else None
            cap_ttm = np.nansum([np.nan if v is None else float(v) for v in cap_vals]) if any(v is not None for v in cap_vals) else None
            if cfo_ttm is None or cap_ttm is None:
                j=_finn_get(s,f"{base}/stock/cash-flow",{"symbol":sym,"frequency":"annual","limit":1,"token":FINN_KEY})
                arr=j.get("cashFlow") or []
                if arr:
                    item0=arr[0]
                    if cfo_ttm is None:
                        v=pick(item0,_FINN_CFO_KEYS); 
                        if v is not None: cfo_ttm=float(v)
                    if cap_ttm is None:
                        v=pick(item0,_FINN_CAPEX_KEYS); 
                        if v is not None: cap_ttm=float(v)
            if cfo_ttm is None or cap_ttm is None: bad.append(sym)
        except Exception: bad.append(sym)
    ms=_now_ms()-t0
    level="HEALTHY" if not bad else ("DEGRADED" if len(bad)<=len(tickers)//2 else "DOWN")
    slow=" SLOW" if ms>=TIMEOUT_MS_WARN else ""
    return f"FINNHUB:{level} bad={len(bad)}/{len(tickers)} latency={_fmt_ms(ms)}{slow}",{
        "level":level,"latency_ms":ms,"bad":bad
    }

# ================================================================
# SEC: companyfacts（Revenue/EPS）ヘルス
# ================================================================
def _sec_headers():
    """
    SECは連絡先付きUser-Agent/Fromを強く推奨・一部で必須。
    SEC_EMAILが空なら最低限のUAにしつつ、403発生時は上位でSKIP扱いにする。
    """
    ua = (f"api-health-probe/1 (+mailto:{SEC_EMAIL})" if SEC_EMAIL else "api-health-probe/1")
    hdr = {
        "User-Agent": ua[:200],
        "Accept": "application/json",
    }
    if SEC_EMAIL:
        hdr["From"] = SEC_EMAIL[:200]
    return hdr

def _sec_get(url: str, params=None, retries=3, sleep_s: float=0.5):
    """
    403やネットワークエラーは上位でSKIP判定できるよう None を返す。
    """
    for i in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_sec_headers(), timeout=15)
            if r.status_code==429:
                time.sleep(min(2**i*sleep_s, 4.0)); continue
            if r.status_code==403:
                # UA/From未設定やアクセス制限。上位でSKIP。
                return None
            r.raise_for_status(); return r.json()
        except Exception:
            time.sleep(min(2**i*sleep_s, 2.0))
    return None

def _sec_ticker_map() -> Dict[str,str]:
    j = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if j is None:
        return {}
    out={}
    it=(j.values() if isinstance(j,dict) else j)
    for item in it:
        try:
            t=(item.get("ticker") or item.get("TICKER") or "").upper()
            cik=str(item.get("cik_str") or item.get("CIK") or "").zfill(10)
            if t and cik: out[t]=cik
        except Exception: continue
    return out

SEC_REV_TAGS=["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","SalesRevenueGoodsNet","SalesRevenueServicesNet","Revenue"]
SEC_EPS_TAGS=["EarningsPerShareDiluted","EarningsPerShareBasicAndDiluted","EarningsPerShare","EarningsPerShareBasic"]

def _normalize_for_sec(sym: str) -> List[str]:
    s=(sym or "").upper(); outs=[]; add=lambda x: outs.append(x) if x and x not in outs else None
    add(s); add(s.replace(".","-")); add(s.replace("-","")); add(s.replace(".","")); return outs

def _units_for_tags(facts: dict, spaces: List[str], tags: List[str]) -> list:
    got=[]
    for sp in spaces:
        d=(facts.get("facts") or {}).get(sp) or {}
        for tg in tags:
            arr=(d.get(tg) or {}).get("units") or {}
            for unit, vals in (arr.items() if isinstance(arr,dict) else []):
                if isinstance(vals,list) and vals: got.append(vals)
    return got

def _series_q_and_a(arrs: list) -> Tuple[list, list]:
    q_pairs,a_pairs=[],[]
    for vals in arrs:
        for v in vals:
            try:
                dt=v.get("end") or v.get("fy"); val=float(v.get("val")); form=(v.get("form") or "").upper()
                if "10-Q" in form or "6-K" in form or form=="Q": q_pairs.append((dt,val))
                elif "10-K" in form or "20-F" in form or form=="K": a_pairs.append((dt,val))
            except Exception: pass
    q_pairs=sorted(q_pairs,key=lambda x: str(x[0]),reverse=True)
    a_pairs=sorted(a_pairs,key=lambda x: str(x[0]),reverse=True)
    return q_pairs,a_pairs

def sec_health(tickers: List[str]) -> Tuple[str, Dict]:
    t0=_now_ms(); t2cik=_sec_ticker_map(); bad=[]
    # CIKマップが取れない（403/ネット断/UA未設定など）はSKIPPED
    if not t2cik:
        ms=_now_ms()-t0
        note="no SEC_EMAIL/403" if not SEC_EMAIL else "SEC endpoint blocked"
        det=f"SEC:SKIPPED ({note}) latency={_fmt_ms(ms)}"
        return det,{"level":"SKIPPED","latency_ms":ms,"bad":[]}
    for t in tickers:
        cands=_normalize_for_sec(t); cik=next((t2cik.get(x) for x in cands if t2cik.get(x)), None)
        if not cik: bad.append(t); continue
        try:
            j=_sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
            if j is None:
                bad.append(t); continue
            rev_arr=_units_for_tags(j,["us-gaap","ifrs-full"],SEC_REV_TAGS)
            eps_arr=_units_for_tags(j,["us-gaap","ifrs-full"],SEC_EPS_TAGS)
            rev_q,rev_a=_series_q_and_a(rev_arr); eps_q,eps_a=_series_q_and_a(eps_arr)
            if not (rev_q or rev_a) or not (eps_q or eps_a): bad.append(t)
        except Exception: bad.append(t)
        time.sleep(0.30)  # SEC負荷配慮
    ms=_now_ms()-t0
    level="HEALTHY" if not bad else ("DEGRADED" if len(bad)<=len(tickers)//2 else "DOWN")
    slow=" SLOW" if ms>=TIMEOUT_MS_WARN else ""
    return f"SEC:{level} bad={len(bad)}/{len(tickers)} latency={_fmt_ms(ms)}{slow}",{
        "level":level,"latency_ms":ms,"bad":bad
    }

# ================================================================
# Orchestration
# ================================================================
def main():
    cur_path, cand_path = _autodiscover_csv()
    if not cur_path or not cand_path:
        msg = f"⚠️ CSV not found. cur={cur_path} cand={cand_path} (set CSV_CURRENT/CSV_CANDIDATE or place files)"
        print(msg); _post_slack(msg)
        if SOFT_FAIL:
            sys.exit(0)
        sys.exit(78)

    tickers=sorted(set(_read_tickers(cur_path)+_read_tickers(cand_path)))
    if not tickers:
        msg = f"⚠️ No tickers from CSV. cur={cur_path} cand={cand_path}"
        print(msg); _post_slack(msg)
        if SOFT_FAIL:
            sys.exit(0)
        sys.exit(78)

    # YF
    det_price,meta_price=yf_price_health(tickers)
    det_info ,meta_info =yf_fastinfo_health(tickers)
    det_fin  ,meta_fin  =yf_financials_health(tickers)

    # SEC
    det_sec  ,meta_sec  =sec_health(tickers)

    # Finnhub（必要時のみ。YF財務NG銘柄へのフォールバック検証）
    need_finn=meta_fin["bad"]
    det_finn,meta_finn  =finnhub_health(need_finn if need_finn else tickers[:0])

    # API別レベル
    levels_map = {
        "YF_PRICE": meta_price["level"],
        "YF_INFO" : meta_info ["level"],
        "YF_FIN"  : meta_fin  ["level"],
        "SEC"     : meta_sec  ["level"],
        "FINNHUB" : meta_finn.get("level","SKIPPED"),
    }
    pri={"DOWN":3,"DEGRADED":2,"HEALTHY":1,"SKIPPED":0}
    # コアAPI（OPTIONAL_APIS 以外）のワースト
    core_levels = [lvl for api,lvl in levels_map.items() if api not in OPTIONAL_APIS]
    core_worst = max(core_levels, key=lambda x: pri.get(x,0)) if core_levels else "HEALTHY"
    # 全体ワースト（表示用）
    all_worst  = max(levels_map.values(), key=lambda x: pri.get(x,0))
    # ただし、DOWN が OPTIONAL_APIS のみから来ている場合は全体を DEGRADED までに抑制
    if all_worst=="DOWN" and core_worst!="DOWN":
        worst = "DEGRADED"
    else:
        worst = all_worst
    emoji={"HEALTHY":"✅","DEGRADED":"⚠️","DOWN":"🛑"}.get(worst,"ℹ️")

    # 共通障害（同一日だけの欠損が過半）を簡易検知（価格系列ベース）
    outage_note=""
    try:
        from collections import Counter
        missing_dates=meta_price.get("per_ticker_missing",{})
        date_counter=Counter(); one_day_missing=0
        for _,info in missing_dates.items():
            dates=info.get("dates",set()); max_gap=info.get("max_gap",0)
            if len(dates)==1 and max_gap==1:
                one_day_missing+=1; date_counter.update(dates)
        threshold=max(1,len(tickers)//2)
        if one_day_missing>=threshold:
            (missing_day,hits),=date_counter.most_common(1)
            outage_note=f" | OUTAGE: common_missing_day={missing_day} hits={hits}"
            if worst=="HEALTHY":
                worst="DEGRADED"; emoji="🟠"
    except Exception:
        pass

    summary=f"{emoji} API_HEALTH {worst}{outage_note} (exit_on={EXIT_ON_LEVEL})\n{det_price} | {det_info} | {det_fin} | {det_sec} | {det_finn}"
    has_problem=("DEGRADED" in worst) or ("DOWN" in worst)

    if has_problem:
        def head_problem(xs): return ", ".join(xs[:10]) + (f" …(+{len(xs)-10})" if len(xs)>10 else "")
        lines=[]
        if meta_price["missing"] or meta_price["nf"]:
            xs=[*meta_price["nf"],*meta_price["missing"]]; lines.append(f"YF_PRICE NG: {head_problem(xs)}")
        if meta_info["bad"]:  lines.append(f"YF_INFO NG: {head_problem(meta_info['bad'])}")
        if meta_fin["bad"]:   lines.append(f"YF_FIN NG: {head_problem(meta_fin['bad'])}")
        if meta_sec["bad"]:   lines.append(f"SEC NG: {head_problem(meta_sec['bad'])}")
        if meta_finn.get("bad"): lines.append(f"FINNHUB NG: {head_problem(meta_finn['bad'])}")
        text=summary + ("\n" + "\n".join(lines) if lines else "")
    else:
        text=summary

    # “変なティッカー”は毎回通報
    def head_pair(pairs):
        xs=[f"{a}->{b}" for (a,b) in pairs[:10]]
        return ", ".join(xs) + (f" …(+{len(pairs)-10})" if len(pairs)>10 else "")
    def head(xs):
        return ", ".join(xs[:10]) + (f" …(+{len(xs)-10})" if len(xs)>10 else "")
    alias_fixed = meta_price.get("alias_fixed", [])
    still_missing = meta_price.get("nf", [])
    weird_lines = []
    if alias_fixed:
        weird_lines.append(f"Weird tickers (alias fixed): {head_pair(alias_fixed)}")
    if still_missing:
        weird_lines.append(f"Weird tickers (not found): {head(still_missing)}")
    if weird_lines:
        text = text + "\n" + "\n".join(weird_lines)

    print(text); _post_slack(text)
    if SOFT_FAIL: sys.exit(0)
    # 退出判定：基準は“コアAPIの状態”。OPTIONALがDOWNでも coreがHEALTHY/DEGRADEDなら緩和。
    exit_by = core_worst if core_worst!="HEALTHY" else worst
    def _rank(x): return {"HEALTHY":1,"DEGRADED":2,"DOWN":3}.get(x,0)
    # EXIT_ON_LEVEL 未満なら成功終了
    if _rank(exit_by) < _rank(EXIT_ON_LEVEL):
        sys.exit(0)
    sys.exit(20 if exit_by=="DOWN" else 10)

if __name__=="__main__":
    main()
