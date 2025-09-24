#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_health_probe.py — 選定プログラム依存API（Yahoo Finance / SEC / Finnhub）の総合ヘルスチェック

機能:
- CSV自動検出（current*/candidate*）
- 各APIのヘルス: YF価格/YF fast_info/YF財務/SEC companyfacts/Finnhub cash-flow
- 遅延測定・しきい値SLOW表示
- 共通欠損日の簡易OUTAGE検知（価格系列ベース）
- “変なティッカー”の常時通報（aliasで回復 / not found）
- Slack通知はアイコン付き、NG銘柄は改行して全件列挙
- EXIT_ON_LEVEL でCIの失敗基準を制御（既定DEGRADED、workflowでDOWNに設定推奨）
- Finnhubは任意API（OPTIONAL_APIS=FINNHUB）＝単独DOWNでも全体は最大DEGRADED

Env:
  SLACK_WEBHOOK_URL=[必須] Slack Incoming Webhook
  FINNHUB_API_KEY   [任意]
  SEC_CONTACT_EMAIL [推奨]  # 無い場合はSECをSKIPPED（403回避）
  # 後方互換: SEC_EMAIL があれば SEC_CONTACT_EMAIL の代替として使用
  CSV_CURRENT=./current.csv
  CSV_CANDIDATE=./candidate.csv
  YF_PERIOD=1y
  YF_MIN_LEN=120
  TIMEOUT_MS_WARN=5000
  MAX_WORKERS=8
  OPTIONAL_APIS=FINNHUB
  EXIT_ON_LEVEL=DEGRADED  # workflow側で DOWN を指定すると“DOWNの時だけ”失敗
  SOFT_FAIL=0             # 1なら常にexit 0
"""
import os, sys, time, json, math, csv, re, concurrent.futures as cf
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ==== Settings
CSV_CURRENT = os.getenv("CSV_CURRENT","./current.csv")
CSV_CANDIDATE= os.getenv("CSV_CANDIDATE","./candidate.csv")
YF_PERIOD   = os.getenv("YF_PERIOD","1y")
YF_MIN_LEN  = int(os.getenv("YF_MIN_LEN","120"))
TIMEOUT_MS_WARN = int(os.getenv("TIMEOUT_MS_WARN","5000"))
SOFT_FAIL   = os.getenv("SOFT_FAIL","0") == "1"
FINN_KEY      = os.getenv("FINNHUB_API_KEY")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK")
# SECメールは SEC_CONTACT_EMAIL を優先（後方互換で SEC_EMAIL）
SEC_CONTACT_EMAIL = (os.getenv("SEC_CONTACT_EMAIL") or os.getenv("SEC_EMAIL") or "").strip()
MAX_WORKERS = int(os.getenv("MAX_WORKERS","8"))
OPTIONAL_APIS = set([x.strip().upper() for x in os.getenv("OPTIONAL_APIS","FINNHUB").split(",") if x.strip()])
EXIT_ON_LEVEL = os.getenv("EXIT_ON_LEVEL","DEGRADED").upper()

# ==== Utils
def _now_ms() -> int: return int(time.time()*1000)

def _post_slack(text: str):
    if not SLACK_WEBHOOK:
        print("[SLACK] webhook missing; print only\n"+text); return
    try:
        r = requests.post(SLACK_WEBHOOK, json={"text": text}, timeout=8)
        print(f"[SLACK] status={r.status_code}"); r.raise_for_status()
    except Exception as e: print(f"[SLACK] send error: {e}")

def _read_tickers(path: str) -> List[str]:
    if not os.path.exists(path): return []
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
    cur, cand = (CSV_CURRENT if os.path.exists(CSV_CURRENT) else None,
                 CSV_CANDIDATE if os.path.exists(CSV_CANDIDATE) else None)
    if cur and cand: return cur, cand
    for root, _, files in os.walk(".", topdown=True):
        for fn in files:
            if not fn.lower().endswith(".csv"): continue
            p = os.path.join(root, fn); fl = fn.lower()
            if "current" in fl and not cur: cur = p
            if "candidate" in fl and not cand: cand = p
        if cur and cand: break
    return cur, cand

def _fmt_ms(ms: int) -> str:
    return f"{ms}ms" if ms < 1000 else f"{ms/1000:.2f}s"

def _latency_icon(ms: int) -> str:
    return "🐢" if ms >= TIMEOUT_MS_WARN else "✅"

# ==== SEC helpers
def _sec_headers():
    """
    SECは連絡先付きUser-Agent/Fromが推奨（SEC_CONTACT_EMAIL）。
    連絡先が空でも動かすが、403時は上位でSKIP。
    """
    mail = SEC_CONTACT_EMAIL
    ua   = f"api-health-probe/1 ({mail})" if mail else "api-health-probe/1"
    h    = {"User-Agent": ua[:200], "Accept": "application/json"}
    if mail:
        h["From"] = mail[:200]
    return h

def _sec_get(url: str, params=None, retries=3, sleep_s: float=0.5):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=_sec_headers(), timeout=15)
            if r.status_code == 429:
                time.sleep(min(2**i*sleep_s, 4.0)); continue
            if r.status_code == 403:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2**i*sleep_s, 2.0))
    return None

def _sec_ticker_map() -> Dict[str,str]:
    j = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if j is None: return {}
    out={}
    it=(j.values() if isinstance(j,dict) else j)
    for item in it:
        try:
            t=(item.get("ticker") or item.get("TICKER") or "").upper()
            cik=str(item.get("cik_str") or item.get("CIK") or "").zfill(10)
            if t and cik: out[t]=cik
        except Exception: continue
    return out

# ==== Yahoo Finance: ticker variants (for recovery)
def _yf_variants(sym: str):
    s = (sym or "").upper()
    cands = []
    def add(x):
        if x and x not in cands: cands.append(x)
    add(s)
    add(s.replace(".","-"))            # BRK.B -> BRK-B, PBR.A -> PBR-A
    add(re.sub(r"[.\-^]", "", s))      # 記号除去
    return cands

# ==== YF: price series health
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
    total = len(tickers)
    bad = list(dict.fromkeys([*nf, *missing]))
    level = "HEALTHY" if not bad else ("DOWN" if total > 0 and len(bad) >= total else "DEGRADED")
    det = f"YF_PRICE:{level} bad={len(bad)}/{total} latency={_fmt_ms(ms)} {_latency_icon(ms)}"
    meta = {"level":level,"latency_ms":ms,"ok":ok,"nf":nf,"missing":missing,
            "per_ticker_missing":per_ticker_missing,"alias_fixed":alias_fixed,"bad":bad}
    return det, meta

# ==== YF: fast_info health
def yf_fastinfo_health(tickers: List[str]) -> Tuple[str, Dict]:
    t0 = _now_ms()
    tk = yf.Tickers(" ".join(tickers))
    bad=[]
    for t in tickers:
        try:
            v = tk.tickers[t].fast_info.get("lastPrice", None)
            if v is None or (isinstance(v,float) and math.isnan(v)): bad.append(t)
        except Exception: bad.append(t)
    ms=_now_ms()-t0
    total=len(tickers)
    level = "HEALTHY" if not bad else ("DOWN" if total > 0 and len(bad) >= total else "DEGRADED")
    det = f"YF_INFO:{level} bad={len(bad)}/{total} latency={_fmt_ms(ms)} {_latency_icon(ms)}"
    return det, {"level":level,"latency_ms":ms,"bad":bad}

# ==== YF: financials health (CFO/Capex/FCF)
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
    total=len(tickers)
    level = "HEALTHY" if not bad else ("DOWN" if total > 0 and len(bad) >= total else "DEGRADED")
    det = f"YF_FIN:{level} bad={len(bad)}/{total} latency={_fmt_ms(ms)} {_latency_icon(ms)}"
    return det, {"level":level,"latency_ms":ms,"bad":bad}

# ==== Finnhub: cash-flow fallback
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
        det = f"FINNHUB:SKIPPED bad=0/{len(tickers)} latency=0ms {_latency_icon(0)} (no key)"
        return det, dict(level="SKIPPED",bad=[],latency_ms=0)
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
    total=len(tickers)
    level="HEALTHY" if not bad else ("DOWN" if total>0 and len(bad)>=total else "DEGRADED")
    det = f"FINNHUB:{level} bad={len(bad)}/{total} latency={_fmt_ms(ms)} {_latency_icon(ms)}"
    return det,{"level":level,"latency_ms":ms,"bad":bad}

# ==== SEC: companyfacts (Revenue/EPS) health
SEC_REV_TAGS=["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","SalesRevenueGoodsNet","SalesRevenueServicesNet","Revenue"]
SEC_EPS_TAGS=["EarningsPerShareDiluted","EarningsPerShareBasicAndDiluted","EarningsPerShare","EarningsPerShareBasic"]

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
    if not t2cik:
        ms=_now_ms()-t0
        det = f"SEC:SKIPPED bad=0/{len(tickers)} latency={_fmt_ms(ms)} {_latency_icon(ms)} (no SEC_CONTACT_EMAIL/403)"
        return det, {"level":"SKIPPED","latency_ms":ms,"bad":[]}
    for t in tickers:
        # '.'と'-'のゆらぎを許容した簡易マッチ
        cands = [(t or "").upper(), (t or "").upper().replace(".","-"), (t or "").upper().replace("-",""), (t or "").upper().replace(".","")]
        cik = next((t2cik.get(x) for x in cands if t2cik.get(x)), None)
        if not cik:
            bad.append(t); continue
        try:
            j=_sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
            if j is None: bad.append(t); continue
            rev_arr=_units_for_tags(j,["us-gaap","ifrs-full"],SEC_REV_TAGS)
            eps_arr=_units_for_tags(j,["us-gaap","ifrs-full"],SEC_EPS_TAGS)
            rev_q,rev_a=_series_q_and_a(rev_arr); eps_q,eps_a=_series_q_and_a(eps_arr)
            if not (rev_q or rev_a) or not (eps_q or eps_a): bad.append(t)
        except Exception: bad.append(t)
        time.sleep(0.30)
    ms=_now_ms()-t0
    total=len(tickers)
    level="HEALTHY" if not bad else ("DOWN" if total>0 and len(bad)>=total else "DEGRADED")
    det = f"SEC:{level} bad={len(bad)}/{total} latency={_fmt_ms(ms)} {_latency_icon(ms)}"
    return det,{"level":level,"latency_ms":ms,"bad":bad}

# ==== Orchestration
def main():
    cur_path, cand_path = _autodiscover_csv()
    if not cur_path or not cand_path:
        msg = f"⚠️ CSV not found. cur={cur_path} cand={cand_path} (set CSV_CURRENT/CSV_CANDIDATE or place files)"
        print(msg); _post_slack(msg)
        if SOFT_FAIL: sys.exit(0)
        sys.exit(78)

    tickers=sorted(set(_read_tickers(cur_path)+_read_tickers(cand_path)))
    if not tickers:
        msg = f"⚠️ No tickers from CSV. cur={cur_path} cand={cand_path}"
        print(msg); _post_slack(msg)
        if SOFT_FAIL: sys.exit(0)
        sys.exit(78)

    det_price,meta_price=yf_price_health(tickers)
    det_info ,meta_info =yf_fastinfo_health(tickers)
    det_fin  ,meta_fin  =yf_financials_health(tickers)
    det_sec  ,meta_sec  =sec_health(tickers)

    need_finn=meta_fin["bad"]
    det_finn,meta_finn  =finnhub_health(need_finn if need_finn else tickers[:0])

    levels_map = {
        "YF_PRICE": meta_price["level"],
        "YF_INFO" : meta_info ["level"],
        "YF_FIN"  : meta_fin  ["level"],
        "SEC"     : meta_sec  ["level"],
        "FINNHUB" : meta_finn.get("level","SKIPPED"),
    }
    pri={"DOWN":3,"DEGRADED":2,"HEALTHY":1,"SKIPPED":0}
    core_levels = [lvl for api,lvl in levels_map.items() if api not in OPTIONAL_APIS]
    core_worst = max(core_levels, key=lambda x: pri.get(x,0)) if core_levels else "HEALTHY"
    all_worst  = max(levels_map.values(), key=lambda x: pri.get(x,0))
    worst = "DEGRADED" if (all_worst=="DOWN" and core_worst!="DOWN") else all_worst
    emoji={"HEALTHY":"✅","DEGRADED":"⚠️","DOWN":"🛑"}.get(worst,"ℹ️")

    # 価格系列の共通障害（同一日だけの欠損が過半）簡易検知
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
            if worst=="HEALTHY": worst="DEGRADED"; emoji="🟠"
    except Exception:
        pass

    LABELS = {
        "YF_PRICE": "price",
        "YF_INFO": "fast_info",
        "YF_FIN": "financials (CFO/Capex/FCF)",
        "FINNHUB": "cash-flow (fallback)",
        "SEC": "companyfacts (revenue/eps)",
    }

    def icon_for(level: str) -> str:
        return {"HEALTHY":"✅","DEGRADED":"⚠️","DOWN":"🛑","SKIPPED":"⏭️"}.get(level, "ℹ️")

    def _fmt_block(detail: str, key: str) -> str:
        _, _, metrics = detail.partition(":")
        metrics = metrics.lstrip()
        heading = f"{icon_for(levels_map.get(key, ''))} {key}"
        label = LABELS.get(key)
        if label:
            heading += f" ({label})"
        return f"{heading}:\n{metrics}"

    status_order = [
        ("YF_PRICE", det_price),
        ("YF_INFO", det_info),
        ("YF_FIN", det_fin),
        ("FINNHUB", det_finn),
        ("SEC", det_sec),
    ]

    status_lines = [f"{emoji} API_HEALTH {worst}{outage_note} (exit_on={EXIT_ON_LEVEL})"]
    status_lines.extend(_fmt_block(detail, key) for key, detail in status_order)
    summary = "\n".join(status_lines)
    has_problem=("DEGRADED" in worst) or ("DOWN" in worst)

    if has_problem:
        def all_list(xs): return ", ".join(xs)
        lines=[]
        price_bad = meta_price.get("bad") or []
        if price_bad:
            lines.append("🆖YF_PRICE NG:\n" + all_list(price_bad))
        if meta_info.get("bad"):
            lines.append("🆖YF_INFO NG:\n" + all_list(meta_info["bad"]))
        if meta_fin.get("bad"):
            lines.append("🆖YF_FIN NG:\n" + all_list(meta_fin["bad"]))
        if meta_finn.get("bad"):
            lines.append("🆖FINNHUB NG:\n" + all_list(meta_finn["bad"]))
        if meta_sec.get("bad"):
            lines.append("🆖SEC NG:\n" + all_list(meta_sec["bad"]))
        text=summary + ("\n" + "\n".join(lines) if lines else "")
    else:
        text=summary

    # 変なティッカーは毎回全件通報
    def pair_all(pairs): return ", ".join(f"{a}->{b}" for (a,b) in pairs)
    def list_all(xs): return ", ".join(xs)
    alias_fixed = meta_price.get("alias_fixed", [])
    still_missing = meta_price.get("nf", [])
    weird_lines = []
    if alias_fixed:
        weird_lines.append("Weird tickers (alias fixed):\n" + pair_all(alias_fixed))
    if still_missing:
        weird_lines.append("Weird tickers (not found):\n" + list_all(still_missing))
    if weird_lines:
        text = text + "\n" + "\n".join(weird_lines)

    print(text); _post_slack(text)
    if SOFT_FAIL: sys.exit(0)
    # 退出判定：コアAPIを優先。OPTIONALがDOWNでも coreがHEALTHY/DEGRADEDなら緩和。
    exit_by = core_worst if core_worst!="HEALTHY" else worst
    def _rank(x): return {"HEALTHY":1,"DEGRADED":2,"DOWN":3}.get(x,0)
    if _rank(exit_by) < _rank(EXIT_ON_LEVEL): sys.exit(0)
    sys.exit(20 if exit_by=="DOWN" else 10)

if __name__=="__main__":
    main()
