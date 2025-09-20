'''ROLE: Orchestration ONLY（外部I/O・SSOT・Slack出力）, 計算は scorer.py'''
# === NOTE: 機能・入出力・ログ文言・例外挙動は不変。安全な短縮（import統合/複数代入/内包表記/メソッドチェーン/一行化/空行圧縮など）のみ適用 ===
BONUS_COEFF = 0.55  # 推奨: 攻め=0.45 / 中庸=0.55 / 守り=0.65
SWAP_DELTA_Z = 0.15   # 僅差判定: σの15%。(緩め=0.10 / 標準=0.15 / 固め=0.20)
SWAP_KEEP_BUFFER = 3  # n_target+この順位以内の現行は保持。(粘り弱=2 / 標準=3 / 粘り強=4〜5)
import logging, os, time, requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import zscore  # used via scorer

from scorer import Scorer, ttm_div_yield_portfolio, _log
import config

import warnings, atexit, threading
from collections import Counter, defaultdict

# ---------- 重複警告の集約ロジック ----------
_warn_lock = threading.Lock()
_warn_seen = set()                     # 初回表示済みキー
_warn_count = Counter()                # (category, message, module) → 件数
_warn_first_ctx = {}                   # 初回の (filename, lineno)

def _warn_key(message, category, filename, lineno, *_args, **_kwargs):
    # "同じ警告" を定義: カテゴリ + 正規化メッセージ + モジュールパス(先頭数階層)
    mod = filename.split("/site-packages/")[-1] if "/site-packages/" in filename else filename
    mod = mod.rsplit("/", 3)[-1]  # 長すぎ抑制（末尾3階層まで）
    msg = str(message).strip()
    return (category.__name__, msg, mod)

_orig_showwarning = warnings.showwarning

def _compact_showwarning(message, category, filename, lineno, file=None, line=None):
    key = _warn_key(message, category, filename, lineno)
    with _warn_lock:
        _warn_count[key] += 1
        if key not in _warn_seen:
            # 初回だけ1行で出す（カテゴリ | モジュール | メッセージ）
            _warn_seen.add(key)
            _warn_first_ctx[key] = (filename, lineno)
            # 1行フォーマット（行数節約）
            txt = f"[WARN][{category.__name__}] {message} | {filename}:{lineno}"
            print(txt)
        # 2回目以降は出さない（集約）

warnings.showwarning = _compact_showwarning

# ベースポリシー: 通常は警告を出す（default）→ ただし同一メッセージは集約
warnings.resetwarnings()
warnings.simplefilter("default")

# 2) ピンポイント間引き: yfinance 'Ticker.earnings' は "once"（初回のみ可視化）
warnings.filterwarnings(
    "once",
    message="'Ticker.earnings' is deprecated",
    category=DeprecationWarning,
    module="yfinance"
)

# 3) 最終サマリ: 同一警告が何回出たかを最後に1行で
@atexit.register
def _print_warning_summary():
    suppressed = []
    for key, cnt in _warn_count.items():
        if cnt > 1:
            (cat, msg, mod) = key
            filename, lineno = _warn_first_ctx.get(key, ("", 0))
            suppressed.append((cnt, cat, msg, mod, filename, lineno))
    if suppressed:
        suppressed.sort(reverse=True)  # 件数降順
        # 最多上位だけ出す（必要なら上限制御：ここでは上位10件）
        top = suppressed[:10]
        print(f"[WARN-SUMMARY] duplicated warning groups: {len(suppressed)}")
        for cnt, cat, msg, mod, filename, lineno in top:
            print(f"[WARN-SUMMARY] {cnt-1} more | [{cat}] {msg} | {mod} ({filename}:{lineno})")
        if len(suppressed) > len(top):
            print(f"[WARN-SUMMARY] ... and {len(suppressed)-len(top)} more groups suppressed")

# 4) 追加（任意）: 1ジョブあたりの総警告上限を設定したい場合
#    例: 上限1000を超えたら以降は完全サイレント
_WARN_HARD_LIMIT = int(os.getenv("WARN_HARD_LIMIT", "0") or "0")  # 0なら無効
if _WARN_HARD_LIMIT > 0:
    _orig_warn_func = warnings.warn
    def _limited_warn(*a, **k):
        total = sum(_warn_count.values())
        if total < _WARN_HARD_LIMIT:
            return _orig_warn_func(*a, **k)
        # 超過後は捨てる（最後にsummaryだけ残る）
    warnings.warn = _limited_warn

# ---------- ここまでで警告の“可視性は維持”しつつ“重複で行数爆発”を抑止 ----------

# その他
debug_mode, FINNHUB_API_KEY = True, os.environ.get("FINNHUB_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=(logging.INFO if debug_mode else logging.WARNING), force=True)

class T:
    t = perf_counter()

    @staticmethod
    def log(tag):
        now = perf_counter()
        print(f"[T] {tag}: {now - T.t:.2f}s")
        T.t = now

T.log("start")

# === ユニバースと定数（冒頭に固定） ===
exist, cand = [pd.read_csv(f, header=None)[0].tolist() for f in ("current_tickers.csv","candidate_tickers.csv")]
T.log(f"csv loaded: exist={len(exist)} cand={len(cand)}")
CAND_PRICE_MAX, bench = 450, '^GSPC'  # 価格上限・ベンチマーク
N_G, N_D = config.N_G, config.N_D  # G/D枠サイズ（NORMAL基準: G12/D8）
g_weights = {'GROWTH_F':0.35,'MOM':0.55,'VOL':-0.10}
D_BETA_MAX = float(os.environ.get("D_BETA_MAX", "0.8"))
FILTER_SPEC = {"G":{"pre_mask":["trend_template"]},"D":{"pre_filter":{"beta_max":D_BETA_MAX}}}
D_weights = {'QAL':0.1,'YLD':0.3,'VOL':-0.5,'TRD':0.1}
_fmt_w = lambda w: " ".join(f"{k}{int(v*100)}" for k, v in w.items())

# DRRS 初期プール・各種パラメータ
corrM = 45
DRRS_G, DRRS_D = dict(lookback=252,n_pc=3,gamma=1.2,lam=0.68,eta=0.8), dict(lookback=504,n_pc=4,gamma=0.8,lam=0.85,eta=0.5)
DRRS_SHRINK = 0.10  # 残差相関の対角シュリンク（基礎）

# クロス相関ペナルティ（未定義なら設定）
try: CROSS_MU_GD
except NameError: CROSS_MU_GD = 0.40  # 推奨 0.35–0.45（lam=0.85想定）

# 出力関連
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === 共有DTO（クラス間I/O契約）＋ Config ===
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
    df_full: pd.DataFrame | None = None
    df_full_z: pd.DataFrame | None = None
    scaler: Any | None = None

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
    debug_mode: bool = False

# === 共通ユーティリティ（複数クラスで使用） ===
# (unused local utils removed – use scorer.py versions if needed)

_env_true = lambda name, default=False: (os.getenv(name) or str(default)).strip().lower() == "true"

def _post_slack(payload: dict):
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url: print("⚠️ SLACK_WEBHOOK_URL 未設定"); return
    try:
        requests.post(url, json=payload).raise_for_status()
    except Exception as e:
        print(f"⚠️ Slack通知エラー: {e}")

def _slack_send_text_chunks(url: str, text: str, chunk: int = 2800) -> None:
    """Slackへテキストを分割送信（コードブロック形式）。"""

    def _post_text(payload: str) -> None:
        try:
            resp = requests.post(url, json={"text": payload})
            print(f"[DBG] debug_post status={getattr(resp,'status_code',None)} size={len(payload)}")
            if resp is not None:
                resp.raise_for_status()
        except Exception as e:
            print(f"[ERR] debug_post_failed: {e}")

    body = (text or "").strip()
    if not body:
        print("[DBG] skip debug send: empty body")
        return

    block, block_len = [], 0

    def _flush():
        nonlocal block, block_len
        if block:
            _post_text("```" + "\n".join(block) + "```")
            block, block_len = [], 0

    for raw in body.splitlines():
        line = raw or ""
        while len(line) > chunk:
            head, line = line[:chunk], line[chunk:]
            _flush()
            _post_text("```" + head + "```")
        add_len = len(line) if not block else len(line) + 1
        if block and block_len + add_len > chunk:
            _flush(); add_len = len(line)
        block.append(line)
        block_len += add_len
    _flush()

def _disjoint_keepG(top_G, top_D, poolD):
    """G重複をDから除去し、poolDで順次補充（枯渇時は元銘柄維持）。"""
    used, D, i = set(top_G), list(top_D), 0
    for j, t in enumerate(D):
        if t not in used:
            continue
        while i < len(poolD) and (poolD[i] in used or poolD[i] in D):
            i += 1
        if i < len(poolD):
            D[j] = poolD[i]; used.add(D[j]); i += 1
    return top_G, D


def _sticky_keep_current(agg: pd.Series, pick: list[str], incumbents: list[str],
                         n_target: int, delta_z: float, keep_buffer: int) -> list[str]:
    import pandas as pd, numpy as np
    sel = list(pick)
    if not sel: return sel
    ranked_sel = agg.reindex(sel).sort_values(ascending=False)
    kth = ranked_sel.iloc[min(len(sel), n_target)-1]
    std = agg.std()
    sigma = float(std) if pd.notna(std) else 0.0
    thresh = kth - delta_z * sigma
    ranked_all = agg.sort_values(ascending=False)
    cand = [t for t in incumbents if (t not in sel) and (t in agg.index)]
    for t in cand:
        within_score = pd.notna(agg[t]) and agg[t] >= thresh
        within_rank = t in ranked_all.index and ranked_all.index.get_loc(t) < n_target + keep_buffer
        if not (within_score or within_rank):
            continue
        non_inc = [x for x in sel if x not in incumbents]
        if not non_inc:
            break
        weakest = min(non_inc, key=lambda x: agg.get(x, -np.inf))
        if weakest in sel and agg.get(t, -np.inf) >= agg.get(weakest, -np.inf):
            sel.remove(weakest); sel.append(t)
    if len(sel) > n_target:
        sel = sorted(sel, key=lambda x: agg.get(x, -1e9), reverse=True)[:n_target]
    return sel


# === Input：外部I/Oと前処理（CSV/API・欠損補完） ===
class Input:
    def __init__(self, cand, exist, bench, price_max, finnhub_api_key=None):
        self.cand, self.exist, self.bench, self.price_max = cand, exist, bench, price_max
        self.api_key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY")

    # ---- （Input専用）EPS補完・FCF算出系 ----
    @staticmethod
    def _sec_headers():
        mail = (os.getenv("SEC_CONTACT_EMAIL") or "yasonba55@gmail.com").strip()
        app = (os.getenv("SEC_APP_NAME") or "FactorBot/1.0").strip()
        return {"User-Agent": f"{app} ({mail})", "From": mail, "Accept": "application/json"}

    @staticmethod
    def _sec_get(url: str, retries: int = 3, backoff: float = 0.5):
        for i in range(retries):
            r = requests.get(url, headers=Input._sec_headers(), timeout=20)
            if r.status_code in (429, 503, 403):
                time.sleep(min(2 ** i * backoff, 8.0))
                continue
            r.raise_for_status(); return r.json()
        r.raise_for_status()

    @staticmethod
    def _sec_ticker_map():
        j = Input._sec_get("https://data.sec.gov/api/xbrl/company_tickers.json")
        mp = {}
        for _, v in (j or {}).items():
            try:
                mp[str(v["ticker"]).upper()] = f"{int(v['cik_str']):010d}"
            except Exception:
                continue
        return mp

    # --- 追加: ADR/OTC向けの簡易正規化（末尾Y/F, ドット等） ---
    @staticmethod
    def _normalize_ticker(sym: str) -> list[str]:
        s = (sym or "").upper().strip()
        # 追加: 先頭の$や全角の記号を除去
        s = s.lstrip("$").replace("＄", "").replace("．", ".").replace("－", "-")
        cand: list[str] = []

        def add(x: str) -> None:
            if x and x not in cand:
                cand.append(x)

        # 1) 原文を最優先（SECは BRK.B, BF.B など . を正式採用）
        add(s)
        # 2) Yahoo系バリアント（. と - の揺れを相互に）
        if "." in s:
            add(s.replace(".", "-"))
            add(s.replace(".", ""))
        if "-" in s:
            add(s.replace("-", "."))
            add(s.replace("-", ""))
        # 3) ドット・ハイフン・ピリオド無し版（最後の保険）
        add(s.replace("-", "").replace(".", ""))
        # 4) ADR簡易：末尾Y/Fの除去（SECマップは本体ティッカーを持つことがある）
        if len(s) >= 2 and s[-1] in {"Y", "F"}:
            add(s[:-1])
        return cand

    @staticmethod
    def _sec_companyfacts(cik: str):
        return Input._sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")

    @staticmethod
    def _units_for_tags(facts: dict, namespaces: list[str], tags: list[str]) -> list[dict]:
        """facts から namespace/tag を横断して units 配列を収集（存在順に連結）。"""
        out: list[dict] = []
        facts = (facts or {}).get("facts", {})
        for ns in namespaces:
            node = facts.get(ns, {}) if isinstance(facts, dict) else {}
            for tg in tags:
                try:
                    units = node[tg]["units"]
                except Exception:
                    continue
                picks: list[dict] = []
                if "USD/shares" in units:
                    picks.extend(list(units["USD/shares"]))
                if "USD" in units:
                    picks.extend(list(units["USD"]))
                if not picks:
                    for arr in units.values():
                        picks.extend(list(arr))
                out.extend(picks)
        return out

    @staticmethod
    def _only_quarterly(arr: list[dict]) -> list[dict]:
        """companyfactsの混在配列から『四半期』だけを抽出。

        - frame に "Q" を含む（例: CY2024Q2I）
        - fp が Q1/Q2/Q3/Q4
        - form が 10-Q/10-Q/A/6-K
        """
        if not arr:
            return []
        q_forms = {"10-Q", "10-Q/A", "6-K"}
        out = [
            x
            for x in arr
            if (
                "Q" in (x.get("frame") or "").upper()
                or (x.get("fp") or "").upper() in {"Q1", "Q2", "Q3", "Q4"}
                or (x.get("form") or "").upper() in q_forms
            )
        ]
        out.sort(key=lambda x: (x.get("end") or ""), reverse=True)
        return out

    @staticmethod
    def _series_from_facts_with_dates(arr, key_val="val", key_dt="end", normalize=float):
        """companyfactsアイテム配列から (date,value) を返す。dateはYYYY-MM-DDを想定。"""
        out: List[Tuple[str, float]] = []
        for x in (arr or []):
            try:
                d = x.get(key_dt)
                if d is None:
                    continue
                v = x.get(key_val)
                out.append((str(d), normalize(v) if v is not None else float("nan")))
            except Exception:
                continue
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    def fetch_eps_rev_from_sec(self, tickers: list[str]) -> dict:
        out = {}
        t2cik = self._sec_ticker_map()
        n_map = n_rev = n_eps = 0
        miss_map: list[str] = []
        miss_facts: list[str] = []
        for t in tickers:
            base = (t or "").upper()
            candidates: list[str] = []
            for key in [base, *self._normalize_ticker(t)]:
                if key and key not in candidates:
                    candidates.append(key)
            cik = next((t2cik.get(key) for key in candidates if t2cik.get(key)), None)
            if not cik:
                out[t] = {}
                miss_map.append(t)
                continue
            try:
                j = self._sec_companyfacts(cik)
                facts = j or {}
                rev_tags = [
                    "Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet",
                    "SalesRevenueGoodsNet",
                    "SalesRevenueServicesNet",
                    "Revenue",
                ]
                eps_tags = [
                    "EarningsPerShareDiluted",
                    "EarningsPerShareBasicAndDiluted",
                    "EarningsPerShare",
                    "EarningsPerShareBasic",
                ]
                rev_arr = self._units_for_tags(facts, ["us-gaap", "ifrs-full"], rev_tags)
                eps_arr = self._units_for_tags(facts, ["us-gaap", "ifrs-full"], eps_tags)
                rev_q_items = self._only_quarterly(rev_arr)
                eps_q_items = self._only_quarterly(eps_arr)
                # (date,value) で取得
                rev_pairs = self._series_from_facts_with_dates(rev_q_items)
                eps_pairs = self._series_from_facts_with_dates(eps_q_items)
                rev_vals = [v for (_d, v) in rev_pairs]
                eps_vals = [v for (_d, v) in eps_pairs]
                rev_q = float(rev_vals[0]) if rev_vals else float("nan")
                eps_q = float(eps_vals[0]) if eps_vals else float("nan")
                rev_ttm = float(sum(v for v in rev_vals[:4] if v == v)) if rev_vals else float("nan")
                eps_ttm = float(sum(v for v in eps_vals[:4] if v == v)) if eps_vals else float("nan")
                out[t] = {
                    "eps_q_recent": eps_q,
                    "eps_ttm": eps_ttm,
                    "rev_q_recent": rev_q,
                    "rev_ttm": rev_ttm,
                    # 後段でDatetimeIndex化できるよう (date,value) を保持。値だけの互換キーも残す。
                    # 3年運用に合わせて四半期は直近12本のみ保持（約3年=12Q）
                    "eps_q_series_pairs": eps_pairs[:12],
                    "rev_q_series_pairs": rev_pairs[:12],
                    "eps_q_series": eps_vals[:12],
                    "rev_q_series": rev_vals[:12],
                }
                n_map += 1
                if rev_vals:
                    n_rev += 1
                if eps_vals:
                    n_eps += 1
            except Exception:
                out[t] = {}
                miss_facts.append(t)
            time.sleep(0.30)
        # 取得サマリをログ（Actionsで確認しやすいよう print）
        try:
            total = len(tickers)
            print(f"[SEC] map={n_map}/{total}  rev_q_hit={n_rev}  eps_q_hit={n_eps}")
            # デバッグ: 取得本数の分布（先頭のみ）
            try:
                lens = [len((out.get(t, {}) or {}).get("rev_q_series", [])) for t in tickers]
                print(f"[SEC] rev_q_series length: min={min(lens) if lens else 0} "
                      f"p25={np.percentile(lens,25) if lens else 0} median={np.median(lens) if lens else 0} "
                      f"p75={np.percentile(lens,75) if lens else 0} max={max(lens) if lens else 0}")
            except Exception:
                pass
            if miss_map:
                print(f"[SEC] no CIK map: {len(miss_map)} (サンプル例) {miss_map[:20]}")
            if miss_facts:
                print(f"[SEC] CIKあり だが対象factなし: {len(miss_facts)} (サンプル例) {miss_facts[:20]}")
        except Exception:
            pass
        return out

    def sec_dryrun_sample(self, tickers: list[str] | None = None) -> None:
        if not _env_true("SEC_DRYRUN_SAMPLE", False):
            return
        sample = tickers or ["BRK.B", "BF.B", "GOOGL", "META", "UBER", "PBR.A", "TSM", "NARI", "EVBN", "SWAV"]
        print(f"[SEC-DRYRUN] sample tickers: {sample}")
        try:
            t2cik = self._sec_ticker_map()
            hits = 0
            for sym in sample:
                candidates: list[str] = []

                def add(key: str) -> None:
                    if key and key not in candidates:
                        candidates.append(key)

                add((sym or "").upper())
                for alt in self._normalize_ticker(sym):
                    add(alt)
                if any(t2cik.get(key) for key in candidates):
                    hits += 1
            sec_data = self.fetch_eps_rev_from_sec(sample)
            rev_hits = sum(1 for v in sec_data.values() if v.get("rev_q_series"))
            eps_hits = sum(1 for v in sec_data.values() if v.get("eps_q_series"))
            total = len(sample)
            print(f"[SEC-DRYRUN] CIK map hit: {hits}/{total}  rev_q_series hits: {rev_hits}  eps_q_series hits: {eps_hits}")
        except Exception as e:
            print(f"[SEC-DRYRUN] error: {e}")
    @staticmethod
    def impute_eps_ttm(df: pd.DataFrame, ttm_col: str="eps_ttm", q_col: str="eps_q_recent", out_col: str|None=None) -> pd.DataFrame:
        out_col = out_col or ttm_col; df = df.copy(); df["eps_imputed"] = False
        cand = df[q_col]*4; ok = df[ttm_col].isna() & cand.replace([np.inf,-np.inf], np.nan).notna()
        df.loc[ok, out_col], df.loc[ok,"eps_imputed"] = cand[ok], True; return df

    _CF_ALIASES = {"cfo":["Operating Cash Flow","Total Cash From Operating Activities"], "capex":["Capital Expenditure","Capital Expenditures"]}

    @staticmethod
    def _pick_row(df: pd.DataFrame, names: list[str]) -> pd.Series|None:
        if df is None or df.empty: return None
        idx_lower={str(i).lower():i for i in df.index}
        for n in names:
            k=n.lower()
            if k in idx_lower: return df.loc[idx_lower[k]]
        return None

    @staticmethod
    def _sum_last_n(s: pd.Series|None, n: int) -> float|None:
        if s is None or s.empty: return None
        v=s.dropna().astype(float); return None if v.empty else v.iloc[:n].sum()

    @staticmethod
    def _latest(s: pd.Series|None) -> float|None:
        if s is None or s.empty: return None
        v=s.dropna().astype(float); return v.iloc[0] if not v.empty else None

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
            rows=[f.result() for f in as_completed(ex.submit(one,t) for t in tickers)]
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

    def _build_eps_df(self, tickers, tickers_bulk, info, sec_map: dict | None = None):
        eps_rows=[]
        for t in tickers:
            info_t = info[t]
            sec_t = (sec_map or {}).get(t, {})
            eps_ttm = sec_t.get("eps_ttm", info_t.get("trailingEps", np.nan))
            eps_q = sec_t.get("eps_q_recent", np.nan)
            try:
                qearn, so = tickers_bulk.tickers[t].quarterly_earnings, info_t.get("sharesOutstanding")
                if so and qearn is not None and not qearn.empty and "Earnings" in qearn.columns:
                    eps_ttm_q = qearn["Earnings"].head(4).sum()/so
                    if pd.notna(eps_ttm_q) and (pd.isna(eps_ttm) or (abs(eps_ttm)>0 and abs(eps_ttm/eps_ttm_q)>3)): eps_ttm = eps_ttm_q
                    if pd.isna(eps_q):
                        eps_q = qearn["Earnings"].iloc[-1]/so
            except Exception: pass
            rev_ttm = sec_t.get("rev_ttm", np.nan)
            rev_q = sec_t.get("rev_q_recent", np.nan)
            if (not sec_t) or pd.isna(rev_ttm):
                try:
                    tk = tickers_bulk.tickers[t]
                    qfin = getattr(tk, "quarterly_financials", None)
                    if qfin is not None and not qfin.empty:
                        idx_lower = {str(i).lower(): i for i in qfin.index}
                        rev_idx = None
                        for name in ("Total Revenue", "TotalRevenue"):
                            key = name.lower()
                            if key in idx_lower:
                                rev_idx = idx_lower[key]
                                break
                        if rev_idx is not None:
                            rev_series = pd.to_numeric(qfin.loc[rev_idx], errors="coerce").dropna()
                            if not rev_series.empty:
                                rev_ttm_yf = float(rev_series.head(4).sum())
                                if pd.isna(rev_ttm):
                                    rev_ttm = rev_ttm_yf
                                if pd.isna(rev_q):
                                    rev_q = float(rev_series.iloc[0])
                except Exception:
                    pass
            eps_rows.append({"ticker":t,"eps_ttm":eps_ttm,"eps_q_recent":eps_q,"rev_ttm":rev_ttm,"rev_q_recent":rev_q})
        return self.impute_eps_ttm(pd.DataFrame(eps_rows).set_index("ticker"))

    def prepare_data(self):
        """Fetch price and fundamental data for all tickers."""
        self.sec_dryrun_sample()
        cand_info = yf.Tickers(" ".join(self.cand))

        def _price(t: str) -> float:
            try:
                return cand_info.tickers[t].fast_info.get("lastPrice", np.inf)
            except Exception as e:
                print(f"{t}: price fetch failed ({e})")
                return np.inf

        cand_prices = {t: _price(t) for t in self.cand}
        cand_f = [t for t, p in cand_prices.items() if p <= self.price_max]
        T.log("price cap filter done (CAND_PRICE_MAX)")
        # 入力ティッカーの重複を除去し、現行→候補の順序を維持
        tickers = list(dict.fromkeys(self.exist + cand_f))
        T.log(f"universe prepared: unique={len(tickers)} bench={self.bench}")
        data = yf.download(tickers + [self.bench], period="600d",
                           auto_adjust=True, progress=False, threads=False)
        T.log("yf.download done")
        px = data["Close"].dropna(how="all", axis=1).ffill(limit=2)
        spx = data["Close"][self.bench].reindex(px.index).ffill()
        clip_days = int(os.getenv("PRICE_CLIP_DAYS", "0"))   # 0なら無効（既定）
        if clip_days > 0:
            px, spx = px.tail(clip_days + 1), spx.tail(clip_days + 1)
            logger.info("[T] price window clipped by env: %d rows (PRICE_CLIP_DAYS=%d)", len(px), clip_days)
        else:
            logger.info("[T] price window clip skipped; rows=%d", len(px))
        tickers_bulk, info = yf.Tickers(" ".join(tickers)), {}
        for t in tickers:
            try:
                info[t] = tickers_bulk.tickers[t].info
            except Exception as e:
                logger.info("[warn] %s: info fetch failed (%s)", t, e)
                info[t] = {}
        try:
            sec_map = self.fetch_eps_rev_from_sec(tickers)
        except Exception as e:
            logger.warning("[SEC] fetch_eps_rev_from_sec failed: %s", e)
            sec_map = {}

        def _brief_len(s):
            try:
                if isinstance(s, pd.Series):
                    return int(s.dropna().size)
                if isinstance(s, (list, tuple)):
                    return len([v for v in s if pd.notna(v)])
                if isinstance(s, np.ndarray):
                    return int(np.count_nonzero(~pd.isna(s)))
                return int(bool(s))
            except Exception:
                return 0

        def _has_entries(val) -> bool:
            try:
                if isinstance(val, pd.Series):
                    return not val.dropna().empty
                if isinstance(val, (list, tuple)):
                    return any(pd.notna(v) for v in val)
                return bool(val)
            except Exception:
                return False

        have_rev = 0
        have_eps = 0
        rev_lens: list[int] = []
        eps_lens: list[int] = []
        rev_y_lens: list[int] = []
        samples: list[tuple[str, int, str, float | None, int, str, float | None]] = []

        for t in tickers:
            entry = info.get(t, {})
            m = (sec_map or {}).get(t) or {}
            if entry is None or not isinstance(entry, dict):
                entry = {}
                info[t] = entry

            if m:
                pairs_r = m.get("rev_q_series_pairs") or []
                pairs_e = m.get("eps_q_series_pairs") or []
                if pairs_r:
                    idx = pd.to_datetime([d for (d, _v) in pairs_r], errors="coerce")
                    val = pd.to_numeric([v for (_d, v) in pairs_r], errors="coerce")
                    s = pd.Series(val, index=idx).sort_index()
                    entry["SEC_REV_Q_SERIES"] = s
                else:
                    entry["SEC_REV_Q_SERIES"] = m.get("rev_q_series") or []
                if pairs_e:
                    idx = pd.to_datetime([d for (d, _v) in pairs_e], errors="coerce")
                    val = pd.to_numeric([v for (_d, v) in pairs_e], errors="coerce")
                    s = pd.Series(val, index=idx).sort_index()
                    entry["SEC_EPS_Q_SERIES"] = s
                else:
                    entry["SEC_EPS_Q_SERIES"] = m.get("eps_q_series") or []

            r = entry.get("SEC_REV_Q_SERIES")
            e = entry.get("SEC_EPS_Q_SERIES")
            # 年次は直近3件（約3年）だけ保持。重み分岐の nY 判定は従来通り。
            try:
                if hasattr(r, "index") and isinstance(r.index, pd.DatetimeIndex):
                    y = r.resample("Y").sum().dropna()
                    entry["SEC_REV_Y_SERIES"] = y.tail(3)
                else:
                    entry["SEC_REV_Y_SERIES"] = []
            except Exception:
                entry["SEC_REV_Y_SERIES"] = []
            ry = entry.get("SEC_REV_Y_SERIES")
            if _has_entries(r):
                have_rev += 1
            if _has_entries(e):
                have_eps += 1
            lr = _brief_len(r)
            le = _brief_len(e)
            rev_lens.append(lr)
            eps_lens.append(le)
            rev_y_lens.append(_brief_len(ry))
            if len(samples) < 8:
                try:
                    rd = getattr(r, "index", [])[-1] if lr > 0 else None
                    rv = float(r.iloc[-1]) if lr > 0 else None
                    ed = getattr(e, "index", [])[-1] if le > 0 else None
                    ev = float(e.iloc[-1]) if le > 0 else None
                    samples.append((t, lr, str(rd) if rd is not None else "-", rv, le, str(ed) if ed is not None else "-", ev))
                except Exception:
                    samples.append((t, lr, "-", None, le, "-", None))

        logger.info("[SEC] series attach: rev_q=%d/%d, eps_q=%d/%d", have_rev, len(tickers), have_eps, len(tickers))
        logger.info(
            "[SEC_SERIES] rev_q=%d (<=12), eps_q=%d (<=12), rev_y=%d (<=3)",
            max(rev_lens) if rev_lens else 0,
            max(eps_lens) if eps_lens else 0,
            max(rev_y_lens) if rev_y_lens else 0,
        )

        if rev_lens:
            rev_lens_sorted = sorted(rev_lens)
            eps_lens_sorted = sorted(eps_lens)
            _log(
                "SEC_SERIES",
                f"rev_len min/med/max={rev_lens_sorted[0]}/{rev_lens_sorted[len(rev_lens)//2]}/{rev_lens_sorted[-1]} "
                f"eps_len min/med/max={eps_lens_sorted[0]}/{eps_lens_sorted[len(eps_lens)//2]}/{eps_lens_sorted[-1]}",
            )
        for (t, lr, rd, rv, le, ed, ev) in samples:
            _log("SEC_SERIES_SMP", f"{t}  rev_len={lr} last=({rd},{rv})  eps_len={le} last=({ed},{ev})")
        eps_df = self._build_eps_df(tickers, tickers_bulk, info, sec_map=sec_map)
        # index 重複があると .loc[t, col] が Series になり代入時に ValueError を誘発する
        if not eps_df.index.is_unique:
            eps_df = eps_df[~eps_df.index.duplicated(keep="last")]
        eps_df = eps_df.assign(
            EPS_TTM=eps_df["eps_ttm"],
            EPS_Q_LastQ=eps_df["eps_q_recent"],
            REV_TTM=eps_df["rev_ttm"],
            REV_Q_LastQ=eps_df["rev_q_recent"],
        )
        # ここで非NaN件数をサマリ表示（欠損状況の即時把握用）
        try:
            n = len(eps_df)
            c_eps = int(eps_df["EPS_TTM"].notna().sum())
            c_rev = int(eps_df["REV_TTM"].notna().sum())
            print(f"[SEC] eps_ttm non-NaN: {c_eps}/{n}  rev_ttm non-NaN: {c_rev}/{n}")
        except Exception:
            pass
        fcf_df = self.compute_fcf_with_fallback(tickers, finnhub_api_key=self.api_key)
        T.log("eps/fcf prep done")
        returns = px[tickers].pct_change()
        T.log("price prep/returns done")
        return dict(cand=cand_f, tickers=tickers, data=data, px=px, spx=spx, tickers_bulk=tickers_bulk, info=info, eps_df=eps_df, fcf_df=fcf_df, returns=returns)

# === Selector：相関低減・選定（スコア＆リターンだけ読む） ===
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
    def select_bucket_drrs(cls, returns_df: pd.DataFrame, score_ser: pd.Series, pool_tickers: list[str], k: int, *, n_pc: int, gamma: float, lam: float, lookback: int, shrink: float=0.10, g_fixed_tickers: list[str]|None=None, mu: float=0.0):
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
        selected_tickers = [pool_eff[i] for i in S]
        return dict(idx=S, tickers=selected_tickers, avg_res_corr=cls.avg_corr(C_within,S), sum_score=float(score[S].sum()), objective=float(Jn))

    # ---- 選定（スコア Series / returns だけを受ける）----
# === Output：出力整形と送信（表示・Slack） ===
class Output:

    def __init__(self, debug=None):
        # self.debug は使わない（互換のため引数は受けるが無視）
        self.miss_df = self.g_table = self.d_table = self.io_table = self.df_metrics_fmt = self.debug_table = None
        self.g_title = self.d_title = ""
        self.g_formatters = self.d_formatters = {}
        # 低スコア（GSC+DSC）Top10 表示/送信用
        self.low10_table = None
        self.debug_text = ""   # デバッグ用本文はここに一本化
        self._debug_logged = False

    # --- 表示（元 display_results のロジックそのまま） ---
    def display_results(self, *, exist, bench, df_z, g_score, d_score_all,
                        init_G, init_D, top_G, top_D, **kwargs):
        logger.info("📌 reached display_results")
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
                except Exception:
                    v = None
                if v is not None and not (isinstance(v, float) and v != v):
                    return v
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
        self.g_table = pd.concat([df_z.loc[G_UNI,['GROWTH_F','MOM','TRD','VOL']], gsc_series], axis=1)
        self.g_table.index = [t + ("⭐️" if t in top_G else "") for t in G_UNI]
        self.g_formatters = {col:"{:.2f}".format for col in ['GROWTH_F','MOM','TRD','VOL']}; self.g_formatters['GSC'] = "{:.3f}".format
        self.g_title = (f"[G枠 / {N_G} / {_fmt_w(g_weights)} / corrM={corrM} / "
                        f"LB={DRRS_G['lookback']} nPC={DRRS_G['n_pc']} γ={DRRS_G['gamma']} λ={DRRS_G['lam']} η={DRRS_G['eta']} shrink={DRRS_SHRINK}]")
        if near_G:
            add = [t for t in near_G if t not in set(G_UNI)][:10]
            if len(add) < 10:
                try:
                    aggG = getattr(sc, "_agg_G", pd.Series(dtype=float)).sort_values(ascending=False)
                    out_now = sorted(set(exist) - set(top_G + top_D))  # 今回 OUT
                    used = set(G_UNI + add)
                    def _push(lst):
                        nonlocal add, used
                        for t in lst:
                            if len(add) == 10: break
                            if t in aggG.index and t not in used:
                                add.append(t); used.add(t)
                    _push(out_now)           # ① 今回 OUT を優先
                    _push(list(aggG.index))  # ② まだ足りなければ上位で充填
                except Exception:
                    pass
            if add:
                near_tbl = pd.concat([df_z.loc[add,['GROWTH_F','MOM','TRD','VOL']], pd.Series({t: g_score.get(t) for t in add}, name='GSC')], axis=1)
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
            add = [t for t in near_D if t not in set(D_UNI)][:10]
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

        all_tickers = list(set(exist + list(top_G) + list(top_D) + [bench])); prices = yf.download(all_tickers, period='1y', auto_adjust=True, progress=False, threads=False)['Close'].ffill(limit=2)
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
        self.debug_text = ""
        if debug_mode:
            logger.info("debug_mode=True: df_z dump handled in scorer; skipping factor-side debug output")
        else:
            logger.debug(
                "skip debug log: debug_mode=%s debug_text_empty=%s",
                debug_mode, True
            )
        self._debug_logged = True

    # --- Slack送信（元 notify_slack のロジックそのまま） ---
    def notify_slack(self):
        SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

        if not SLACK_WEBHOOK_URL:
            print("⚠️ SLACK_WEBHOOK_URL not set (main report skipped)")
            return

        def _filter_suffix_from(spec: dict, group: str) -> str:
            g = spec.get(group, {})
            parts = [str(m) for m in g.get("pre_mask", [])]
            for k, v in (g.get("pre_filter", {}) or {}).items():
                base, op = (k[:-4], "<") if k.endswith("_max") else ((k[:-4], ">") if k.endswith("_min") else (k, "="))
                name = {"beta": "β"}.get(base, base)
                try:
                    val = f"{float(v):g}"
                except Exception:
                    val = str(v)
                parts.append(f"{name}{op}{val}")
            return "" if not parts else " / filter:" + " & ".join(parts)

        def _inject_filter_suffix(title: str, group: str) -> str:
            suf = _filter_suffix_from(FILTER_SPEC, group)
            return f"{title[:-1]}{suf}]" if suf and title.endswith("]") else (title + suf)

        def _blk(title, tbl, fmt=None, drop=()):
            if tbl is None or getattr(tbl, 'empty', False):
                return f"{title}\n(選定なし)\n"
            if drop and hasattr(tbl, 'columns'):
                keep = [c for c in tbl.columns if c not in drop]
                tbl, fmt = tbl[keep], {k: v for k, v in (fmt or {}).items() if k in keep}
            return f"{title}\n```{tbl.to_string(formatters=fmt)}```\n"

        message = "📈 ファクター分散最適化の結果\n"
        if self.miss_df is not None and not self.miss_df.empty:
            message += "Missing Data\n```" + self.miss_df.to_string(index=False) + "```\n"
        message += _blk(_inject_filter_suffix(self.g_title, "G"), self.g_table, self.g_formatters, drop=("TRD",))
        message += _blk(_inject_filter_suffix(self.d_title, "D"), self.d_table, self.d_formatters)
        message += "Changes\n" + ("(変更なし)\n" if self.io_table is None or getattr(self.io_table, 'empty', False) else f"```{self.io_table.to_string(index=False)}```\n")
        message += "Performance Comparison:\n```" + self.df_metrics_fmt.to_string() + "```"

        try:
            r = requests.post(SLACK_WEBHOOK_URL, json={"text": message})
            print(f"[DBG] main_post status={getattr(r, 'status_code', None)} size={len(message)}")
            if r is not None:
                r.raise_for_status()
        except Exception as e:
            print(f"[ERR] main_post_failed: {e}")

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

# === パイプライン可視化：G/D共通フロー（出力は不変） ===

def io_build_input_bundle() -> InputBundle:
    """
    既存の『データ取得→前処理』を実行し、InputBundle を返す。
    処理内容・列名・丸め・例外・ログ文言は現行どおり（変更禁止）。
    """
    state = Input(cand=cand, exist=exist, bench=bench, price_max=CAND_PRICE_MAX, finnhub_api_key=FINNHUB_API_KEY).prepare_data()
    return InputBundle(cand=state["cand"], tickers=state["tickers"], bench=bench, data=state["data"], px=state["px"], spx=state["spx"], tickers_bulk=state["tickers_bulk"], info=state["info"], eps_df=state["eps_df"], fcf_df=state["fcf_df"], returns=state["returns"])

def run_group(sc: Scorer, group: str, inb: InputBundle, cfg: PipelineConfig,
              n_target: int) -> tuple[list, float, float, float]:
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
        agg = agg[sc.filter_candidates(inb, agg, group, cfg)]

    selector = Selector()
    if hasattr(sc, "select_diversified"):
        pick, avg_r, sum_sc, obj = sc.select_diversified(agg, group, cfg, n_target,
            selector=selector, prev_tickers=None,
            corrM=cfg.drrs.corrM, shrink=cfg.drrs.shrink,
            cross_mu=cfg.drrs.cross_mu_gd)
    else:
        if group == "G":
            init = agg.nlargest(min(cfg.drrs.corrM, len(agg))).index.tolist()
            res = selector.select_bucket_drrs(returns_df=inb.returns, score_ser=agg, pool_tickers=init, k=n_target,
                n_pc=cfg.drrs.G.get("n_pc", 3), gamma=cfg.drrs.G.get("gamma", 1.2),
                lam=cfg.drrs.G.get("lam", 0.68),
                lookback=cfg.drrs.G.get("lookback", 252),
                shrink=cfg.drrs.shrink, g_fixed_tickers=None, mu=0.0)
        else:
            init = agg.nlargest(min(cfg.drrs.corrM, len(agg))).index.tolist()
            g_fixed = getattr(sc, "_top_G", None)
            res = selector.select_bucket_drrs(returns_df=inb.returns, score_ser=agg, pool_tickers=init, k=n_target,
                n_pc=cfg.drrs.D.get("n_pc", 4), gamma=cfg.drrs.D.get("gamma", 0.8),
                lam=cfg.drrs.D.get("lam", 0.85),
                lookback=cfg.drrs.D.get("lookback", 504),
                shrink=cfg.drrs.shrink, g_fixed_tickers=g_fixed,
                mu=cfg.drrs.cross_mu_gd)
        pick = res["tickers"]; avg_r = res["avg_res_corr"]
        sum_sc = res["sum_score"]; obj = res["objective"]
        if group == "D":
            _, pick = _disjoint_keepG(getattr(sc, "_top_G", []), pick, init)
            T.log("selection finalized (G/D)")
    try:
        inc = [t for t in exist if t in agg.index]
        pick = _sticky_keep_current(
            agg=agg, pick=pick, incumbents=inc, n_target=n_target,
            delta_z=SWAP_DELTA_Z, keep_buffer=SWAP_KEEP_BUFFER
        )
    except Exception as _e:
        print(f"[warn] sticky_keep_current skipped: {str(_e)}")
    # --- Near-Miss: 惜しくも選ばれなかった上位10を保持（Slack表示用） ---
    # 5) Near-Miss と最終集計Seriesを保持（表示専用。計算へ影響なし）
    try:
        pool = agg.drop(index=[t for t in pick if t in agg.index], errors="ignore")
        near10 = list(pool.sort_values(ascending=False).head(10).index)
        setattr(sc, f"_near_{group}", near10)
        setattr(sc, f"_agg_{group}", agg)
    except Exception:
        pass

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
        drrs=DRRSParams(
            corrM=corrM, shrink=DRRS_SHRINK,
            G=DRRS_G, D=DRRS_D, cross_mu_gd=CROSS_MU_GD
        ),
        price_max=CAND_PRICE_MAX,
        debug_mode=debug_mode
    )
    sc = Scorer()
    top_G, avgG, sumG, objG = run_group(sc, "G", inb, cfg, N_G)
    poolG = list(getattr(sc, "_agg_G", pd.Series(dtype=float)).sort_values(ascending=False).index)
    alpha = Scorer.spx_to_alpha(inb.spx)
    sectors = {t:(inb.info.get(t,{}).get("sector") or "U") for t in poolG}; scores = {t:Scorer.g_score.get(t,0.0) for t in poolG}
    top_G = Scorer.pick_top_softcap(scores, sectors, N=N_G, cap=2, alpha=alpha, hard=5)
    sc._top_G = top_G
    try:
        aggG = getattr(sc, "_agg_G", pd.Series(dtype=float)).sort_values(ascending=False)
        sc._near_G = [t for t in aggG.index if t not in set(top_G)][:10]
    except Exception:
        pass
    base = sum(Scorer.g_score.get(t,0.0) for t in poolG[:N_G])
    effs = sum(Scorer.g_score.get(t,0.0) for t in top_G)
    print(f"[soft_cap2] score_cost={(base-effs)/max(1e-9,abs(base)):.2%}, alpha={alpha:.3f}")
    top_D, avgD, sumD, objD = run_group(sc, "D", inb, cfg, N_D)
    fb = getattr(sc, "_feat", None)
    near_G = getattr(sc, "_near_G", [])
    selected12 = list(top_G)
    df = fb.df if fb is not None else pd.DataFrame()
    guni = _infer_g_universe(df, selected12, near_G)
    try:
        fire_recent = [t for t in guni
                       if (str(df.at[t, "G_BREAKOUT_recent_5d"]) == "True") or
                          (str(df.at[t, "G_PULLBACK_recent_5d"]) == "True")]
    except Exception: fire_recent = []

    lines = [
        "【G枠レポート｜週次モニタ（直近5営業日）】",
        "【凡例】🔥=直近5営業日内に「ブレイクアウト確定」または「押し目反発」を検知",
        f"選定{N_G}: {', '.join(_fmt_with_fire_mark(selected12, df))}" if selected12 else f"選定{N_G}: なし",
        f"次点10: {', '.join(_fmt_with_fire_mark(near_G, df))}" if near_G else "次点10: なし",]

    if fire_recent:
        fire_list = ", ".join([_label_recent_event(t, df) for t in fire_recent])
        lines.append(f"過去5営業日の検知: {fire_list}")
    else:
        lines.append("過去5営業日の検知: なし")

    try:
        webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
        if webhook:
            requests.post(webhook, json={"text": "\n".join([s for s in lines if s != ""])}, timeout=10)
    except Exception:
        pass

    out = Output()
    # 表示側から選定時の集計へアクセスできるように保持（表示専用・副作用なし）
    try: out._sc = sc
    except Exception: pass
    if hasattr(sc, "_feat"):
        try:
            fb = sc._feat
            out.miss_df = fb.missing_logs
            out.display_results(
                exist=exist,
                bench=bench,
                df_z=fb.df_z,
                g_score=fb.g_score,
                d_score_all=fb.d_score_all,
                init_G=top_G,
                init_D=top_D,
                top_G=top_G,
                top_D=top_D,
                df_full_z=getattr(fb, "df_full_z", None),
                prev_G=getattr(sc, "_prev_G", exist),
                prev_D=getattr(sc, "_prev_D", exist),
            )
        except Exception:
            pass
    out.notify_slack()
    sb = SelectionBundle(resG={"tickers": top_G, "avg_res_corr": avgG,
              "sum_score": sumG, "objective": objG},
        resD={"tickers": top_D, "avg_res_corr": avgD,
              "sum_score": sumD, "objective": objD},
        top_G=top_G, top_D=top_D, init_G=top_G, init_D=top_D)

    # --- Low Score Candidates (GSC+DSC bottom 10) : send before debug dump ---
    try:
        _low_df = (pd.DataFrame({"GSC": fb.g_score, "DSC": fb.d_score_all})
              .assign(G_plus_D=lambda x: x["GSC"] + x["DSC"])
              .sort_values("G_plus_D")
              .head(10)
              .round(3))
        low_msg = "Low Score Candidates (GSC+DSC bottom 10)\n" + _low_df.to_string(index=True, index_names=False)
        _post_slack({"text": f"```{low_msg}```"})
    except Exception as _e:
        _post_slack({"text": f"```Low Score Candidates: 作成失敗: {_e}```"})

    return sb

if __name__ == "__main__":
    run_pipeline()
