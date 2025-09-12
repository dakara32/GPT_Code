# 共通設定（factor / drift から参照）
from dataclasses import dataclass

TOTAL_TARGETS = 20

# 基準のバケット数（NORMAL）
COUNTS_BASE = {"G": 12, "D": 8}

# モード別の推奨バケット数
COUNTS_BY_MODE = {
    "NORMAL": {"G": 12, "D": 8},
    "CAUTION": {"G": 10, "D": 8},
    "EMERG": {"G": 8,  "D": 8},
}

# モード別のドリフト閾値（%）
DRIFT_THRESHOLD_BY_MODE = {"NORMAL": 12, "CAUTION": 14, "EMERG": float("inf")}

# モード別のTS（基本幅, 小数=割合）
TS_BASE_BY_MODE = {"NORMAL": 0.15, "CAUTION": 0.13, "EMERG": 0.10}
# 利益到達(+30/+60/+100%)時の段階タイト化（ポイント差）
TS_STEP_DELTAS_PT = (3, 6, 8)

# Breadthの校正は N_G に連動（緊急解除=ceil(1.5*N_G), 通常復帰=3*N_G）
N_G = COUNTS_BASE["G"]
N_D = COUNTS_BASE["D"]

