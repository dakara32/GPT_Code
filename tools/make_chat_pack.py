#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chat_pack.py (single-file version)
- FILES の各ファイルを読み込み
- 各ファイルに L1.. の行番号を付与
- 1つのテキスト chat_build/chat-pack.txt に連結して出力
- iPhone での一括コピーを想定（チャンクは出さない）
"""

import os, sys, textwrap, hashlib

OWNER  = "dakara32"
REPO   = "GPT_Code"
BRANCH = "main"

FILES = [
    # コアコード
    "factor.py",
    "scorer.py",
    "drift.py",

    # ワークフロー
    ".github/workflows/weekly-report.yml",
    ".github/workflows/daily-report.yml",

    # ドキュメント
    "documents/README.md",
    "documents/drift_design.md",
    "documents/factor_design.md",

    # ティッカー管理（リポ直下）
    "current_tickers.csv",
    "candidate_tickers.csv",
]

def add_line_numbers(body: str) -> str:
    out = []
    for i, line in enumerate(body.splitlines(), start=1):
        out.append(f"L{i}{'' if line=='' else ' '}{line}")
    return "\n".join(out) if out else "L1"

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def header_block(all_targets) -> str:
    return textwrap.dedent(f"""\
    # === Chat Paste Pack (single file) ===
    # Repo: {OWNER}/{REPO} @ {BRANCH}
    # Files: {', '.join(all_targets)}
    # 使い方: 下記を丸ごとコピーしてチャットに貼り付け（iPhoneはGitHubアプリの「Copy file contents」が便利）。
    ---
    """)

def make_bundle() -> str:
    parts = []
    targets = FILES
    parts.append(header_block(targets))
    for fp in targets:
        if not os.path.exists(fp):
            parts.append(f"## <{fp}> (NOT FOUND)\n```text\nL1\n```\n")
            continue
        body = read_file(fp)
        parts.append(f"## <{fp}>\n```text\n{add_line_numbers(body)}\n```\n")
    return "\n".join(parts)

def main():
    out_dir = "chat_build"
    os.makedirs(out_dir, exist_ok=True)

    bundle = make_bundle()
    sha = hashlib.sha1(bundle.encode("utf-8")).hexdigest()[:12]

    pack_path = os.path.join(out_dir, "chat-pack.txt")
    with open(pack_path, "w", encoding="utf-8") as w:
        w.write(bundle)

    # Actions の Job Summary にも短いガイドを追加
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as s:
            s.write(f"### Chat Paste Pack\n- SHA: {sha}\n- Output: {pack_path}\n")

if __name__ == "__main__":
    sys.exit(main())
