#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chat_pack.py
- リポ内の FILES を読み込み
- ファイルごとに L1.. の行番号を付与
- 1つの巨大テキストに連結
- 指定文字数で安全に分割して、コードフェンス付きの「貼り付け用チャンク」を作る
- 出力: chat_build/chat-pack.txt と chat_build/chunks/chunk-XX.txt（複数）
"""

import os, sys, textwrap, hashlib

# 対象メタ
OWNER  = "dakara32"
REPO   = "GPT_Code"
BRANCH = "main"

# ここに貼り付け対象ファイルを列挙
FILES = [
    "factor.py",
    "scorer.py",
    "drift.py",
    ".github/workflows/weekly-report.yml",
    ".github/workflows/daily-report.yml",
    "documents/README.md",
]

# このチャットが扱いやすい1チャンクの最大文字数（必要に応じて微調整 9000–14000）
CHUNK_SIZE = 12000

def add_line_numbers(body: str) -> str:
    out = []
    for i, line in enumerate(body.splitlines(), start=1):
        out.append(f"L{i}{'' if line=='' else ' '}{line}")
    return "\n".join(out) if out else "L1"

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def header_block() -> str:
    return textwrap.dedent(f"""\
    # === Chat Paste Pack ===
    # Repo: {OWNER}/{REPO} @ {BRANCH}
    # Files: {', '.join(FILES)}
    # 使い方: 下のチャンクをこのチャットに順番に貼るだけで、全ソースを把握できます。
    # 注記: 各ファイルは個別に L1.. の行番号で始まります（ファイルごとにリセット）。
    ---
    """)

def make_bundle() -> str:
    parts = [header_block()]
    for fp in FILES:
        if not os.path.exists(fp):
            parts.append(f"## <{fp}> (NOT FOUND)\n```text\nL1\n```\n")
            continue
        body = read_file(fp)
        parts.append(f"## <{fp}>\n```text\n{add_line_numbers(body)}\n```\n")
    return "\n".join(parts)

def chunkify(s: str, n: int):
    chunks = []
    i = 0
    while i < len(s):
        chunks.append(s[i:i+n])
        i += n
    return chunks

def main():
    out_dir   = "chat_build"
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    bundle = make_bundle()
    sha = hashlib.sha1(bundle.encode("utf-8")).hexdigest()[:12]

    # フル版
    pack_path = os.path.join(out_dir, "chat-pack.txt")
    with open(pack_path, "w", encoding="utf-8") as w:
        w.write(bundle)

    # チャンク版（コードフェンス付き）
    chunks = chunkify(bundle, CHUNK_SIZE)
    index_lines = [f"# Chat Paste Pack (SHA:{sha}) / {len(chunks)} chunks", ""]
    for idx, raw in enumerate(chunks, start=1):
        fname = os.path.join(chunk_dir, f"chunk-{idx:02d}.txt")
        with open(fname, "w", encoding="utf-8") as w:
            w.write(f"```text\n{raw}\n```")
        index_lines.append(f"- chunk-{idx:02d}.txt")

    with open(os.path.join(out_dir, "INDEX.txt"), "w", encoding="utf-8") as w:
        w.write("\n".join(index_lines))

    # Actions の Job Summary にも出力（オプション）
    guide = textwrap.dedent(f"""\
    ### Chat Paste Pack
    - SHA: {sha}
    - Total chunks: {len(chunks)}
    - How to use: Artifacts から 'chat-pack.txt' を開いて全部コピー、または 'chunks/' を順にコピー&貼り付け
    """)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as s:
            s.write(guide)

if __name__ == "__main__":
    sys.exit(main())
