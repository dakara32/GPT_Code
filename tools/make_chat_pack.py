#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chat_pack.py
- リポ内の FILES_GROUPS を読み込み（2パック: factor_pack / drift_pack）
- 各ファイルに L1.. の行番号を付与
- 1つの巨大テキストに連結し、指定文字数で安全に分割
- 出力:
    chat_build/factor_pack/chat-pack.txt
    chat_build/factor_pack/chunks/chunk-XX.txt
    chat_build/drift_pack/chat-pack.txt
    chat_build/drift_pack/chunks/chunk-XX.txt
    chat_build/INDEX.txt（両方の目次）
- Actions の Job Summary にも簡易ガイドを出力
"""

import os, sys, textwrap, hashlib

# 対象メタ
OWNER  = "dakara32"
REPO   = "GPT_Code"
BRANCH = "main"

# このチャットが扱いやすい1チャンクの最大文字数（必要に応じて 9000–14000）
CHUNK_SIZE = 12000

# === ここがポイント：2種類のパックを定義 ===
FILES_GROUPS = {
    # ① factor系
    "factor_pack": [
        "factor.py",
        "scorer.py",
        ".github/workflows/weekly-report.yml",
        "documents/README.md",
        "documents/factor_design.md",
    ],
    # ② drift系
    "drift_pack": [
        "drift.py",
        ".github/workflows/daily-report.yml",
        "documents/README.md",
        "documents/drift_design.md",
    ],
}

def add_line_numbers(body: str) -> str:
    out = []
    for i, line in enumerate(body.splitlines(), start=1):
        out.append(f"L{i}{'' if line=='' else ' '}{line}")
    return "\n".join(out) if out else "L1"

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def header_block(file_list) -> str:
    return textwrap.dedent(f"""\
    # === Chat Paste Pack ===
    # Repo: {OWNER}/{REPO} @ {BRANCH}
    # Files: {', '.join(file_list)}
    # 使い方: 下のチャンクをこのチャットに順番に貼るだけで、全ソースを把握できます。
    # 注記: 各ファイルは個別に L1.. の行番号で始まります（ファイルごとにリセット）。
    ---
    """)

def make_bundle(file_list) -> str:
    parts = [header_block(file_list)]
    for fp in file_list:
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
    root_out = "chat_build"
    os.makedirs(root_out, exist_ok=True)

    # グローバルINDEXの見出し
    global_index = ["# Chat Paste Pack Index", ""]

    for pack_name, file_list in FILES_GROUPS.items():
        bundle = make_bundle(file_list)
        sha = hashlib.sha1(bundle.encode("utf-8")).hexdigest()[:12]

        pack_dir = os.path.join(root_out, pack_name)
        chunk_dir = os.path.join(pack_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        # フル版
        pack_path = os.path.join(pack_dir, "chat-pack.txt")
        with open(pack_path, "w", encoding="utf-8") as w:
            w.write(bundle)

        # チャンク版（コードフェンス付き）
        chunks = chunkify(bundle, CHUNK_SIZE)
        local_index = [f"# {pack_name} (SHA:{sha}) / {len(chunks)} chunks", ""]
        for idx, raw in enumerate(chunks, start=1):
            fname = os.path.join(chunk_dir, f"chunk-{idx:02d}.txt")
            with open(fname, "w", encoding="utf-8") as w:
                w.write(f"```text\n{raw}\n```")
            local_index.append(f"- chunks/chunk-{idx:02d}.txt")

        with open(os.path.join(pack_dir, "INDEX.txt"), "w", encoding="utf-8") as w:
            w.write("\n".join(local_index))

        # グローバルINDEXにも登録
        global_index.append(f"## {pack_name}")
        global_index.append(f"- {pack_name}/chat-pack.txt")
        global_index.append(f"- {pack_name}/INDEX.txt")
        global_index.append("")

    # ルートのINDEX.txt
    with open(os.path.join(root_out, "INDEX.txt"), "w", encoding="utf-8") as w:
        w.write("\n".join(global_index))

    # Actions の Job Summary（両方ぶんまとめて）
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        guide = textwrap.dedent(f"""\
        ### Chat Paste Pack (multi)
        - Output Root: chat_build/
        - Packs:
          - factor_pack/: chat-pack.txt, chunks/, INDEX.txt
          - drift_pack/:  chat-pack.txt, chunks/, INDEX.txt

        **How to use**
        1) Open the 'chat-paste-pack' artifact
        2) Choose 'factor_pack' or 'drift_pack'
        3) Copy 'chat-pack.txt' (or the files under 'chunks/') into ChatGPT
        """)
        with open(step_summary, "a", encoding="utf-8") as s:
            s.write(guide)

if __name__ == "__main__":
    sys.exit(main())
