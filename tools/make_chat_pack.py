#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chat_pack.py
- FILES_GROUPS（2パック）を読み込み
- 各ファイルに L1.. の行番号を付与し連結
- CodeForChat/<pack>/chat-pack.txt と chunks/ を生成
- CodeForChat/INDEX.txt（全パック目次）も出力
"""

import os, sys, textwrap, hashlib
import datetime

OWNER  = "dakara32"
REPO   = "GPT_Code"
BRANCH = "main"

CHUNK_SIZE = 12000  # iPhoneコピペ配慮で ~9k–14k の範囲で可

FILES_GROUPS = {
    "drift_pack": [
        "config.py",
        "drift.py",
        ".github/workflows/daily-report.yml",
        "documents/README.md",
        "documents/drift_design.md",
    ],
    "factor_pack": [
        "config.py",
        "factor.py",
        "scorer.py",
        ".github/workflows/weekly-report.yml",
        "documents/README.md",
        "documents/factor_design.md",
    ],
    "make_chat_pack_pack": [
        "tools/make_chat_pack.py",
        ".github/workflows/prepare-chat-pack.yml",
    ],
    "api-health_pack": [
        "tools/api_health_probe.py",
        ".github/workflows/api-health.yml",
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
    JST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    return textwrap.dedent(f"""\
    # === Chat Paste Pack ===
    # Repo: {OWNER}/{REPO} @ {BRANCH}
    # Files: {', '.join(file_list)}
    # 作成日時: {now} (JST)
    # 使い方: 下のチャンクを順に貼ればこのチャットで全体把握できます。
    # 注記: 各ファイルは個別に L1.. で行番号付与。
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
    return [s[i:i+n] for i in range(0, len(s), n)]

def main():
    root_out = "CodeForChat"
    os.makedirs(root_out, exist_ok=True)

    global_index = ["# Chat Paste Pack Index", ""]

    for pack_name, file_list in FILES_GROUPS.items():
        bundle = make_bundle(file_list)
        sha = hashlib.sha1(bundle.encode("utf-8")).hexdigest()[:12]

        pack_dir  = os.path.join(root_out, pack_name)
        chunk_dir = os.path.join(pack_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        # full
        with open(os.path.join(pack_dir, "chat-pack.txt"), "w", encoding="utf-8") as w:
            w.write(bundle)

        # chunks
        chunks = chunkify(bundle, CHUNK_SIZE)
        local_index = [f"# {pack_name} (SHA:{sha}) / {len(chunks)} chunks", ""]
        for idx, raw in enumerate(chunks, start=1):
            with open(os.path.join(chunk_dir, f"chunk-{idx:02d}.txt"), "w", encoding="utf-8") as w:
                w.write(f"```text\n{raw}\n```")
            local_index.append(f"- chunks/chunk-{idx:02d}.txt")

        with open(os.path.join(pack_dir, "INDEX.txt"), "w", encoding="utf-8") as w:
            w.write("\n".join(local_index))

        global_index += [f"## {pack_name}",
                         f"- {pack_name}/chat-pack.txt",
                         f"- {pack_name}/INDEX.txt", ""]

    with open(os.path.join(root_out, "INDEX.txt"), "w", encoding="utf-8") as w:
        w.write("\n".join(global_index))

if __name__ == "__main__":
    sys.exit(main())
