# -*- coding: utf-8 -*-
"""受限深度 Zebrahub Drive 遍历 v2：单层解析 + 自控递归 + 流式打印。
跳过 raw 输出目录；目标：定位 zf_atlas_full_v4_release.h5ad 的 file id。"""
import os
import re
import requests

from gdown.download_folder import _parse_embedded_folder_view

TOP = "1SIkI6FAOtNp3hCeJYxfi19EgnXCDCgOu"
SKIP_PAT = re.compile(r"cellranger|outs|epigenomics|atac|arc|TDR|fastq|bcl|raw", re.I)
MAX_DEPTH = 4

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})


def walk(folder_id: str, path: str, depth: int):
    if depth > MAX_DEPTH:
        print(f"MAXD  {path}", flush=True)
        return
    for attempt in range(3):
        try:
            _, children = _parse_embedded_folder_view(sess=session, folder_id=folder_id)
            break
        except Exception as e:
            print(f"!! retry {attempt+1} {path}: {e}", flush=True)
    else:
        return
    for child_id, name, ctype in children:
        full = f"{path}/{name}"
        if ctype == "application/vnd.google-apps.folder":
            if SKIP_PAT.search(name or ""):
                print(f"SKIP  {full}", flush=True)
                continue
            print(f"DIR   {full}  [id={child_id}]", flush=True)
            walk(child_id, full, depth + 1)
        else:
            hit = "  <== TARGET" if "atlas" in (name or "").lower() and name.endswith(".h5ad") else ""
            print(f"FILE  {full}  [id={child_id}]{hit}", flush=True)


if __name__ == "__main__":
    walk(TOP, "zebrahub", 0)
    print("DONE", flush=True)
