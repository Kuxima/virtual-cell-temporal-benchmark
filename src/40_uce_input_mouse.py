# -*- coding: utf-8 -*-
"""40 小鼠 UCE 输入准备:ENSMUSG -> MGI symbol 映射(mygene.info,带缓存),
输出 mouse_t1_joint_uce.h5ad(行序不变,var 为 symbol;无法映射的基因剔除并记录)。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import anndata as ad
import numpy as np
import requests

from common import DATA_PROC, RESULTS

CACHE = RESULTS / "ensmusg2sym_mygene.json"
OUT = DATA_PROC / "mouse_t1_joint_uce.h5ad"

adata = ad.read_h5ad(DATA_PROC / "mouse_t1_joint.h5ad")
ens_ids = list(adata.var_names)
print("mouse joint:", adata.shape)

if CACHE.exists():
    m = json.loads(CACHE.read_text())
else:
    m = {}
todo = [g for g in ens_ids if g not in m]
print(f"cached: {sum(1 for g in ens_ids if g in m)}, to query: {len(todo)}")

# 与 12b 相同的可靠模式:GET /v3/query 单基因 + 8 线程
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://mygene.info/v3/query"

def query(session, ensg):
    for attempt in range(3):
        try:
            response = session.get(
                API_URL,
                params={"q": ensg, "fields": "symbol,ensembl.gene", "species": "mouse", "size": 5},
                timeout=15,
            )
            if response.status_code == 200:
                for hit in response.json().get("hits", []):
                    ids = hit.get("ensembl")
                    ens_list = [ids] if isinstance(ids, str) else (
                        [d.get("gene") for d in ids] if isinstance(ids, list) else [])
                    if ensg in ens_list and hit.get("symbol"):
                        return ensg, hit["symbol"]
                return ensg, None
        except Exception:
            pass
    return ensg, None

import requests, time
S = requests.Session()
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, (ensg, sym) in enumerate(ex.map(lambda g: query(S, g), todo)):
        if sym:
            m[ensg] = sym
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(todo)} queried, hits {len(m)}", flush=True)
CACHE.write_text(json.dumps(m))

symbols = [m.get(g) for g in ens_ids]
keep = [i for i, s in enumerate(symbols) if s]
drop = len(ens_ids) - len(keep)
print(f"mapped {len(keep)}/{len(ens_ids)} genes; dropping {drop}")
out = adata[:, keep].copy()
out.var_names = [symbols[i] for i in keep]
out.var_names.name = "gene_symbol"
# 重复 symbol:保留首个出现(write 需要唯一 var_names)
import pandas as pd
dup = out.var_names[out.var_names.duplicated(keep="first")]
if len(dup):
    print("duplicate symbols:", sorted(set(dup))[:10], "— keeping first occurrence")
    out = out[:, ~out.var_names.duplicated(keep="first")].copy()
out.write_h5ad(OUT, compression="gzip")
print("saved", OUT, out.shape)
