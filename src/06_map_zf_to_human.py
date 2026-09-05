# -*- coding: utf-8 -*-
"""用 Zebrahub 官方 zf2mouse 表 + 小鼠/人类基因符号大小写规则，构建斑马鱼→人类映射。
小鼠基因符号首字母大写，人类全大写；一对多时只保留 IsBestScore=Yes。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from common import DATA_PROC, RESULTS, DATA_RAW


def load_zf2human():
    df = pd.read_csv(DATA_RAW / "zf2mouse.txt", sep="\t")
    # 只要最佳双向匹配
    if "IsBestScore" in df.columns:
        df = df[df["IsBestScore"].astype(str).str.lower().eq("yes")]
    zf = df["Gene1Symbol"].astype(str)
    ms = df["Gene2Symbol"].astype(str)
    # 小鼠符号 → 人类符号（常规规则）
    human = ms.str.upper()
    out = pd.DataFrame({"zf": zf, "mouse": ms, "human": human})
    out = out.drop_duplicates("zf")
    return dict(zip(out["zf"], out["human"]))


def collapse(adata, mapping):
    keep = [g for g in adata.var_names if g in mapping]
    sub = adata[:, keep].copy()
    human_sym = [mapping[g] for g in sub.var_names]
    X = sub.X.tocsr() if sparse.issparse(sub.X) else sparse.csr_matrix(sub.X)
    order, seen = [], {}
    for i, h in enumerate(human_sym):
        seen.setdefault(h, []).append(i)
        if h not in order:
            order.append(h)
    mats = [X[:, seen[h]].sum(axis=1) for h in order]
    Xh = sparse.hstack([sparse.csr_matrix(m) for m in mats]).tocsr().astype(np.float32)
    out = sc.AnnData(X=Xh, obs=sub.obs.copy())
    out.var_names = order
    out.var["gene_name"] = order
    out.var["ensembl_id"] = order
    return out


def main():
    mapping = load_zf2human()
    print(f"官方表映射条目: {len(mapping)}")
    train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
    hit = [g for g in train.var_names if g in mapping]
    print(f"命中 HVG: {len(hit)} / {train.n_vars}")
    (RESULTS / "zf2human_orthologs.json").write_text(json.dumps({g: mapping[g] for g in hit}, indent=2))
    if len(hit) < 200:
        raise SystemExit("同源映射过少")
    for name in ["train", "test"]:
        adata = sc.read_h5ad(DATA_PROC / f"zebrahub_t1_{name}.h5ad")
        mapped = collapse(adata, mapping)
        outp = DATA_PROC / f"zebrahub_t1_{name}_humanorth.h5ad"
        mapped.write_h5ad(outp, compression="gzip")
        print(f"{name}: {adata.shape} -> {mapped.shape}  {outp}")


if __name__ == "__main__":
    main()
