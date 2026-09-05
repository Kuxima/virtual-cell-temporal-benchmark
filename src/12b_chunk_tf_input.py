# -*- coding: utf-8 -*-
"""12b 把 TF 输入切成 ~2 万细胞/块（绕开 dataloader 整文件单 batch 的 16.5GB 字符串矩阵）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import scanpy as sc
from common import DATA_PROC

SRC = DATA_PROC / "zebrahub_t1_joint_tfinput.h5ad"
OUTDIR = DATA_PROC / "tf_chunks"
OUTDIR.mkdir(exist_ok=True)
CHUNK = 5000

adata = sc.read_h5ad(SRC)
print(f"源 {adata.shape}")
n = adata.n_obs
for s in range(0, n, CHUNK):
    e = min(s + CHUNK, n)
    sub = adata[s:e].copy()
    out = OUTDIR / f"tf_chunk_{s//CHUNK:02d}.h5ad"
    sub.write_h5ad(out, compression="gzip")
    print(f"  [{s},{e}) -> {out.name}")
print("CHUNK DONE")
