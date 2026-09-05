# -*- coding: utf-8 -*-
"""01 预处理 Zebrahub scRNA h5ad：QC → 保存标准化分析对象。
保留 raw counts 于 layers["counts"]，主 X 为 log1p 归一化。
时间点列名下载后确认（候选：Time_point / stage / hpf）。"""
import scanpy as sc
import anndata as ad
from common import DATA_RAW, DATA_PROC, SEED

import glob
import sys

# 定位输入（下载完成后确认实际文件名）
candidates = sorted(glob.glob(str(DATA_RAW / "zebrahub" / "**" / "*.h5ad"), recursive=True))
if not candidates:
    sys.exit("未找到 h5ad，请先完成 Zebrahub 下载")
print("候选文件:")
for c in candidates:
    print("  ", c)

INPUT = candidates[0]
adata = sc.read_h5ad(INPUT)
print(f"读入: {adata.shape[0]} cells x {adata.shape[1]} genes")
print("obs 列:", list(adata.obs.columns)[:30])

# --- QC ---
if "mt" not in adata.var.columns and adata.var_names.str.startswith("mt-").any():
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# --- HVG + 归一化（保留 counts）---
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", layer="counts", subset=False)

out = DATA_PROC / "zebrahub_processed.h5ad"
adata.write_h5ad(out, compression="gzip")
print(f"已保存: {out}")
