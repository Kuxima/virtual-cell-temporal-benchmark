# -*- coding: utf-8 -*-
"""12 构建 TranscriptFormer 输入：T1 joint（train+test）counts + ENSDARG Ensembl ID。
用原始图集 var 的 gene_ids（ENSDARG）按符号对上我们的 4000 HVG。"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
from common import DATA_PROC, DATA_RAW

# 1) 符号 -> ENSDARG（来自原始图集 var）
with h5py.File(DATA_RAW / "zebrahub" / "zf_atlas_full_v4_release.h5ad", "r") as f:
    sym = [v.decode() if isinstance(v, bytes) else v for v in f["var"]["_index"][:]]
    gid = [v.decode() if isinstance(v, bytes) else v for v in f["var"]["gene_ids"][:]]
sym2ens = dict(zip(sym, gid))
print(f"图集基因 {len(sym2ens)}，ENSDARG 示例 {list(sym2ens.items())[:2]}")

# 2) T1 train/test（4000 HVG counts，斑马鱼符号）拼 joint
train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
train.obs["split"] = "train"
test.obs["split"] = "test"
joint = ad.concat([train, test], join="inner", index_unique=None, label=None)
print(f"joint {joint.shape}")

# 3) 加 ensembl_id 列
ens = [sym2ens.get(s, None) for s in joint.var_names]
hit = sum(e is not None for e in ens)
print(f"符号→ENSDARG 命中 {hit}/{joint.n_vars}")
joint.var["ensembl_id"] = ens

# 4) 去掉没映射上的基因（TF 无法用）
mask = joint.var["ensembl_id"].notna()
joint = joint[:, mask.values].copy()
print(f"保留 {joint.n_vars} 基因")

# 5) X 必须是原始 counts；设置 raw
from scipy import sparse
if not sparse.isspmatrix_csr(joint.X):
    joint.X = sparse.csr_matrix(joint.X)
joint.X = joint.X.astype(np.float32)

out = DATA_PROC / "zebrahub_t1_joint_tfinput.h5ad"
joint.write_h5ad(out, compression="gzip")
print("saved", out)
# 诊断：每细胞非零基因数（= token 数）
nz = (joint.X > 0).sum(axis=1).A1 if sparse.issparse(joint.X) else (joint.X > 0).sum(1)
print(f"每细胞非零基因: median={np.median(nz):.0f} p90={np.percentile(nz, 90):.0f} max={nz.max()}")
