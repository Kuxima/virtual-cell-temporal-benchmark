# -*- coding: utf-8 -*-
"""01b v2 低内存版：h5py 直读 counts 层构建 AnnData（跳过 X 层，适配无页面文件系统），
时间切分 + 训练集内选 HVG，输出 train/test 两套 counts h5ad。"""
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from common import DATA_RAW, DATA_PROC

TRAIN_TP = ["10hpf", "12hpf", "14hpf", "16hpf"]
PATH = DATA_RAW / "zebrahub" / "zf_atlas_full_v4_release.h5ad"


def decode(x):
    return x.decode() if isinstance(x, bytes) else x


with h5py.File(PATH, "r") as f:
    # 旧版 h5ad：obs 索引列是 cell_id，var 索引是 _index
    n_obs = f["obs"]["cell_id"].shape[0]
    n_var = f["var"]["_index"].shape[0]

    # 只读 counts 层（CSR），分块读取以适配低提交内存系统
    def read_chunked(ds, dtype, chunk=50_000_000):
        n = ds.shape[0]
        out = np.empty(n, dtype=dtype)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e] = ds[s:e]
        return out

    g = f["layers"]["counts"]
    data = read_chunked(g["data"], np.float32)
    indices = read_chunked(g["indices"], np.int32)
    indptr = g["indptr"][:]
    X = sparse.csr_matrix((data, indices, indptr), shape=(n_obs, n_var))
    del data, indices, indptr
    gc.collect()
    print(f"counts 矩阵: {X.shape}, nnz={X.nnz}, dtype={X.dtype}")

    var_names = [decode(v) for v in f["var"]["_index"][:]]

    cats = f["obs"]["__categories"]

    def read_cat_col(col):
        labels = [decode(v) for v in cats[col][:]]
        codes = f["obs"][col][:]
        return pd.Series([labels[c] if c >= 0 else None for c in codes], dtype="category")

    obs = pd.DataFrame(
        {
            "timepoint": read_cat_col("timepoint"),
            "fish": read_cat_col("fish"),
            "lineage": read_cat_col("zebrafish_anatomy_ontology_class"),
            "cluster": read_cat_col("timepoint_cluster"),
        }
    )
    obs.index = [decode(v) for v in f["obs"]["cell_id"][:]]

adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=var_names))
del X
gc.collect()
print(f"AnnData 构建: {adata.shape}")

# 基因过滤
sc.pp.filter_genes(adata, min_cells=10)
print(f"基因过滤后: {adata.shape}")

tp = adata.obs["timepoint"].astype(str)
train_mask = tp.isin(TRAIN_TP).values
print(f"训练 ({'+'.join(TRAIN_TP)}): {train_mask.sum()}  测试 (19hpf-10dpf): {(~train_mask).sum()}")
print("测试时间点:", tp[~train_mask].value_counts().to_dict())

train = adata[train_mask].copy()
test = adata[~train_mask].copy()
del adata
gc.collect()

# HVG 仅在训练集上选（避免泄漏）
sc.pp.highly_variable_genes(train, n_top_genes=4000, flavor="seurat_v3")
hvg = train.var_names[train.var["highly_variable"]]
train = train[:, hvg].copy()
test = test[:, hvg].copy()
print(f"HVG: {len(hvg)}")

for obj, name in [(train, "train"), (test, "test")]:
    out = DATA_PROC / f"zebrahub_t1_{name}.h5ad"
    obj.write_h5ad(out, compression="gzip")
    print(f"{name}: {obj.shape} -> {out}  "
          f"(lineage {obj.obs['lineage'].nunique()} 类, cluster {obj.obs['cluster'].nunique()} 类)")

print("SPLIT DONE")
