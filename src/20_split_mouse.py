# -*- coding: utf-8 -*-
"""20 P4 小鼠原肠胚（E-MTAB-6967, Pijuan-Sala 2019）T1 时间切分。
协议与 Zebrahub 完全同构：train E6.5-E7.0 → test E7.25-E8.5；
lineage = 38 celltype 归为 8 谱系类；cluster = cluster.stage（时间点内聚类）；
HVG 4000 仅在 train 上选。输出 counts h5ad（var 索引 = ENSMUSG 原生）。"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from scipy import sparse

from common import DATA_PROC, DATA_RAW

ATLAS = DATA_RAW / "pijuan_sala" / "atlas"
TRAIN_STAGES = {"E6.5", "E6.75", "E7.0"}

LINEAGE_MAP = {}
for ct in ["Epiblast", "Caudal epiblast", "NMP"]:
    LINEAGE_MAP[ct] = "epiblast"
for ct in ["Primitive Streak", "Anterior Primitive Streak"]:
    LINEAGE_MAP[ct] = "primitive_streak"
for ct in ["Nascent mesoderm", "Mixed mesoderm", "Caudal Mesoderm", "Paraxial mesoderm",
           "Somitic mesoderm", "Intermediate mesoderm", "Pharyngeal mesoderm",
           "Mesenchyme", "Notochord", "Cardiomyocytes"]:
    LINEAGE_MAP[ct] = "mesoderm"
for ct in ["ExE mesoderm", "Allantois"]:
    LINEAGE_MAP[ct] = "extraembryonic_mesoderm"
for ct in ["Def. endoderm", "Gut", "Visceral endoderm", "ExE endoderm", "Parietal endoderm"]:
    LINEAGE_MAP[ct] = "endoderm"
for ct in ["Rostral neurectoderm", "Caudal neurectoderm", "Forebrain/Midbrain/Hindbrain",
           "Neural crest", "Spinal cord", "Surface ectoderm", "ExE ectoderm"]:
    LINEAGE_MAP[ct] = "ectoderm"
for ct in ["Haematoendothelial progenitors", "Blood progenitors 1", "Blood progenitors 2",
           "Erythroid1", "Erythroid2", "Erythroid3", "Endothelium"]:
    LINEAGE_MAP[ct] = "haematoendothelial"
LINEAGE_MAP["PGC"] = "germline"

# ---------- 读 counts（gene×cell MatrixMarket，pandas C 引擎解析三元组）----------
print("解析 mtx...", flush=True)
with open(ATLAS / "raw_counts.mtx", "r") as f:
    banner = f.readline()
    n_gene, n_cell, nnz = map(int, f.readline().split())
print(f"mtx: {n_gene} genes x {n_cell} cells, nnz={nnz}", flush=True)
trip = pd.read_csv(ATLAS / "raw_counts.mtx", skiprows=2, sep=" ", header=None,
                   names=["i", "j", "v"], dtype={"i": np.int32, "j": np.int32, "v": np.float32},
                   comment="%")
assert len(trip) == nnz
M = sparse.coo_matrix((trip["v"].values, (trip["i"].values - 1, trip["j"].values - 1)),
                      shape=(n_gene, n_cell))
del trip
gc.collect()
X = sparse.csr_matrix(M.T)  # cell×gene
del M
gc.collect()

genes = pd.read_csv(ATLAS / "genes.tsv", sep="\t", header=None, names=["ensembl", "symbol"])
assert len(genes) == X.shape[1]

meta = pd.read_csv(ATLAS / "meta.csv", dtype={"doublet": str, "stripped": str})
assert len(meta) == X.shape[0]
meta.index = meta["cell"].astype(str)

obs = pd.DataFrame(index=meta.index.astype(str))
obs["stage"] = meta["stage"].astype(str).values
obs["sample"] = meta["sample"].astype(str).values
obs["seq_batch"] = meta["sequencing.batch"].astype(str).values
obs["celltype"] = meta["celltype"].astype(str).values
obs["cluster"] = meta["cluster.stage"].astype(str).values
obs["lineage"] = [LINEAGE_MAP.get(c, "NA") for c in obs["celltype"]]
obs["doublet"] = meta["doublet"].astype(str).values
obs["stripped"] = meta["stripped"].astype(str).values

adata = ad.AnnData(X=X.astype(np.float32), obs=obs,
                   var=pd.DataFrame(index=genes["ensembl"].astype(str).values))
adata.var["symbol"] = genes["symbol"].astype(str).values
del X
gc.collect()
print(f"AnnData {adata.shape}", flush=True)

# ---------- 过滤 ----------
keep = (~adata.obs["celltype"].isin(["NA"])) \
    & (adata.obs["stage"] != "mixed_gastrulation") \
    & (~adata.obs["doublet"].str.upper().isin(["TRUE"])) \
    & (~adata.obs["stripped"].str.upper().isin(["TRUE"]))
print(f"过滤: {keep.sum()}/{adata.n_obs}", flush=True)
adata = adata[keep].copy()
sc.pp.filter_genes(adata, min_cells=10)
print(f"基因过滤后 {adata.shape}", flush=True)

tp = adata.obs["stage"].astype(str)
train_mask = tp.isin(TRAIN_STAGES).values
print(f"train {train_mask.sum()} / test {(~train_mask).sum()}")
print("train lineage:", adata.obs.loc[train_mask, "lineage"].value_counts().to_dict())

train = adata[train_mask].copy()
test = adata[~train_mask].copy()
del adata
gc.collect()

sc.pp.highly_variable_genes(train, n_top_genes=4000, flavor="seurat_v3")
hvg = train.var_names[train.var["highly_variable"]]
train = train[:, hvg].copy()
test = test[:, hvg].copy()

train.obs["split"] = "train"
test.obs["split"] = "test"
train.write_h5ad(DATA_PROC / "mouse_t1_train.h5ad", compression="gzip")
test.write_h5ad(DATA_PROC / "mouse_t1_test.h5ad", compression="gzip")
joint = ad.concat([train, test], join="inner", index_unique=None, label=None)
joint.write_h5ad(DATA_PROC / "mouse_t1_joint.h5ad", compression="gzip")
print(f"HVG {len(hvg)}；train {train.shape} / test {test.shape}；joint {joint.shape}")
print("train cluster:", train.obs["cluster"].nunique(), "test cluster:", test.obs["cluster"].nunique())
print("SPLIT DONE")
