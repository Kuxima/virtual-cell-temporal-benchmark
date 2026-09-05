# -*- coding: utf-8 -*-
"""14 novel cluster 生物学注释：22 个测试期新出现 cluster 的 marker 基因
（4000 HVG 上 Wilcoxon rank-sum，one-vs-rest），输出 csv 供人工核对细胞类型。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
from common import DATA_PROC, RESULTS

train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
train.obs["split"] = "train"
test.obs["split"] = "test"

import anndata as ad
joint = ad.concat([train, test], join="inner", index_unique=None, label=None)
sc.pp.normalize_total(joint, target_sum=1e4)
sc.pp.log1p(joint)
joint.obs["cluster"] = joint.obs["cluster"].astype(str)
joint.obs["lineage"] = joint.obs["lineage"].astype(str)
joint.obs["timepoint"] = joint.obs["timepoint"].astype(str)

train_clu = set(joint.obs.loc[joint.obs["split"] == "train", "cluster"])
all_clu = set(joint.obs["cluster"])
novel_clu = sorted(all_clu - train_clu, key=lambda c: int(c))
shared_clu = sorted(all_clu & train_clu, key=lambda c: int(c))
print(f"train clusters {len(train_clu)}，novel {len(novel_clu)}")

# 每个 cluster：n cells、时间点分布、主要 lineage、top markers
sc.tl.rank_genes_groups(joint, groupby="cluster", method="wilcoxon", n_genes=15)
markers = {}
for clu in novel_clu:
    sub = joint.obs[joint.obs["cluster"] == clu]
    genes = joint.uns["rank_genes_groups"]["names"][clu][:15]
    markers[clu] = {
        "n_cells": len(sub),
        "timepoints": ",".join(sorted(sub["timepoint"].unique())),
        "lineage_top": sub["lineage"].value_counts().index[0],
        "lineage_frac": round(sub["lineage"].value_counts(normalize=True).iloc[0], 2),
        "markers": ", ".join(genes),
    }

df = pd.DataFrame(markers).T
df.to_csv(RESULTS / "novel_cluster_markers.csv", encoding="utf-8-sig")
print(df[["n_cells", "timepoints", "lineage_top", "lineage_frac"]].to_string())
print("\nmarkers 前 8 个 novel cluster:")
for c in novel_clu[:8]:
    print(f"  [{c}] {markers[c]['markers'][:120]}")
print("saved", RESULTS / "novel_cluster_markers.csv")
