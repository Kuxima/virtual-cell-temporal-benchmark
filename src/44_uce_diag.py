# -*- coding: utf-8 -*-
"""44 UCE 斑马鱼诊断:分布内对照(每时间点 80/20 cluster acc)+ OOD AUROC(15-NN 平均余弦距离)。
输出结构与 indist_tf_gf.json / ood_auroc_models.json 的条目同构,写入 results/uce_diag.json:
  {"UCE": {"in_dist": {tp: acc}}, {"global": auroc, "per_timepoint": {tp: auroc}}}"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS

K = 15


def knn_acc(Xtr, ytr, Xte, yte):
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


def mean_knn_dist(Xref, Xq, k=K, chunk=4096):
    Xr = Xref / (np.linalg.norm(Xref, axis=1, keepdims=True) + 1e-12)
    Xq = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(Xq), dtype=np.float32)
    for s in range(0, len(Xq), chunk):
        e = min(s + chunk, len(Xq))
        sim = Xq[s:e] @ Xr.T
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        out[s:e] = 1.0 - np.take_along_axis(sim, idx, axis=1).mean(axis=1)
    return out


t0 = time.time()
a = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_uce.h5ad")
obs = a.obs.copy()
for c in ("timepoint", "fish", "lineage", "cluster", "split"):
    obs[c] = obs[c].astype(str)
tr_m = (obs["split"] == "train").values
X = np.asarray(a.obsm["X_uce"], dtype=np.float32)
print(f"UCE zf: {a.shape}, train {tr_m.sum()}")

# 分布内对照(全 10 个时间点)
indist = {}
tp_all = obs["timepoint"].values
clu_all = obs["cluster"].values
for t in sorted(set(tp_all)):
    idx = np.where(tp_all == t)[0]
    yc = clu_all[idx]
    if len(idx) < 200 or len(set(yc)) < 2:
        continue
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42, stratify=yc)
    indist[t] = round(knn_acc(X[itr], yc[np.searchsorted(idx, itr)], X[ite], yc[np.searchsorted(idx, ite)]), 3)
print("分布内:", indist)

# OOD AUROC(测试细胞,novel vs shared;全局 + 按时间点)
ytr_clu = obs.loc[tr_m, "cluster"].values
yte_clu = obs.loc[~tr_m, "cluster"].values
novel = ~np.isin(yte_clu, np.unique(ytr_clu))
dist = mean_knn_dist(X[tr_m], X[~tr_m])
aurocs = {"global": round(float(roc_auc_score(novel.astype(int), dist)), 3)}
tp_te = obs.loc[~tr_m, "timepoint"].values
for t in sorted(set(tp_te)):
    m = tp_te == t
    if novel[m].sum() > 20 and (~novel[m]).sum() > 20:
        aurocs[t] = round(float(roc_auc_score(novel[m].astype(int), dist[m])), 3)
print("OOD AUROC:", aurocs)

rep = {"UCE": {"in_dist": indist, "ood": aurocs},
       "runtime_s": round(time.time() - t0, 1)}
(RESULTS / "uce_diag.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
print("saved results/uce_diag.json")
