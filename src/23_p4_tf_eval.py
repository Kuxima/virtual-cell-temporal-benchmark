# -*- coding: utf-8 -*-
"""23 P4 TF 小鼠收官评测：统一协议 eval_embedding + 分布内对照 + OOD AUROC。"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS, shared_novel_mask

spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding

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
tf = sc.read_h5ad(DATA_PROC / "mouse_t1_joint_tf.h5ad")
obs = tf.obs.copy()
for c in ("stage", "sample", "lineage", "cluster", "split"):
    obs[c] = obs[c].astype(str)
tr_m = (obs["split"] == "train").values
X = np.asarray(tf.obsm["X_tf"], dtype=np.float32)
print(f"mouse TF: {tf.shape}, train {tr_m.sum()}")

ytr_lin = obs.loc[tr_m, "lineage"].values
yte_lin = obs.loc[~tr_m, "lineage"].values
ytr_clu = obs.loc[tr_m, "cluster"].values
yte_clu = obs.loc[~tr_m, "cluster"].values
lin_classes = sorted(set(ytr_lin))
clu_classes = sorted(set(ytr_clu))
lin_map = {c: i for i, c in enumerate(lin_classes)}
clu_map = {c: i for i, c in enumerate(clu_classes)}
yte_clu_codes = np.array([clu_map.get(c, -1) for c in yte_clu], dtype=np.int32)
shared, novel = shared_novel_mask(ytr_clu, yte_clu)
assert np.all(yte_clu_codes[novel] == -1)
assert np.all(yte_clu_codes[shared] >= 0)
assert int(shared.sum() + novel.sum()) == len(yte_clu)
clu2lin = pd.DataFrame({"c": [clu_map[c] for c in ytr_clu], "l": [lin_map[c] for c in ytr_lin]}
                       ).groupby("c")["l"].agg(lambda s: s.value_counts().index[0]).to_dict()
ctx = dict(
    yte_lin=np.array([lin_map[c] for c in yte_lin]),
    yte_clu=yte_clu_codes,
    shared=shared, novel=novel,
    lin_codes_tr=np.array([lin_map[c] for c in ytr_lin]),
    lin_codes_te=np.array([lin_map[c] for c in yte_lin]),
    clu_codes_tr=np.array([clu_map[c] for c in ytr_clu]),
    lin_classes=lin_classes, clu_classes=clu_classes, clu2lin_code=clu2lin,
    tp_te=obs.loc[~tr_m, "stage"].values,
    fish_te_codes=pd.factorize(obs.loc[~tr_m, "sample"].values)[0],
)
assert int(tr_m.sum()) == 20308
assert int((~tr_m).sum()) == 88549
assert len(clu_classes) == 11
assert len(set(obs.loc[~tr_m, "cluster"])) == 22
assert int(shared.sum()) == 61784
assert int(novel.sum()) == 26765
assert np.array_equal(novel, ~shared)

rep = {"TF_mouse": eval_embedding("TF_mouse_frozen", "frozen", X[tr_m], X[~tr_m], ytr_lin, ytr_clu, ctx)}

# 分布内对照
indist = {}
tp_all = obs["stage"].values
clu_all = obs["cluster"].values
for t in sorted(set(tp_all)):
    idx = np.where(tp_all == t)[0]
    yc = clu_all[idx]
    if len(idx) < 200 or len(set(yc)) < 2:
        continue
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42, stratify=yc)
    indist[t] = round(knn_acc(X[itr], yc[np.searchsorted(idx, itr)], X[ite], yc[np.searchsorted(idx, ite)]), 3)
rep["in_dist_control"] = indist
print("分布内:", indist)

# OOD AUROC
dist = mean_knn_dist(X[tr_m], X[~tr_m])
shared, novel = ctx["shared"], ~ctx["shared"]
aurocs = {"global": round(float(roc_auc_score(novel.astype(int), dist)), 3)}
tp_te = ctx["tp_te"]
for t in sorted(set(tp_te)):
    m = tp_te == t
    if novel[m].sum() > 20 and (~novel[m]).sum() > 20:
        aurocs[t] = round(float(roc_auc_score(novel[m].astype(int), dist[m])), 3)
rep["ood_auroc"] = aurocs
print("OOD AUROC:", aurocs)

rep["runtime_s"] = round(time.time() - t0, 1)
(RESULTS / "p4_tf_eval.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
print("saved results/p4_tf_eval.json")
