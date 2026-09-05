# -*- coding: utf-8 -*-
"""11d TF/Geneformer 的分布内对照（与 11 相同协议：时间点内 cluster 分层 80/20 + kNN）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from common import DATA_PROC, RESULTS

K = 15


def knn_acc(Xtr, ytr, Xte, yte):
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


out = {}
for name, path, key in [("TF", "zebrahub_t1_joint_tf.h5ad", "X_tf"),
                        ("Geneformer", "zebrahub_t1_joint_gf.h5ad", "X_gf")]:
    a = sc.read_h5ad(DATA_PROC / path)
    X = np.asarray(a.obsm[key], dtype=np.float32)
    obs = a.obs.copy()
    obs["timepoint"] = obs["timepoint"].astype(str)
    obs["cluster"] = obs["cluster"].astype(str)
    res = {}
    for t in sorted(set(obs["timepoint"])):
        m = (obs["timepoint"] == t).values
        idx = np.where(m)[0]
        yc = obs["cluster"].values[idx]
        if len(idx) < 200 or len(set(yc)) < 2:
            continue
        itr, ite = train_test_split(idx, test_size=0.2, random_state=42, stratify=yc)
        acc_c = knn_acc(X[itr], yc[np.searchsorted(idx, itr)], X[ite], yc[np.searchsorted(idx, ite)])
        res[t] = round(acc_c, 3)
    out[name] = res
    print(name, {k: v for k, v in res.items()})

(RESULTS / "indist_tf_gf.json").write_text(json.dumps(out, indent=2))
print("saved")
