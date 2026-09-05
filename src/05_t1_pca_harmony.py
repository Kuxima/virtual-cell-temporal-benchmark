# -*- coding: utf-8 -*-
"""T1 经典基线：PCA（严格无泄漏）+ Harmony（联合空间，标准实践）。
输出与 scVI 基线同结构的 JSON，便于画对比图。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
from harmony import harmonize
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS, SEED

LABEL = "lineage"
K = 15


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def lognorm(adata):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def knn_metrics(X_train, y_train, X_test, y_test, timepoints):
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    known = np.isin(y_test, np.unique(y_train))
    per_tp = {}
    for t in sorted(np.unique(timepoints)):
        m = timepoints == t
        per_tp[str(t)] = float(accuracy_score(y_test[m], pred[m]))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy_known_types_only": float(accuracy_score(y_test[known], pred[known])) if known.any() else None,
        "accuracy_per_timepoint": per_tp,
        "n_novel_types": int(len(set(y_test) - set(y_train))),
    }


def main():
    t0 = time.time()
    train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
    test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
    print(f"train {train.shape}  test {test.shape}")

    y_train = train.obs[LABEL].astype(str).values
    y_test = test.obs[LABEL].astype(str).values
    tp = test.obs["timepoint"].astype(str).values

    train_n = lognorm(train)
    test_n = lognorm(test)

    # --- PCA：只在训练集拟合，测试集投影（严格无泄漏）---
    pca = PCA(n_components=50, random_state=SEED)
    Xtr = pca.fit_transform(to_dense(train_n.X))
    Xte = pca.transform(to_dense(test_n.X))
    pca_m = knn_metrics(Xtr, y_train, Xte, y_test, tp)
    pca_m.update(
        {
            "model": "PCA-50 + kNN (fit on train only)",
            "n_train": int(train.n_obs),
            "n_test": int(test.n_obs),
            "label": LABEL,
            "runtime_s": round(time.time() - t0, 1),
        }
    )
    (RESULTS / "t1_pca.json").write_text(json.dumps(pca_m, indent=2, ensure_ascii=False))
    print("PCA:", json.dumps({k: pca_m[k] for k in ("accuracy", "macro_f1", "accuracy_per_timepoint")}, ensure_ascii=False, indent=2))

    # --- Harmony：拼接后按 fish 校正，再 kNN（标准实践，轻微泄漏）---
    t1 = time.time()
    joint = train_n.concatenate(test_n, batch_key="split", batch_categories=["train", "test"], index_unique=None)
    joint.obsm["X_pca"] = np.vstack([Xtr, Xte])
    joint.obsm["X_harmony"] = harmonize(joint.obsm["X_pca"], joint.obs, batch_key="fish")
    H = joint.obsm["X_harmony"]
    n_tr = train.n_obs
    harm_m = knn_metrics(H[:n_tr], y_train, H[n_tr:], y_test, tp)
    harm_m.update(
        {
            "model": "Harmony(fish) + kNN on concatenated PCA",
            "n_train": int(train.n_obs),
            "n_test": int(test.n_obs),
            "label": LABEL,
            "runtime_s": round(time.time() - t1, 1),
        }
    )
    (RESULTS / "t1_harmony.json").write_text(json.dumps(harm_m, indent=2, ensure_ascii=False))
    print("Harmony:", json.dumps({k: harm_m[k] for k in ("accuracy", "macro_f1", "accuracy_per_timepoint")}, ensure_ascii=False, indent=2))
    print("PCA+HARMONY DONE")


if __name__ == "__main__":
    main()
