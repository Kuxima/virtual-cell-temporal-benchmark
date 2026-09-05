# -*- coding: utf-8 -*-
"""T1 硬评测（公平协议）：所有模型都是「无监督嵌入 + kNN」，不用 SCANVI 标签。
任务：
  B) 只在训练集出现过的 cluster 上做分类（shared clusters）
  C) 测试集新出现的 cluster：看是否仍映射到正确 lineage
同时重跑 PCA / Harmony / scVI(无监督) / STATE。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from harmony import harmonize
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS, SEED

scvi.settings.seed = SEED
torch.set_float32_matmul_precision("high")
K = 15


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def lognorm(adata):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def knn_pred(Xtr, ytr, Xte):
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def pack(name, y_true, y_pred, mask=None):
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
    }


def eval_model(name, Xtr, Xte, train, test):
    ytr_lin = train.obs["lineage"].astype(str).values
    yte_lin = test.obs["lineage"].astype(str).values
    ytr_clu = train.obs["cluster"].astype(str).values
    yte_clu = test.obs["cluster"].astype(str).values
    shared = np.isin(yte_clu, np.unique(ytr_clu))
    novel = ~shared

    pred_lin = knn_pred(Xtr, ytr_lin, Xte)
    pred_clu = knn_pred(Xtr, ytr_clu, Xte)
    # 新 cluster 的细胞：用预测到的 train-cluster 的多数 lineage 当“父谱系”
    clu2lin = pd.Series(ytr_lin, index=np.arange(len(ytr_lin))).groupby(ytr_clu).agg(lambda s: s.value_counts().index[0])
    pred_lin_via_clu = np.array([clu2lin.get(c, "NA") for c in pred_clu])

    tp = test.obs["timepoint"].astype(str).values
    per_tp_lin = {t: float(accuracy_score(yte_lin[tp == t], pred_lin[tp == t])) for t in sorted(set(tp))}
    per_tp_clu_shared = {}
    for t in sorted(set(tp)):
        m = (tp == t) & shared
        per_tp_clu_shared[t] = float(accuracy_score(yte_clu[m], pred_clu[m])) if m.any() else None

    out = {
        "model": name,
        "lineage_all": pack(name, yte_lin, pred_lin),
        "cluster_shared": pack(name, yte_clu, pred_clu, shared),
        "lineage_on_novel_clusters": pack(name, yte_lin, pred_lin, novel),
        "lineage_via_predicted_cluster_on_novel": pack(name, yte_lin, pred_lin_via_clu, novel),
        "n_test_shared_cluster": int(shared.sum()),
        "n_test_novel_cluster": int(novel.sum()),
        "frac_novel_cluster_cells": float(novel.mean()),
        "accuracy_lineage_per_timepoint": per_tp_lin,
        "accuracy_cluster_shared_per_timepoint": per_tp_clu_shared,
    }
    print(name, "lineage", out["lineage_all"]["accuracy"], "cluster_shared", out["cluster_shared"].get("accuracy"), "novel_lin", out["lineage_on_novel_clusters"].get("accuracy"))
    return out


def main():
    t0 = time.time()
    train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
    test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
    print(f"train {train.shape} test {test.shape}")
    maj = test.obs["lineage"].value_counts(normalize=True).iloc[0]
    maj_c = test.obs["cluster"].value_counts(normalize=True).iloc[0]
    print("majority lineage", float(maj), "majority cluster", float(maj_c))

    reports = {
        "majority_lineage": float(maj),
        "majority_cluster": float(maj_c),
        "n_train_cluster": int(train.obs["cluster"].nunique()),
        "n_test_cluster": int(test.obs["cluster"].nunique()),
        "n_novel_cluster": int(len(set(test.obs["cluster"].astype(str)) - set(train.obs["cluster"].astype(str)))),
    }

    # --- PCA ---
    trn, ten = lognorm(train), lognorm(test)
    pca = PCA(n_components=50, random_state=SEED)
    Xtr = pca.fit_transform(to_dense(trn.X))
    Xte = pca.transform(to_dense(ten.X))
    reports["PCA"] = eval_model("PCA-50 + kNN", Xtr, Xte, train, test)

    # --- Harmony ---
    joint = sc.concat([trn, ten], label="split", keys=["train", "test"], index_unique=None)
    joint.obsm["X_pca"] = np.vstack([Xtr, Xte])
    joint.obsm["X_harmony"] = harmonize(joint.obsm["X_pca"], joint.obs, batch_key="fish")
    H = joint.obsm["X_harmony"]
    reports["Harmony"] = eval_model("Harmony + kNN", H[: train.n_obs], H[train.n_obs :], train, test)

    # --- scVI unsupervised (no SCANVI labels) ---
    print("training unsupervised scVI...")
    scvi.model.SCVI.setup_anndata(train, batch_key="fish")
    vae = scvi.model.SCVI(train, n_latent=30, n_layers=2, gene_likelihood="nb")
    vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
    Xtr_s = vae.get_latent_representation()
    q = scvi.model.SCVI.load_query_data(test, vae)
    q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
    Xte_s = q.get_latent_representation()
    reports["scVI"] = eval_model("scVI unsupervised + kNN", Xtr_s, Xte_s, train, test)

    # --- STATE (already embedded; gene space is human-ortholog, cells aligned by split) ---
    st = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_state.h5ad")
    trs = st[st.obs["split"].astype(str) == "train"]
    tes = st[st.obs["split"].astype(str) == "test"]
    reports["STATE"] = eval_model("STATE SE-600M + kNN", trs.obsm["X_state"], tes.obsm["X_state"], train, test)

    reports["runtime_s"] = round(time.time() - t0, 1)
    (RESULTS / "t1_hard_eval.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print(json.dumps({k: reports[k] if k not in ("PCA", "Harmony", "scVI", "STATE") else {
        "lineage": reports[k]["lineage_all"].get("accuracy"),
        "cluster_shared": reports[k]["cluster_shared"].get("accuracy"),
        "lineage_novel": reports[k]["lineage_on_novel_clusters"].get("accuracy"),
        "frac_novel": reports[k]["frac_novel_cluster_cells"],
    } for k in reports}, indent=2))
    print("HARD EVAL DONE")


if __name__ == "__main__":
    main()
