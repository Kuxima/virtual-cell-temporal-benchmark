# -*- coding: utf-8 -*-
"""11 分布内对照实验 + 数据划分完整性检查。

目的：回答"cluster_shared 准确率≈随机（3.7-6.9%）到底是
  (a) 模型嵌入无法区分细粒度 cluster（协议/模型问题），还是
  (b) 跨时间迁移本身就失败（真实的外推发现）"。

做法：对每个时间点，把该时间点细胞按 cluster 分层 80/20 切分，
在 80% 上做 kNN（k=15, cosine, distance-weighted），预测 20%：
  - STATE 嵌入（已有，零额外推理）
  - PCA-50（humanorth logcounts，仅 fit 在 80% 上，无泄漏）
若同时间点 cluster 准确率高而跨时间低 → 结论 (b) 成立。

附带：
  A) 划分完整性：时间点/鱼批次重叠、lineage 覆盖、cluster 全局 ID 检验
  B) cluster_shared 的正确 majority 基线（在相同子集上）
  C) 新颖性检测 AUROC（STATE 嵌入到训练集的 15-NN 平均距离；
     全局 + 按时间点分层以消除时间漂移混淆）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS, SEED

K = 15


def knn_acc(Xtr, ytr, Xte, yte):
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return float(accuracy_score(yte, pred)), float(balanced_accuracy_score(yte, pred))


def mean_knn_dist(Xref, Xq, k=K, chunk=4096):
    """cosine 距离的 k-NN 平均距离，分块 BLAS matmul（sklearn brute 更慢）。"""
    Xr = Xref / (np.linalg.norm(Xref, axis=1, keepdims=True) + 1e-12)
    Xq = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(Xq), dtype=np.float32)
    for s in range(0, len(Xq), chunk):
        e = min(s + chunk, len(Xq))
        sim = Xq[s:e] @ Xr.T  # cosine similarity
        # 取 top-k 相似（含自身所在类参考中无自身，k 直接取）
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        top = np.take_along_axis(sim, idx, axis=1)
        out[s:e] = 1.0 - top.mean(axis=1)
    return out


def main():
    t0 = time.time()
    st = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_state.h5ad")
    ho = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    assert (st.obs.index == ho.obs.index).all(), "joint_state 与 joint_humanorth 细胞顺序不一致"
    Xs = np.asarray(st.obsm["X_state"], dtype=np.float32)
    obs = st.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    print(f"joint {st.shape}, X_state {Xs.shape}, genes(ortholog) {ho.n_vars}")

    rep = {"X_state_dim": int(Xs.shape[1]), "n_genes_ortholog": int(ho.n_vars)}

    # ---------- A) 划分完整性 ----------
    tr, te = obs["split"] == "train", obs["split"] == "test"
    fish_tr, fish_te = set(obs.loc[tr, "fish"]), set(obs.loc[te, "fish"])
    lin_tr, lin_te = set(obs.loc[tr, "lineage"]), set(obs.loc[te, "lineage"])
    clu_tr, clu_te = set(obs.loc[tr, "cluster"]), set(obs.loc[te, "cluster"])
    rep["split_check"] = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "train_timepoints": sorted(set(obs.loc[tr, "timepoint"])),
        "test_timepoints": sorted(set(obs.loc[te, "timepoint"])),
        "fish_overlap": sorted(fish_tr & fish_te),
        "lineage_train_missing_in_test": sorted(lin_tr - lin_te),
        "lineage_test_missing_in_train": sorted(lin_te - lin_tr),
        "n_cluster_train": len(clu_tr), "n_cluster_test": len(clu_te),
        "n_cluster_novel": len(clu_te - clu_tr),
    }
    print(json.dumps(rep["split_check"], ensure_ascii=False, indent=1))

    # cluster ID 是否全局（跨时间点复用）
    cnt_tp = obs.groupby("cluster")["timepoint"].nunique()
    rep["cluster_global_check"] = {
        "frac_clusters_in_ge2_timepoints": float((cnt_tp >= 2).mean()),
        "n_clusters": int(len(cnt_tp)),
    }
    print("cluster 跨时间点复用比例:", rep["cluster_global_check"])

    # novel cluster 的时间点分布
    novel_clu = clu_te - clu_tr
    nov_mask_cells = obs["cluster"].isin(novel_clu).values & te.values
    rep["novel_cluster_timepoint_counts"] = obs.loc[nov_mask_cells, "timepoint"].value_counts().to_dict()
    print("novel cluster 细胞按时间点:", rep["novel_cluster_timepoint_counts"])

    # ---------- B) cluster_shared 的正确 majority 基线 ----------
    yte_clu = obs.loc[te, "cluster"].values
    ytr_clu = obs.loc[tr, "cluster"].values
    shared = np.isin(yte_clu, np.unique(ytr_clu))
    maj_cluster_train = pd.Series(ytr_clu).value_counts().index[0]
    rep["majority_cluster_shared_subset"] = float((yte_clu[shared] == maj_cluster_train).mean())
    print(f"cluster_shared 子集上的 majority 基线: {rep['majority_cluster_shared_subset']:.4f}")

    # ---------- C) 分布内对照（核心） ----------
    tp_all = obs["timepoint"].values
    y_all_clu = obs["cluster"].values
    y_all_lin = obs["lineage"].values
    Xh = ho.X
    Xh = Xh.toarray() if sparse.issparse(Xh) else np.asarray(Xh)  # 120444x2581, ~1.2GB
    control = {}
    for t in sorted(set(tp_all)):
        m = tp_all == t
        idx = np.where(m)[0]  # 全局行号，升序
        yc = y_all_clu[idx]
        if len(idx) < 200 or len(set(yc)) < 2:
            continue
        idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=yc)
        pos_tr = np.searchsorted(idx, idx_tr)  # 全局行号 -> 本时间点内局部位置
        pos_te = np.searchsorted(idx, idx_te)
        # STATE
        acc_s, bacc_s = knn_acc(Xs[idx_tr], y_all_clu[idx_tr], Xs[idx_te], y_all_clu[idx_te])
        acc_sl, _ = knn_acc(Xs[idx_tr], y_all_lin[idx_tr], Xs[idx_te], y_all_lin[idx_te])
        # PCA（fit 在 80% 上）
        sub = sc.AnnData(X=Xh[idx].astype(np.float32))
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        pca = PCA(n_components=50, random_state=SEED)
        Ztr = pca.fit_transform(sub.X[pos_tr])
        Zte = pca.transform(sub.X[pos_te])
        acc_p, bacc_p = knn_acc(Ztr, y_all_clu[idx_tr], Zte, y_all_clu[idx_te])
        maj = pd.Series(yc).value_counts(normalize=True).iloc[0]
        control[t] = {
            "n": int(len(idx)), "n_cluster": int(len(set(yc))),
            "cluster_majority": float(maj),
            "cluster_state_acc": acc_s, "cluster_state_bacc": bacc_s,
            "cluster_pca_acc": acc_p,
            "lineage_state_acc": acc_sl,
        }
        print(f"[{t}] n={len(idx)} clusters={len(set(yc))} maj={maj:.3f} | "
              f"STATE cluster acc={acc_s:.3f} (bacc {bacc_s:.3f}) | "
              f"PCA cluster acc={acc_p:.3f} | STATE lineage acc={acc_sl:.3f}")
    rep["in_dist_control"] = control

    # ---------- D) 新颖性检测 AUROC（STATE） ----------
    Xtr_s, Xte_s = Xs[tr.values], Xs[te.values]
    dist = mean_knn_dist(Xtr_s, Xte_s)
    nov = ~shared  # novel cluster 细胞
    rep["ood_auroc_global"] = float(roc_auc_score(nov.astype(int), dist))
    print(f"OOD AUROC (global, novel vs shared): {rep['ood_auroc_global']:.3f}")
    # 按时间点分层（消除"后期时间点距离天然更远"的混淆）
    tp_te = tp_all[te.values]
    per_tp = {}
    for t in sorted(set(tp_te)):
        m = tp_te == t
        if nov[m].sum() > 20 and (~nov[m]).sum() > 20:
            per_tp[t] = float(roc_auc_score(nov[m].astype(int), dist[m]))
    rep["ood_auroc_per_timepoint"] = per_tp
    print("OOD AUROC 按时间点:", {k: round(v, 3) for k, v in per_tp.items()})

    rep["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS / "in_dist_control.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
