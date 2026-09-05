# -*- coding: utf-8 -*-
"""32 Statistical grounding for the Figure 3 geometry claim.

For each primary model, in the ORIGINAL embedding space (UMAP distorts
distances; the on-figure arrows remain descriptive):
  - permutation test (B=1000 label shuffles) on the train-vs-late centroid
    displacement, one-sided p;
  - drift table: mean cosine distance from late cells to their 15 nearest
    train cells, versus the within-train 15-NN baseline; ratio = drift index.
Reuses the same embedding caches as everything else.
Output: results/geometry_stats.json"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc

from common import DATA_PROC, RESULTS

SRC = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
BASE = DATA_PROC / "zebrahub_t1_baselines.h5ad"
UMAP_CACHE = DATA_PROC / "zebrahub_t1_umap.h5ad"
K = 15
B = 1000
RNG = np.random.default_rng(42)

SOURCES = [
    ("PCA-50", BASE, "X_pca"),
    ("Harmony", BASE, "X_harmony"),
    ("scVI (seed 42)", BASE, "X_scvi"),
    ("STATE SE-600M", DATA_PROC / "zebrahub_t1_joint_state.h5ad", "X_state"),
    ("UCE", DATA_PROC / "zebrahub_t1_joint_uce.h5ad", "X_uce"),
    ("Geneformer V1", DATA_PROC / "zebrahub_t1_joint_gf.h5ad", "X_gf"),
    ("TranscriptFormer", DATA_PROC / "zebrahub_t1_joint_tf.h5ad", "X_tf"),
]


def norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def mean_knn_dist(Xref, Xq, k=K, chunk=4096):
    Xr = norm_rows(Xref)
    Xq = norm_rows(Xq)
    out = np.empty(len(Xq), dtype=np.float32)
    for s in range(0, len(Xq), chunk):
        e = min(s + chunk, len(Xq))
        sim = Xq[s:e] @ Xr.T
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        out[s:e] = 1.0 - np.take_along_axis(sim, idx, axis=1).mean(axis=1)
    return out


def centroid_disp(X, tr_m, late_m):
    return float(np.linalg.norm(X[tr_m].mean(0) - X[late_m].mean(0)))


def perm_p(X, tr_m, late_m, d_obs, b=B):
    n_tr, n_late = int(tr_m.sum()), int(late_m.sum())
    idx = np.arange(len(X))
    hits = 0
    for _ in range(b):
        perm = RNG.permutation(idx)
        m_tr = np.zeros(len(X), dtype=bool)
        m_tr[perm[:n_tr]] = True
        m_late = np.zeros(len(X), dtype=bool)
        m_late[perm[n_tr:n_tr + n_late]] = True
        if centroid_disp(X, m_tr, m_late) >= d_obs:
            hits += 1
    return (hits + 1) / (b + 1)


def main():
    t0 = time.time()
    ref = sc.read_h5ad(SRC)
    tp = ref.obs["timepoint"].astype(str).to_numpy()
    ref.file.close()
    tr_m = np.isin(tp, ["10hpf", "12hpf", "14hpf", "16hpf"])
    late_m = np.isin(tp, ["5dpf", "10dpf"])

    umap = sc.read_h5ad(UMAP_CACHE)
    stage = umap.obs["stage"].astype(str).to_numpy()

    out = {"protocol": (f"original embedding space; one-sided permutation (B={B}, seed 42) "
                        f"on train-vs-late centroid displacement; drift index = "
                        f"mean cosine distance(late->train k={K}) / within-train kNN baseline"),
           "models": {}}
    key_to_umap = {"X_pca": "X_pca", "X_harmony": "X_harmony", "X_scvi": "X_scvi",
                   "X_state": "X_state", "X_uce": "X_uce", "X_gf": "X_gf", "X_tf": "X_tf"}

    for name, path, key in SOURCES:
        src = sc.read_h5ad(path, backed="r")
        try:
            X = np.asarray(src.obsm[key], dtype=np.float32)
        finally:
            src.file.close()

        d_obs = centroid_disp(X, tr_m, late_m)
        arrow_len_umap = centroid_disp(
            np.asarray(umap.obsm[f"X_umap_{key_to_umap[key]}"], dtype=np.float32),
            stage == "Train 10–16 hpf", stage == "5–10 dpf")

        p = perm_p(X, tr_m, late_m, d_obs)

        d_late = float(mean_knn_dist(X[tr_m], X[late_m]).mean())
        tr_sub = X[tr_m]
        if len(tr_sub) > 20000:  # within-train baseline on a fixed subsample for speed
            sub = RNG.choice(len(tr_sub), 20000, replace=False)
            tr_sub = tr_sub[sub]
        d_base = float(mean_knn_dist(tr_sub, tr_sub).mean())

        out["models"][name] = {
            "centroid_disp_original": round(d_obs, 4),
            "arrow_len_umap": round(arrow_len_umap, 3),
            "perm_p": round(p, 4),
            "d_late_to_train_knn": round(d_late, 4),
            "d_within_train_knn": round(d_base, 4),
            "drift_index": round(d_late / d_base, 3),
        }
        print(f"[{name}] disp {d_obs:.3f} p={p:.3f} drift {d_late/d_base:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    dst = RESULTS / "geometry_stats.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", dst)


if __name__ == "__main__":
    main()
