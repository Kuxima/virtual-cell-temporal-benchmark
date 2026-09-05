# -*- coding: utf-8 -*-
"""31 Linear-probe replication of the headline zebrafish metrics.

Answers the "kNN-only probe" limitation: refit a multinomial logistic
regression (StandardScaler + LogisticRegression, seed 42) on the SAME frozen
embeddings used everywhere else and re-evaluate lineage_all, cluster_shared
and lineage-on-novel-clusters. Protocol otherwise identical to 11b/17b.

Embedding sources are the shared caches; no model is re-run.
Output: results/linear_probe_eval.json (+ supp table feed)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA_PROC, RESULTS

SRC = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
BASE = DATA_PROC / "zebrahub_t1_baselines.h5ad"

SOURCES = [
    ("PCA-50", BASE, "X_pca"),
    ("Harmony", BASE, "X_harmony"),
    ("scVI (seed 42)", BASE, "X_scvi"),
    ("STATE SE-600M", DATA_PROC / "zebrahub_t1_joint_state.h5ad", "X_state"),
    ("UCE", DATA_PROC / "zebrahub_t1_joint_uce.h5ad", "X_uce"),
    ("Geneformer V1", DATA_PROC / "zebrahub_t1_joint_gf.h5ad", "X_gf"),
    ("TranscriptFormer", DATA_PROC / "zebrahub_t1_joint_tf.h5ad", "X_tf"),
]


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro"))}


def main():
    t0 = time.time()
    ref = sc.read_h5ad(SRC)
    obs = ref.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    ref.file.close()
    tr_m = (obs["split"] == "train").values

    lin_classes = sorted(set(obs.loc[tr_m, "lineage"]))
    lin_map = {c: i for i, c in enumerate(lin_classes)}
    y_lin = obs["lineage"].map(lin_map).to_numpy()

    clu_classes = sorted(set(obs.loc[tr_m, "cluster"]))
    clu_map = {c: i for i, c in enumerate(clu_classes)}
    y_clu = obs["cluster"].map(clu_map).to_numpy()          # test-novel -> NaN
    shared = ~np.isnan(y_clu)
    y_clu_fit = np.nan_to_num(y_clu, nan=-1).astype(int)

    novel = np.zeros(len(obs), dtype=bool)
    novel[~tr_m] = ~shared[~tr_m]

    out = {
        "protocol": ("multinomial logistic regression (StandardScaler + LogisticRegression, "
                     "lbfgs, max_iter=5000, seed 42) fitted on train cells only; identical "
                     "embedding caches and label masks as the kNN protocol"),
        "models": {},
    }

    for name, path, key in SOURCES:
        t1 = time.time()
        src = sc.read_h5ad(path, backed="r")
        try:
            assert src.n_obs == len(obs), name
            names = src.obs_names.astype(str)
            X = np.asarray(src.obsm[key], dtype=np.float32)
        finally:
            src.file.close()
        assert np.array_equal(names, obs.index.astype(str)), f"row order mismatch: {name}"

        res = {}
        for task, yfit, yeval_mask in (
            ("lineage_all", y_lin, np.ones(len(obs), dtype=bool)),
            ("cluster_shared", y_clu_fit, shared),
        ):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, random_state=42)),
            ])
            fit_m = tr_m if task == "lineage_all" else tr_m  # train never has novel cluster
            clf.fit(X[fit_m], yfit[fit_m])
            pred = clf.predict(X)
            m = (~tr_m) & yeval_mask
            if task == "lineage_all":
                res["lineage_all"] = metrics(y_lin[m], pred[m])
                res["lineage_on_novel_clusters"] = metrics(y_lin[novel], pred[novel])
            else:
                res["cluster_shared"] = metrics(y_clu[shared].astype(int), pred[shared])
            print(f"[{name}] {task} done ({time.time()-t1:.0f}s)", flush=True)

        out["models"][name] = res
        print(f"[{name}] lineage {res['lineage_all']['accuracy']:.3f} "
              f"cluster {res['cluster_shared']['accuracy']:.3f} "
              f"novel-lin {res['lineage_on_novel_clusters']['accuracy']:.3f}", flush=True)

    out["runtime_s"] = round(time.time() - t0, 1)
    dst = RESULTS / "linear_probe_eval.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", dst)


if __name__ == "__main__":
    main()
