# -*- coding: utf-8 -*-
"""对 STATE 嵌入做 T1 kNN 时序外推评测，并重画含 STATE 的对比图。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS

LABEL = "lineage"
K = 15
ORDER = ["19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]


def main():
    adata = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_state.h5ad")
    keys = list(adata.obsm.keys())
    print("obsm", keys)
    key = "X_state" if "X_state" in adata.obsm else keys[0]
    print("using", key, adata.obsm[key].shape)
    train = adata[adata.obs["split"].astype(str) == "train"]
    test = adata[adata.obs["split"].astype(str) == "test"]
    y_train = train.obs[LABEL].astype(str).values
    y_test = test.obs[LABEL].astype(str).values
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(train.obsm[key], y_train)
    pred = clf.predict(test.obsm[key])
    tp = test.obs["timepoint"].astype(str).values
    per_tp = {str(t): float(accuracy_score(y_test[tp == t], pred[tp == t])) for t in sorted(set(tp))}
    metrics = {
        "model": "STATE SE-600M + kNN (human orthologs, batch=8)",
        "embedding_key": key,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(adata.n_vars),
        "label": LABEL,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy_per_timepoint": per_tp,
        "n_novel_types": int(len(set(y_test) - set(y_train))),
    }
    (RESULTS / "t1_state.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    files = {
        "PCA": "t1_pca.json",
        "Harmony": "t1_harmony.json",
        "scVI/scANVI": "t1_scvi_baseline.json",
        "STATE SE-600M": "t1_state.json",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=150)
    for name, fn in files.items():
        d = json.loads((RESULTS / fn).read_text(encoding="utf-8"))
        ys = [d["accuracy_per_timepoint"][t] for t in ORDER]
        ax.plot(ORDER, ys, marker="o", label=f"{name}  ({d['accuracy']:.1%})")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Lineage accuracy")
    ax.set_xlabel("Held-out developmental stage")
    ax.set_title("T1 temporal extrapolation on Zebrahub (train: 10–16 hpf)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = RESULTS / "t1_baseline_per_timepoint.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("plot", out)


if __name__ == "__main__":
    main()
