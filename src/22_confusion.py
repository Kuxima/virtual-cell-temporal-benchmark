# -*- coding: utf-8 -*-
"""22 谱系混淆矩阵（斑马鱼，STATE vs TF，晚期时间点）：
错误模式生物学解读——哪些谱系互相混淆、随时间如何恶化。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS
from fig_style import DOUBLE_COL, panel_header, save_fig, setup_nature

setup_nature()

K = 15
ABBREV = {
    "central_nervous_system": "CNS",
    "endoderm": "Endoderm",
    "hematopoietic_system": "Hemato.",
    "intermediate_mesoderm": "Int. meso.",
    "lateral_mesoderm": "Lat. meso.",
    "mesenchyme": "Mesenchyme",
    "neural_crest": "Neural crest",
    "notochord": "Notochord",
    "paraxial_mesoderm": "Paraxial meso.",
    "periderm": "Periderm",
}

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.9))
summary = {}
for ax, letter, (name, path, key) in zip(axes, ["a", "b"], [
        ("STATE SE-600M", "zebrahub_t1_joint_state.h5ad", "X_state"),
        ("TranscriptFormer", "zebrahub_t1_joint_tf.h5ad", "X_tf")]):
    a = sc.read_h5ad(DATA_PROC / path)
    obs = a.obs.copy()
    for c in ("lineage", "split", "timepoint"):
        obs[c] = obs[c].astype(str)
    X = np.asarray(a.obsm[key], dtype=np.float32)
    tr = (obs["split"] == "train").values
    ytr = obs.loc[tr, "lineage"].values
    # 晚期（外推最远）混淆：2dpf-10dpf 合并
    late = obs["timepoint"].isin(["2dpf", "3dpf", "5dpf", "10dpf"]) & ~tr
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(X[tr], ytr)
    pred = clf.predict(X[late.values])
    yte = obs.loc[late, "lineage"].values
    labels = sorted(set(ytr) | set(yte))
    short = [ABBREV.get(lab, lab) for lab in labels]
    cm = confusion_matrix(yte, pred, labels=labels, normalize="true")

    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v >= 0.02:  # 遮蔽近零注释，降低噪音
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 0.55 else "#1f3b57")

    # 红框高亮 top-3 偏对角错误流（论文核心观察：邻近命运混淆）
    off = [(cm[i, j], i, j) for i in range(len(labels)) for j in range(len(labels)) if i != j]
    off.sort(key=lambda t: -t[0])
    for v, i, j in off[:3]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#D55E00", lw=1.0))
    summary[name] = [(labels[i], labels[j], float(v)) for v, i, j in off[:6]]

    ax.set_xlabel("Predicted label")
    panel_header(ax, letter, name)

axes[0].set_ylabel("True label")
fig.tight_layout()

save_fig(fig, RESULTS, "fig4_confusion_late")
(RESULTS / "confusion_summary.json").write_text(__import__("json").dumps(summary, indent=1))
print("confusion_summary.json updated")
