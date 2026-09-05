# -*- coding: utf-8 -*-
"""30 补充图 S1（合并原 fig6 + fig7）：分布偏移诊断，3 面板大写编号。
  A  哑铃图——每个模型"分布内检索上限"（10 个时间点内 80/20 中位数）vs 跨时间外推；
  B  斑马鱼 OOD AUROC 网格（点面积 ∝ novel 细胞数）；
  C  小鼠 OOD AUROC（按 stage）。
  B/C 共享 RdYlBu 色带（色盲安全：红≈机会水平 0.5，蓝≈强）。
数据：in_dist_control / indist_tf_gf / p4_tf_eval / t1_unified_eval / t1_p2_eval / ood_auroc_models。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from common import DATA_PROC, MODEL_COLORS, RESULTS
from fig_style import DOUBLE_COL, panel_label, save_fig, setup_nature

setup_nature()

uni = json.loads((RESULTS / "in_dist_control.json").read_text())
ind2 = json.loads((RESULTS / "indist_tf_gf.json").read_text())
p4t = json.loads((RESULTS / "p4_tf_eval.json").read_text())
t1u = json.loads((RESULTS / "t1_unified_eval.json").read_text())
t1p2 = json.loads((RESULTS / "t1_p2_eval.json").read_text())
auroc = json.loads((RESULTS / "ood_auroc_models.json").read_text())
ucej = json.loads((RESULTS / "t1_uce_eval.json").read_text())
udiag = json.loads((RESULTS / "uce_diag.json").read_text())
auroc["UCE"] = {"global": udiag["UCE"]["ood"]["global"],
                "per_timepoint": {k: v for k, v in udiag["UCE"]["ood"].items() if k != "global"}}

CROSS_TIME = {
    "PCA-50": t1u["PCA_frozen"]["cluster_shared"]["accuracy"],
    "STATE SE-600M": t1u["STATE_frozen"]["cluster_shared"]["accuracy"],
    "UCE": ucej["UCE_frozen"]["cluster_shared"]["accuracy"],
    "Geneformer V1": t1p2["Geneformer_frozen"]["cluster_shared"]["accuracy"],
    "TranscriptFormer": t1p2["TranscriptFormer_frozen"]["cluster_shared"]["accuracy"],
}
TPS_ALL = ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
TEST_TPS = ["19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]  # OOD is defined on test cells only


def indist_median(model):
    if model == "PCA-50":
        vals = [uni["in_dist_control"][t]["cluster_pca_acc"] for t in TPS_ALL]
    elif model == "STATE SE-600M":
        vals = [uni["in_dist_control"][t]["cluster_state_acc"] for t in TPS_ALL]
    elif model == "UCE":
        vals = [udiag["UCE"]["in_dist"][t] for t in TPS_ALL]
    elif model == "Geneformer V1":
        vals = [ind2["Geneformer"][t] for t in TPS_ALL]
    else:
        vals = [ind2["TF"][t] for t in TPS_ALL]
    return float(np.median(vals))


def novel_counts_zf():
    return uni["novel_cluster_timepoint_counts"]


def novel_counts_mouse():
    """Novel-cluster cell counts per test stage, from obs metadata only."""
    a = sc.read_h5ad(DATA_PROC / "mouse_t1_joint.h5ad", backed="r")
    try:
        stage = a.obs["stage"].astype(str).values
        clu = a.obs["cluster"].astype(str).values
        split = a.obs["split"].astype(str).values
    finally:
        a.file.close()
    train_c = set(clu[split == "train"])
    counts = {}
    for st in sorted(set(stage[split == "test"])):
        m = (stage == st) & (split == "test")
        counts[st] = int(np.isin(clu[m], list(train_c), invert=True).sum())
    return counts


fig = plt.figure(figsize=(DOUBLE_COL, 6.3))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.25, 0.7], hspace=0.55)

# ---------- A 哑铃图 ----------
ax = fig.add_subplot(gs[0])
rows = []
for m, ct in CROSS_TIME.items():
    med = indist_median(m)
    rows.append((m, ct, med, med / ct))
rows.sort(key=lambda r: r[3], reverse=True)
for i, (m, ct, med, ratio) in enumerate(rows):
    y = len(rows) - 1 - i
    c = MODEL_COLORS[m]
    ax.plot([ct, med], [y, y], color=c, lw=1.2, alpha=0.8, zorder=1)
    ax.scatter([med], [y], s=30, color=c, zorder=2)
    ax.scatter([ct], [y], s=30, facecolors="white", edgecolors=c, linewidths=0.8, zorder=2)
    ax.text(med + 0.012, y, f"{med:.2f}", va="center", fontsize=5.5, color=c)
    ax.text(ct, y + 0.28, f"{ratio:.0f}×", ha="center", va="bottom",
            fontsize=5.5, color=c, zorder=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_ylabel("Cluster accuracy (shared clusters)")
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.7, len(rows) - 0.35)
maj = uni["majority_cluster_shared_subset"]
ax.axvline(maj, ls="--", lw=0.8, color="#111111")
ax.text(maj + 0.008, -0.62, "majority baseline", fontsize=5.5, color="#111111", va="center")
panel_label(ax, "A")

norm = Normalize(vmin=0.15, vmax=1.0)
cmap = matplotlib.colormaps["RdYlBu"]  # 色盲安全散度色带：红≈机会水平，蓝≈强

# ---------- B 斑马鱼 OOD ----------
ax = fig.add_subplot(gs[1])
models = ["STATE", "TF", "Geneformer", "UCE"]
disp = {"STATE": "STATE SE-600M", "TF": "TranscriptFormer", "Geneformer": "Geneformer V1", "UCE": "UCE"}
n_by_tp = novel_counts_zf()
vmax_n = max(n_by_tp.values())
for yi, m in enumerate(models):
    for xi, tp in enumerate(TEST_TPS):
        v = auroc[m]["per_timepoint"][tp]
        n = n_by_tp.get(tp, 0)
        ax.scatter(xi, yi, s=24 + 300 * n / vmax_n, c=[v], cmap=cmap, norm=norm,
                   edgecolors="#444444", linewidths=0.5, zorder=2)
        ax.text(xi + 0.34, yi, f"{v:.2f}", fontsize=5, va="center")
ax.axhline(0.5, color="#eeeeee", lw=0.4)
ax.axhline(1.5, color="#bbbbbb", lw=0.6)
ax.axhline(2.5, color="#eeeeee", lw=0.4)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([disp[m] for m in models])
ax.set_xticks(range(len(TEST_TPS)))
ax.set_xticklabels(TEST_TPS, rotation=40, ha="right")
ax.set_ylim(-0.6, len(models) - 0.4)
ax.set_xlim(-0.6, len(TEST_TPS) - 0.4)
panel_label(ax, "B")

# ---------- C 小鼠 OOD(TranscriptFormer + UCE 两行) ----------
ax = fig.add_subplot(gs[2])
p4u = json.loads((RESULTS / "p4_uce_eval.json").read_text())
ms_rows = [("TranscriptFormer", p4t["ood_auroc"]), ("UCE", p4u["ood_auroc"])]
ms_tps = [k for k in ms_rows[0][1] if k != "global"]
mcounts = novel_counts_mouse()
nmax = max(mcounts.get(t, 1) for t in ms_tps)
for yi, (_mname, mc) in enumerate(ms_rows):
    for xi, tp in enumerate(ms_tps):
        if tp not in mc:
            continue
        v = mc[tp]
        n = mcounts.get(tp, 0)
        ax.scatter(xi, yi, s=24 + 300 * n / nmax, c=[v], cmap=cmap, norm=norm,
                   edgecolors="#444444", linewidths=0.5)
        ax.text(xi + 0.28, yi, f"{v:.2f}", fontsize=5, va="center")
ax.axhline(0.5, color="#bbbbbb", lw=0.6)
ax.set_yticks(range(len(ms_rows)))
ax.set_yticklabels([m for m, _ in ms_rows])
ax.set_xticks(range(len(ms_tps)))
ax.set_xticklabels(ms_tps, rotation=40, ha="right")
ax.set_ylim(-0.85, len(ms_rows) - 0.15)
ax.set_xlim(-0.6, len(ms_tps) - 0.4)
panel_label(ax, "C")

for ax in (fig.axes[1], fig.axes[2]):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.4, zorder=0)

# 固定 figure 坐标的共享色条：不 steal 子图空间
cbar_ax = fig.add_axes([0.18, 0.035, 0.64, 0.02])
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, orientation="horizontal")
cbar.set_label("OOD AUROC (red ≈ chance 0.5, blue ≈ strong)", fontsize=6)
cbar.ax.tick_params(labelsize=5.5)

save_fig(fig, RESULTS, "figS1_shift_diagnostics")
