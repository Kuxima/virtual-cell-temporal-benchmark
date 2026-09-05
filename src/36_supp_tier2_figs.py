# -*- coding: utf-8 -*-
"""36 补充图 S2（合并原 figS3 + figS4）：切分稳健性控制，5 面板大写编号。
  A/B  外推难度曲线（lineage / shared-cluster vs 训练窗右端点）；
  C    19hpf 内插对照（同一批细胞：train≤16hpf 外推 vs train 含 24hpf 内插）；
  D/E  随机 80/20 切分 vs 时间切分（lineage / shared-cluster），量化乐观偏差。
数据来自 results/shifted_window_eval.json 与 results/random_split_ceiling.json。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np

from common import MODEL_COLORS, RESULTS
from fig_style import DOUBLE_COL, panel_label, save_fig, setup_nature

setup_nature()

MODELS = ["PCA-50", "Harmony", "scVI", "STATE SE-600M", "UCE", "Geneformer V1", "TranscriptFormer"]
BOLD = {"Geneformer V1", "TranscriptFormer"}  # 斑马鱼原生预训练
WINDOWS = ["E16", "E19", "E24", "E72"]
WIN_LABEL = {"E16": "16", "E19": "19", "E24": "24", "E72": "72"}


def stylize_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


rep = json.loads((RESULTS / "shifted_window_eval.json").read_text())
cfg = rep["configs"]
rep2 = json.loads((RESULTS / "random_split_ceiling.json").read_text())
cmp2 = rep2["comparison"]

fig = plt.figure(figsize=(DOUBLE_COL, 5.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.8, wspace=0.62)

# ---------- A/B: 外推曲线 ----------
model_handles = None
for slot, metric, ylab, letter in (
        (gs[0, 0:2], "lineage_all", "Lineage accuracy", "A"),
        (gs[0, 2:4], "cluster_shared", "Shared-cluster accuracy", "B")):
    ax = fig.add_subplot(slot)
    maj_key = "majority_lineage" if metric == "lineage_all" else "majority_cluster_shared"
    maj = [cfg[w][maj_key] for w in WINDOWS]
    ax.plot(range(len(WINDOWS)), maj, "--", color="gray", lw=0.8, zorder=1)
    ax.text(len(WINDOWS) - 1.45, maj[-1], "majority", color="gray", fontsize=5.5,
            va="bottom", ha="right", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))
    for m in MODELS:
        ys = [cfg[w]["models"][m][metric]["accuracy"] for w in WINDOWS]
        ax.plot(range(len(WINDOWS)), ys, "-o", color=MODEL_COLORS[m], lw=1.0,
                ms=2.8, label=m, zorder=3)
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([f"≤{WIN_LABEL[w]} hpf" for w in WINDOWS], fontsize=5.5)
    ax.set_xlabel("Training-set right endpoint")
    ax.set_ylabel(ylab)
    panel_label(ax, letter)
    stylize_ax(ax)
    if model_handles is None:
        model_handles, _ = ax.get_legend_handles_labels()
# 六模型共享图例放到图形顶部，避免压住任何数据线
fig.legend(handles=model_handles, loc="upper center", ncol=6, fontsize=5.0,
           frameon=False, bbox_to_anchor=(0.5, 1.005), handlelength=1.3,
           columnspacing=1.0, handletextpad=0.4)

# ---------- C: 19hpf 内插对照（同一批 5,899 细胞） ----------
ax = fig.add_subplot(gs[0, 4:6])
y_ext, y_int = [], []
for m in MODELS:
    y_ext.append(cfg["E16"]["models"][m]["accuracy_lineage_per_timepoint"]["19hpf"])
    y_int.append(cfg["I19"]["models"][m]["lineage_all"]["accuracy"])
y_ext, y_int = np.array(y_ext), np.array(y_int)
order = np.argsort(-y_ext)
for i, mi in enumerate(order):
    m = MODELS[mi]
    ax.plot([i, i], [y_ext[mi], y_int[mi]], color=MODEL_COLORS[m], lw=1.0,
            alpha=0.8, zorder=2)
    ax.scatter(i, y_ext[mi], s=16, facecolor="white", edgecolor=MODEL_COLORS[m],
               lw=1.0, zorder=3, label="_")
    ax.scatter(i, y_int[mi], s=16, color=MODEL_COLORS[m], zorder=4)
    delta = y_int[mi] - y_ext[mi]
    ax.annotate(f"{delta:+.2f}", (i, max(y_ext[mi], y_int[mi])),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=5.0, color=MODEL_COLORS[m])
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODELS[i] if MODELS[i] not in BOLD else MODELS[i] + "\n(native)"
                    for i in order], fontsize=5.0, rotation=30, ha="right")
for tick, mi in zip(ax.get_xticklabels(), order):
    if MODELS[mi] in BOLD:
        tick.set_fontweight("bold")
ax.scatter([], [], s=14, facecolor="white", edgecolor="0.2", lw=1.0,
           label="extrapolation (train ≤ 16 hpf)")
ax.scatter([], [], s=14, color="0.2", label="interpolation (train adds 24 hpf)")
# 图内已无整块空白（所有列都有高段），图例放到面板下方图外
ax.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.62),
          ncol=1, handlelength=1.0, labelspacing=0.25)
ax.set_ylabel("Lineage accuracy on the same 19-hpf cells")
panel_label(ax, "C")
stylize_ax(ax)

# ---------- D/E: 随机切分天花板 ----------
# xlim 放宽到 1.18：最长条 0.97 的条外数值标签仍留在面板内，
# E 的长 y 刻度名（如 TranscriptFormer）也有完整间隙、不再压到 D 的条端
order2 = sorted(MODELS, key=lambda m: -cmp2[m]["lineage_time"])
for slot, key, ylab, letter in (
        (gs[1, 0:3], "lineage", "Lineage accuracy", "D"),
        (gs[1, 3:6], "cluster", "Shared-cluster accuracy", "E")):
    ax = fig.add_subplot(slot)
    ys_t = [cmp2[m][f"{key}_time"] for m in order2]
    ys_r = [cmp2[m][f"{key}_random"] for m in order2]
    y = np.arange(len(order2))
    ax.barh(y + 0.18, ys_t, height=0.34, color=[MODEL_COLORS[m] for m in order2], zorder=3)
    ax.barh(y - 0.18, ys_r, height=0.34, color=[MODEL_COLORS[m] for m in order2],
            alpha=0.45, hatch="//", zorder=3)
    for yi, (t, r) in enumerate(zip(ys_t, ys_r)):
        ax.text(t + 0.012, yi + 0.18, f"{t:.2f}", va="center", fontsize=5.0, color="0.1")
        ax.text(r + 0.012, yi - 0.18, f"{r:.2f}", va="center", fontsize=5.0, color="0.35")
    ax.set_yticks(y)
    ax.set_yticklabels(order2)
    for tick, m in zip(ax.get_yticklabels(), order2):
        if m in BOLD:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(ylab)
    ax.set_xlim(0, 1.18)
    panel_label(ax, letter)
    stylize_ax(ax)
from matplotlib.patches import Patch
# 面板内已无整块空白（随机切分条普遍很长），temporal/random 图例放 E 下方图外
fig.axes[-1].legend(handles=[Patch(facecolor="0.25", label="temporal split (train ≤ 16 hpf)"),
                             Patch(facecolor="0.25", alpha=0.45, hatch="//", label="random 80/20 split")],
                    fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.38),
                    ncol=2, frameon=False, handlelength=1.2, columnspacing=1.2)

save_fig(fig, RESULTS, "figS2_split_robustness")
