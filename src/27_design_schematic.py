# -*- coding: utf-8 -*-
"""27 Fig 1 研究设计示意图（Nature 规格）：a) 两数据集时间切分；b) 统一评测流程。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from common import RESULTS
from fig_style import DOUBLE_COL, panel_label, save_fig, setup_nature

setup_nature()

TRAIN_C, TEST_C = "#0072B2", "#E69F00"  # Okabe-Ito blue / orange（色盲安全）

fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 6.2), gridspec_kw={"height_ratios": [1.15, 1.0]})

# ---------------- a) 时间切分 ----------------
ax = axes[0]
ax.set_xlim(-0.6, 11.9)
ax.set_ylim(-1.0, 3.4)
ax.axis("off")
panel_label(ax, "a", x=-0.02)

# 斑马鱼（上行）
zf = [("10hpf", 1), ("12hpf", 2), ("14hpf", 3), ("16hpf", 4), ("19hpf", 5), ("24hpf", 6),
      ("2dpf", 7), ("3dpf", 8), ("5dpf", 9), ("10dpf", 10)]
zf_train = {"10hpf", "12hpf", "14hpf", "16hpf"}
for name, x in zf:
    tr = name in zf_train
    ax.add_patch(FancyBboxPatch((x - 0.42, 2.05), 0.84, 0.62,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=TRAIN_C if tr else TEST_C, ec="white", lw=0.8, alpha=0.92))
    ax.text(x, 2.36, name, ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
ax.text(-0.55, 2.36, "Zebrafish\nZebrahub", ha="left", va="center", fontsize=6.0, fontweight="bold")
ax.text(2.5, 1.72, "train  18,109 cells (10–16 hpf)", ha="center", fontsize=6.0, color=TRAIN_C, fontweight="bold")
ax.text(8.0, 1.72, "test  102,335 cells (19 hpf–10 dpf)", ha="center", fontsize=6.0, color="#c87f00", fontweight="bold")

# 小鼠（下行）
ms = [("E6.5", 1), ("E6.75", 2), ("E7.0", 3), ("E7.25", 4), ("E7.5", 5), ("E7.75", 6),
      ("E8.0", 7), ("E8.25", 8), ("E8.5", 9)]
ms_train = {"E6.5", "E6.75", "E7.0"}
for name, x in ms:
    tr = name in ms_train
    ax.add_patch(FancyBboxPatch((x - 0.42, 0.15), 0.84, 0.62,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=TRAIN_C if tr else TEST_C, ec="white", lw=0.8, alpha=0.92))
    ax.text(x, 0.46, name, ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
ax.text(-0.55, 0.46, "Mouse\nE-MTAB-6967", ha="left", va="center", fontsize=6.0, fontweight="bold")
ax.text(2.0, -0.18, "train  20,308 cells", ha="center", fontsize=6.0, color=TRAIN_C, fontweight="bold")
ax.text(7.0, -0.18, "test  88,549 cells (E7.25–E8.5)", ha="center", fontsize=6.0, color="#c87f00", fontweight="bold")

# 发育时间箭头
ax.annotate("", xy=(11.7, 3.1), xytext=(0.8, 3.1),
            arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.8))
ax.text(6.2, 3.28, "developmental time", ha="center", fontsize=6.0, color="gray", style="italic")
ax.text(-0.45, -0.75, "No fish / embryo overlap between train and test; "
        "HVGs selected on train only; novel late states withheld entirely.",
        ha="left", fontsize=5.5, color="dimgray")

# ---------------- b) 流程 ----------------
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
panel_label(ax, "b", x=-0.02)


def box(x, y, w, h, text, fc="#f2f2f2", fs=6.2, bold=False, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, color="dimgray", lw=0.8))


box(0.2, 6.4, 1.9, 2.4, "Raw atlases\ncounts + labels\n(lineage, cluster)", fc="#e8eef7", bold=False)
box(2.8, 6.4, 2.1, 2.4, "Gene\nharmonization\n2,581 ortholog-derived\nhuman symbols\n(native IDs for TF)", fc="#e8eef7")
box(5.5, 7.6, 2.6, 1.2, "7 frozen models\nPCA · Harmony · scVI\nSTATE · UCE · Geneformer · TF", fc="#fde9e9", bold=True)
box(5.5, 5.9, 2.6, 1.2, "adaptation variants\nscArches surgery (default)\nfully-unfrozen query training", fc="#fff5e6")
box(8.4, 6.4, 1.4, 2.4, "kNN transfer\ncosine, k=15\ndistance-\nweighted", fc="#e8f5e9", bold=True)
arrow(2.1, 7.6, 2.8, 7.6)
arrow(4.9, 7.6, 5.5, 8.2)
arrow(4.9, 7.3, 5.5, 6.5)
arrow(8.1, 8.2, 8.4, 7.9)
arrow(8.1, 6.5, 8.4, 6.9)

box(0.2, 2.6, 2.9, 2.2, "Coarse task\nlineage (10 / 8 classes)\naccuracy + 95% CI\n(bootstrap by fish/embryo)", fc="#f7f7f7")
box(3.6, 2.6, 2.9, 2.2, "Fine-grained task\ncluster identity\nshared clusters (29 / 11)\nvs majority baseline", fc="#f7f7f7")
box(7.0, 2.6, 2.8, 2.2, "Controls & diagnostics\nin-distribution 80/20\nOOD AUROC (15-NN dist)\nconfusion · k-sensitivity", fc="#f7f7f7")
# 先垂直引出到模型盒下方，再扇出到三个任务盒，避免穿过 adaptation 盒
arrow(9.05, 6.4, 9.05, 5.5)
arrow(9.05, 5.5, 8.4, 4.8)
arrow(9.05, 5.5, 5.1, 4.8)
arrow(9.05, 5.5, 1.7, 4.8)

ax.text(0.2, 1.5, "Gene-set control: TF re-embedded on the corresponding 2,994 native genes\n"
        "recoverable from the ortholog-derived set; robustness: k in {5,15,30}, "
        "3 scVI seeds, unified 2,581-gene input.",
        fontsize=5.5, color="dimgray", va="top")

fig.tight_layout()
save_fig(fig, RESULTS, "fig1_design")
