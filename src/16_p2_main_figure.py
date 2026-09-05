# -*- coding: utf-8 -*-
"""16 主图：双胚胎时间外推基准总图（6 面板，大写编号 A-F）。
A) 斑马鱼 lineage + 95%CI；B) 斑马鱼 cluster_shared vs majority；
C) 分时间点谱系准确率衰减；D) 分布内对照 vs 跨时间；
E) 小鼠 lineage；F) 小鼠 cluster_shared。
数据：t1_unified_eval / t1_unfrozen_surgery / t1_p2_eval / p4_classical_eval / p4_tf_eval / in_dist_control。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt

from common import MODEL_COLORS, RESULTS
from fig_style import DOUBLE_COL, panel_label, save_fig, setup_nature

setup_nature()

uni = json.loads((RESULTS / "t1_unified_eval.json").read_text())
sur = json.loads((RESULTS / "t1_unfrozen_surgery.json").read_text())
p2 = json.loads((RESULTS / "t1_p2_eval.json").read_text())
ctl = json.loads((RESULTS / "in_dist_control.json").read_text())
ind2 = json.loads((RESULTS / "indist_tf_gf.json").read_text())
m1 = json.loads((RESULTS / "p4_classical_eval.json").read_text())
m2 = json.loads((RESULTS / "p4_tf_eval.json").read_text())
uce = json.loads((RESULTS / "t1_uce_eval.json").read_text())
p4u = json.loads((RESULTS / "p4_uce_eval.json").read_text())

ZFSPLIT = "Zebrafish · train 10–16 hpf → test 19 hpf–10 dpf"
MSSPLIT = "Mouse · train E6.5–7.0 → test E7.25–8.5"

fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, 7.6))


def ci(d, metric):
    boot = d.get("bootstrap95_by_fish") or {}
    return tuple(boot[metric]) if metric in boot else None


def bars(ax, rows, maj, ctx, ctx_left=False):
    """rows: (name, value, ci_or_err, color, bold_tick); maj: majority 值。
    ctx_left: 注释放左上（右侧最高柱的数值标签会顶到注释时用）。"""
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[3] for r in rows]
    for i, (name, v, err, c, bold) in enumerate(rows):
        if err:
            ax.errorbar(i, v, yerr=[[v - err[0]], [err[1] - v]], fmt="none",
                        ecolor="black", elinewidth=0.7, capsize=2.0, alpha=0.85)
    ax.bar(range(len(rows)), vals, color=cols)
    ax.axhline(maj, ls="--", c="#111111", lw=0.9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    for tick, r in zip(ax.get_xticklabels(), rows):
        tick.set_fontweight("bold" if r[4] else "normal")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    species, split = ctx.split(" · ")
    ax.text(0.015 if ctx_left else 0.985, 0.99, f"{species} · {split}\ndashed: majority {maj:.3f}",
            transform=ax.transAxes, ha="left" if ctx_left else "right", va="top",
            fontsize=5.5, color="dimgray", linespacing=1.4)
    top = max((r[1] + (r[2][1] - r[1]) if r[2] else r[1]) for r in rows) + 0.06
    ax.set_ylim(0, min(top + 0.08, 1.08))


# ---------- A 斑马鱼 lineage ----------
ax = axes[0, 0]
rows = [
    ("Majority", uni["majority_lineage_all"], None, MODEL_COLORS["majority"], False),
    ("PCA-50", uni["PCA_frozen"]["lineage_all"]["accuracy"], ci(uni["PCA_frozen"], "lineage_all"), MODEL_COLORS["PCA-50"], False),
    ("Harmony", uni["Harmony_adapted"]["lineage_all"]["accuracy"], ci(uni["Harmony_adapted"], "lineage_all"), MODEL_COLORS["Harmony"], False),
    ("scVI (frozen)", uni["scVI_frozen_summary"]["lineage_all"]["mean"], ci(uni["scVI_frozen_s42"], "lineage_all"), MODEL_COLORS["scVI"], False),
    ("scVI (adapted)", sur["summary"]["lineage_all"]["mean"], ci(sur["s42"], "lineage_all"), MODEL_COLORS["scVI adapted"], False),
    ("STATE SE-600M", uni["STATE_frozen"]["lineage_all"]["accuracy"], ci(uni["STATE_frozen"], "lineage_all"), MODEL_COLORS["STATE SE-600M"], False),
    ("UCE", uce["UCE_frozen"]["lineage_all"]["accuracy"], ci(uce["UCE_frozen"], "lineage_all"), MODEL_COLORS["UCE"], False),
    ("Geneformer V1", p2["Geneformer_frozen"]["lineage_all"]["accuracy"], ci(p2["Geneformer_frozen"], "lineage_all"), MODEL_COLORS["Geneformer V1"], False),
    ("TranscriptFormer", p2["TranscriptFormer_frozen"]["lineage_all"]["accuracy"], ci(p2["TranscriptFormer_frozen"], "lineage_all"), MODEL_COLORS["TranscriptFormer"], True),
]
bars(ax, rows, uni["majority_lineage_all"], ZFSPLIT, ctx_left=True)
ax.set_ylabel("Lineage accuracy")
panel_label(ax, "A")
for i, r in enumerate(rows):
    ytop = r[1] + (r[2][1] - r[1] if r[2] else 0)
    ax.text(i, ytop + 0.02, f"{r[1]:.3f}", ha="center", fontsize=5.5)

# ---------- B 斑马鱼 cluster ----------
ax = axes[0, 1]
rows = [
    ("Majority\n(shared)", uni["majority_cluster_shared_subset"], None, MODEL_COLORS["majority"], False),
    ("PCA-50", uni["PCA_frozen"]["cluster_shared"]["accuracy"], ci(uni["PCA_frozen"], "cluster_shared"), MODEL_COLORS["PCA-50"], False),
    ("Harmony", uni["Harmony_adapted"]["cluster_shared"]["accuracy"], ci(uni["Harmony_adapted"], "cluster_shared"), MODEL_COLORS["Harmony"], False),
    ("scVI", uni["scVI_frozen_summary"]["cluster_shared"]["mean"], ci(uni["scVI_frozen_s42"], "cluster_shared"), MODEL_COLORS["scVI"], False),
    ("STATE", uni["STATE_frozen"]["cluster_shared"]["accuracy"], ci(uni["STATE_frozen"], "cluster_shared"), MODEL_COLORS["STATE SE-600M"], False),
    ("UCE", uce["UCE_frozen"]["cluster_shared"]["accuracy"], ci(uce["UCE_frozen"], "cluster_shared"), MODEL_COLORS["UCE"], False),
    ("Geneformer", p2["Geneformer_frozen"]["cluster_shared"]["accuracy"], ci(p2["Geneformer_frozen"], "cluster_shared"), MODEL_COLORS["Geneformer V1"], False),
    ("TranscriptFormer", p2["TranscriptFormer_frozen"]["cluster_shared"]["accuracy"], ci(p2["TranscriptFormer_frozen"], "cluster_shared"), MODEL_COLORS["TranscriptFormer"], True),
]
bars(ax, rows, uni["majority_cluster_shared_subset"], ZFSPLIT)
ax.set_ylabel("Cluster accuracy (29 shared)")
panel_label(ax, "B")
for i, r in enumerate(rows):
    ytop = r[1] + (r[2][1] - r[1] if r[2] else 0)
    ax.text(i, ytop + 0.004, f"{r[1]:.3f}", ha="center", fontsize=5.5)

# ---------- C 分时间点衰减 ----------
ax = axes[1, 0]
tp_order = ["19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
series = {
    "PCA-50": uni["PCA_frozen"]["accuracy_lineage_per_timepoint"],
    "Harmony": uni["Harmony_adapted"]["accuracy_lineage_per_timepoint"],
    "scVI (frozen)": uni["scVI_frozen_s42"]["accuracy_lineage_per_timepoint"],
    "STATE SE-600M": uni["STATE_frozen"]["accuracy_lineage_per_timepoint"],
    "UCE": uce["UCE_frozen"]["accuracy_lineage_per_timepoint"],
    "Geneformer V1": p2["Geneformer_frozen"]["accuracy_lineage_per_timepoint"],
    "TranscriptFormer": p2["TranscriptFormer_frozen"]["accuracy_lineage_per_timepoint"],
}
colors_c = {
    "PCA-50": MODEL_COLORS["PCA-50"],
    "Harmony": MODEL_COLORS["Harmony"],
    "scVI (frozen)": MODEL_COLORS["scVI"],
    "STATE SE-600M": MODEL_COLORS["STATE SE-600M"],
    "UCE": MODEL_COLORS["UCE"],
    "Geneformer V1": MODEL_COLORS["Geneformer V1"],
    "TranscriptFormer": MODEL_COLORS["TranscriptFormer"],
}
for name, d in series.items():
    ax.plot(range(len(tp_order)), [d[t] for t in tp_order], "o-", label=name,
            color=colors_c[name], ms=3, lw=1.0)
ax.set_xticks(range(len(tp_order)))
ax.set_xticklabels(tp_order)
ax.set_xlabel("Test timepoint (train: 10–16 hpf)")
ax.set_ylabel("Lineage accuracy")
panel_label(ax, "C")
ax.legend(fontsize=5.5, ncol=2, loc="lower left", handlelength=1.6, columnspacing=0.9)
ax.set_ylim(0.1, 1.0)

# ---------- D 分布内 vs 跨时间 ----------
ax = axes[1, 1]
tps = ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
indist_state = [ctl["in_dist_control"][t]["cluster_state_acc"] for t in tps]
indist_pca = [ctl["in_dist_control"][t]["cluster_pca_acc"] for t in tps]
ax.plot(range(10), indist_state, "o-", color=MODEL_COLORS["STATE SE-600M"], ms=3, lw=1.0,
        label="STATE — in-distribution (80/20)")
ax.plot(range(10), indist_pca, "o-", color=MODEL_COLORS["PCA-50"], ms=3, lw=1.0,
        label="PCA — in-distribution (80/20)")
ax.plot(range(10), [ind2["TF"][t] for t in tps], "o-", color=MODEL_COLORS["TranscriptFormer"], ms=3, lw=1.0,
        label="TranscriptFormer — in-distribution")
ax.axhline(uni["STATE_frozen"]["cluster_shared"]["accuracy"], ls="--", color=MODEL_COLORS["STATE SE-600M"], lw=0.9,
           label="STATE — cross-time (10–16hpf → later)")
ax.axhline(uni["PCA_frozen"]["cluster_shared"]["accuracy"], ls="--", color=MODEL_COLORS["PCA-50"], lw=0.9,
           label="PCA — cross-time")
ax.axhline(p2["TranscriptFormer_frozen"]["cluster_shared"]["accuracy"], ls="--", color=MODEL_COLORS["TranscriptFormer"], lw=0.9,
           label="TranscriptFormer — cross-time (0.040)")
ax.set_xticks(range(10))
ax.set_xticklabels(tps, rotation=35, ha="right", fontsize=5.5)
ax.set_ylabel("Cluster accuracy")
panel_label(ax, "D")
ax.legend(fontsize=5.5, loc="center right")

# ---------- E/F 小鼠 ----------
def ms_rows(metric):
    if metric == "lineage_all":
        scvi_val = m1["scVI_summary"]["lineage_all"]["mean"]
        scvi_err = m1["scVI_summary"]["lineage_all"]["std"]
    else:
        scvi_val = m1["scVI_summary"]["cluster_shared"]["mean"]
        scvi_err = m1["scVI_summary"]["cluster_shared"]["std"]
    return [
        ("PCA-50", m1["PCA_frozen"][metric]["accuracy"], ci(m1["PCA_frozen"], metric), MODEL_COLORS["PCA-50"], False),
        ("Harmony", m1["Harmony_adapted"][metric]["accuracy"], ci(m1["Harmony_adapted"], metric), MODEL_COLORS["Harmony"], False),
        ("scVI (adapted)", scvi_val, (max(scvi_val - scvi_err, 0.0), min(scvi_val + scvi_err, 1.0)), MODEL_COLORS["scVI"], False),
        ("UCE", p4u["UCE_mouse"][metric]["accuracy"], ci(p4u["UCE_mouse"], metric), MODEL_COLORS["UCE"], False),
        ("TranscriptFormer", m2["TF_mouse"][metric]["accuracy"], ci(m2["TF_mouse"], metric), MODEL_COLORS["TranscriptFormer"], True),
    ]


ax = axes[2, 0]
rows = ms_rows("lineage_all")
bars(ax, rows, m1["majority_lineage"], MSSPLIT)
ax.set_ylabel("Lineage accuracy")
panel_label(ax, "E")
for i, r in enumerate(rows):
    ytop = r[1] + (r[2][1] - r[1] if r[2] else 0)
    ax.text(i, ytop + 0.02, f"{r[1]:.3f}", ha="center", fontsize=5.5)

ax = axes[2, 1]
rows = ms_rows("cluster_shared")
bars(ax, rows, m1["majority_cluster_shared"], MSSPLIT)
ax.set_ylabel("Cluster accuracy (shared)")
panel_label(ax, "F")
for i, r in enumerate(rows):
    ytop = r[1] + (r[2][1] - r[1] if r[2] else 0)
    ax.text(i, ytop + 0.004, f"{r[1]:.3f}", ha="center", fontsize=5.5)

fig.tight_layout()
save_fig(fig, RESULTS, "fig2_multimodel_benchmark")
