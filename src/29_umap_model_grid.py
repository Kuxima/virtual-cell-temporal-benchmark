# -*- coding: utf-8 -*-
"""29 UMAP small-multiples grid over all six primary models.

Identical-layout convention (UCE / scGPT / Genome-Biol-2025): one UMAP per
model fitted independently on the full 120,444-cell embedding with identical
scanpy parameters, colored by a shared four-level stage palette, plus a
PRESCIENT-style centroid arrow (train window -> late-test bucket) to show how
far future embryos drift out of the reference manifold.

Layout mirrors the three-generations narrative: top row classical
(PCA/Harmony/scVI), bottom row foundation models (STATE/Geneformer/TF).
UMAP coordinates are cached in data/processed/zebrahub_t1_umap.h5ad."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import DATA_PROC, RESULTS
from fig_style import DOUBLE_COL, panel_header, save_fig, setup_nature

setup_nature()

CACHE = DATA_PROC / "zebrahub_t1_umap.h5ad"
SRC_REF = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"

PANELS = [
    ("A", "PCA-50", DATA_PROC / "zebrahub_t1_baselines.h5ad", "X_pca"),
    ("B", "Harmony", DATA_PROC / "zebrahub_t1_baselines.h5ad", "X_harmony"),
    ("C", "scVI", DATA_PROC / "zebrahub_t1_baselines.h5ad", "X_scvi"),
    ("D", "STATE SE-600M", DATA_PROC / "zebrahub_t1_joint_state.h5ad", "X_state"),
    ("E", "UCE", DATA_PROC / "zebrahub_t1_joint_uce.h5ad", "X_uce"),
    ("F", "Geneformer V1", DATA_PROC / "zebrahub_t1_joint_gf.h5ad", "X_gf"),
    ("G", "TranscriptFormer", DATA_PROC / "zebrahub_t1_joint_tf.h5ad", "X_tf"),
]

STAGE_ORDER = ["Train 10–16 hpf", "19–24 hpf", "2–3 dpf", "5–10 dpf"]
STAGE_COLORS = {"Train 10–16 hpf": "#9aa5b1", "19–24 hpf": "#e6a23c",
                "2–3 dpf": "#c94f4f", "5–10 dpf": "#3d7ea6"}


def stage_group(tp):
    if tp in ("10hpf", "12hpf", "14hpf", "16hpf"):
        return STAGE_ORDER[0]
    if tp in ("19hpf", "24hpf"):
        return STAGE_ORDER[1]
    if tp in ("2dpf", "3dpf"):
        return STAGE_ORDER[2]
    return STAGE_ORDER[3]


def umap_cache_uptodate(n_obs):
    if not CACHE.exists():
        return False
    cached = sc.read_h5ad(CACHE, backed="r")
    ok = (
        all(f"X_umap_{key}" in cached.obsm for _, _, _, key in PANELS)
        and cached.n_obs == n_obs
        and "stage" in cached.obs
    )
    cached.file.close()
    return ok


def build_umap_cache():
    ref = sc.read_h5ad(SRC_REF, backed="r")
    obs_names = pd.Index(ref.obs_names.astype(str))
    timepoints = None
    n_obs = ref.n_obs
    try:
        obs_cols = list(ref.obs.columns)
        tp_col = ref.obs["timepoint"]
        timepoints = np.asarray([s for s in tp_col.astype(str)])
    finally:
        ref.file.close()
    base = ad.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32))
    base.obs_names = obs_names
    base.obs["timepoint"] = timepoints

    for letter, label, path, key in PANELS:
        t0 = time.time()
        src = sc.read_h5ad(path)
        X = np.asarray(src.obsm[key], dtype=np.float32)
        assert X.shape[0] == n_obs, f"{label}: {X.shape[0]} != {n_obs}"
        tmp = sc.AnnData(X=X.copy())
        sc.pp.neighbors(tmp, n_neighbors=15, use_rep="X", random_state=42)
        sc.tl.umap(tmp, random_state=42)
        base.obsm[f"X_umap_{key}"] = np.asarray(tmp.obsm["X_umap"], dtype=np.float32)
        del tmp, src, X
        print(f"umap[{label}] ({time.time()-t0:.0f}s)", flush=True)

    base.obs["stage"] = [stage_group(t) for t in base.obs["timepoint"].astype(str)]
    base.write_h5ad(CACHE, compression="gzip")
    print("saved", CACHE)


def main():
    ref_probe = sc.read_h5ad(SRC_REF, backed="r")
    n_obs = ref_probe.n_obs
    ref_probe.file.close()
    if not umap_cache_uptodate(n_obs):
        build_umap_cache()
    else:
        print("valid umap cache ->", CACHE)

    cache = sc.read_h5ad(CACHE)
    stages = cache.obs["stage"].astype(str).values
    # paint the large grey training cloud first (bottom layer); later stages stay on top
    stage_rank = np.array([STAGE_ORDER.index(s) for s in stages])
    paint_order = np.argsort(stage_rank)

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 4.0))
    for ax, (letter, label, _path, key) in zip(axes.ravel(), PANELS):
        xy = np.asarray(cache.obsm[f"X_umap_{key}"], dtype=np.float32)
        for st in STAGE_ORDER:
            m = stages[paint_order] == st
            ax.scatter(xy[paint_order][m, 0], xy[paint_order][m, 1],
                       s=1.8, c=STAGE_COLORS[st],
                       alpha=0.85 if st != STAGE_ORDER[0] else 0.45,
                       linewidths=0, rasterized=True)
        tr_m = stages == STAGE_ORDER[0]
        late_m = stages == STAGE_ORDER[3]
        c_tr, c_late = xy[tr_m].mean(0), xy[late_m].mean(0)
        ax.annotate("", xy=c_late, xytext=c_tr,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
        mid = (c_tr + c_late) / 2 + 2.0
        ax.text(mid[0], mid[1], "train→late", fontsize=5.5, ha="center")
        panel_header(ax, letter.lower(), label)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    # 2x4 网格第 8 格留白
    axes.ravel()[-1].axis("off")

    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=4, label=s)
               for s, c in zip(STAGE_ORDER, STAGE_COLORS.values())]
    handles.append(Line2D([], [], color="black", marker="$\\rightarrow$", ls="",
                          markersize=7, label="train→late centroid"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, RESULTS, "fig3_umap_grid")


if __name__ == "__main__":
    main()
