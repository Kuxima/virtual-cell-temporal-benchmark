# -*- coding: utf-8 -*-
"""33 Module-score validation of the novel-cluster biology.

AddModuleScore-style signature scores (expression-binned control sets,
100 controls, seed 42) on the 4,000-HVG log-normalized matrix, for five
late-differentiation programs cited in Results 3.4. For each test cluster we
report the mean score and a Mann-Whitney test against all other test cells
(BH-corrected within each program).

Memory-safe: all operations stay on the CSR counts matrix; no densification.
Output: results/novel_cluster_modulescore.csv"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from common import DATA_PROC, RESULTS

RNG = np.random.default_rng(42)
N_CONTROLS = 100
N_BINS = 10

PROGRAMS = {
    "neutrophil": ["mpx", "cebpa", "cotl1", "spi1b", "cxcl8a", "csf3r"],
    "choroid_plexus": ["rbp4", "apoeb", "fabp7a"],
    "intestinal_epithelium": ["fabp2", "krt18"],
    "endothelium": ["cdh5", "tie1", "gata5", "flt1"],
    "glia_oligodendrocyte": ["sox10", "ugt8", "cldnb", "mbpa"],
}


def col_mean(X, idx):
    """Mean of selected columns per cell (CSR-safe)."""
    return np.asarray(X[:, idx].mean(axis=1)).ravel()


def main():
    t0 = time.time()
    tr = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
    te = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
    assert (tr.var_names == te.var_names).all(), "HVG order mismatch"

    a = sc.concat([tr, te], join="inner", index_unique=None, label=None)
    genes = pd.Index(a.var_names.astype(str))
    # sc.concat's default obs merge drops 'split' (values differ across objects):
    # rebuild obs explicitly from the two sources instead.
    # NB: the zebrafish train/test h5ads have no 'split' column (it is added
    # later, when the joint files are built), so derive it from file origin.
    tr_obs = tr.obs[["cluster", "timepoint"]].astype(str).copy()
    tr_obs["split"] = "train"
    te_obs = te.obs[["cluster", "timepoint"]].astype(str).copy()
    te_obs["split"] = "test"
    obs = pd.concat([tr_obs, te_obs])
    obs.index = a.obs_names.astype(str)
    a.obs = obs

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X.tocsr()

    train_clusters = set(obs.loc[obs["split"] == "train", "cluster"])
    obs["is_novel"] = [c not in train_clusters for c in obs["cluster"]]
    test_idx = np.where((obs["split"] == "test").to_numpy())[0]

    gene_mean = np.asarray(X.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(gene_mean), N_BINS, labels=False, duplicates="drop").to_numpy()
    bin_members = [np.where(bins == b)[0] for b in range(N_BINS)]

    results = []
    for prog, gene_list in PROGRAMS.items():
        present = [g for g in gene_list if g in set(genes)]
        if len(present) < 2:
            print(f"[{prog}] skipped, <2 genes present ({present})", flush=True)
            continue
        s_idx = np.array([genes.get_loc(g) for g in present])

        per_bin_sizes = {b: int((bins[s_idx] == b).sum()) for b in range(N_BINS)}
        control_sets = []
        for _ in range(N_CONTROLS):
            parts = [RNG.choice(bin_members[b], size=n, replace=False)
                     for b, n in per_bin_sizes.items() if n]
            control_sets.append(np.concatenate(parts))

        sig = col_mean(X, s_idx)
        ctrl = np.stack([col_mean(X, c) for c in control_sets], axis=1)
        score = sig - ctrl.mean(axis=1)
        del ctrl

        test_clu = obs["cluster"].to_numpy()[test_idx]
        sc_test = score[test_idx]
        nov_test = obs["is_novel"].to_numpy()[test_idx]

        rows = []
        for clu in sorted(set(test_clu)):
            m = test_clu == clu
            rest = ~m
            u, pval = stats.mannwhitneyu(sc_test[m], sc_test[rest],
                                         alternative="two-sided")
            rows.append({
                "program": prog, "cluster": clu,
                "is_novel_cluster": bool(nov_test[m][0]) if m.any() else False,
                "n_cells": int(m.sum()),
                "genes_used": ",".join(present),
                "mean_score": float(sc_test[m].mean()),
                "mean_score_rest": float(sc_test[rest].mean()),
                "p_mannwhitney": float(pval),
            })
        df = pd.DataFrame(rows)
        df["p_bh"] = stats.false_discovery_control(df["p_mannwhitney"].to_numpy())
        results.append(df)

        nov = df[df["is_novel_cluster"]]
        best = nov.loc[nov["p_bh"].idxmin()] if len(nov) else None
        print(f"[{prog}] genes={present} | best novel cluster: "
              f"{'' if best is None else f'{best.cluster} (n={best.n_cells}, mean={best.mean_score:.3f}, p_bh={best.p_bh:.1e})'}",
              flush=True)

    out = pd.concat(results, ignore_index=True)
    dst = RESULTS / "novel_cluster_modulescore.csv"
    out.to_csv(dst, index=False)
    print("saved", dst, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
