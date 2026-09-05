# -*- coding: utf-8 -*-
"""28 Cache the three baseline embeddings used by every figure-level comparison.

Evaluation scripts 05/11b computed PCA / Harmony / scVI transiently; only the
foundation-model embeddings live in obsm. This script reproduces the exact
reference protocol once (seed 42, 60-epoch scVI + fully-unfrozen query surgery,
matching 21_p4-style adaptation on the zebrafish) and persists:

    data/processed/zebrahub_t1_baselines.h5ad
        X_pca      PCA-50   (log-normalized counts, fitted on train only)
        X_harmony  Harmony  (transductive, batch = fish)
        X_scvi     scVI     (query-training-unfrozen, seed 42)

Re-running with an existing valid cache is a no-op; set FORCE_BASELINE_CACHE=1
to overwrite. Embedding row order matches zebrahub_t1_joint_humanorth.h5ad."""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from harmony import harmonize
from sklearn.decomposition import PCA

from common import DATA_PROC, SEED

OUT = DATA_PROC / "zebrahub_t1_baselines.h5ad"
SRC = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
torch.set_float32_matmul_precision("high")


def valid_cache(n_obs):
    if not OUT.exists() or os.environ.get("FORCE_BASELINE_CACHE", "0") == "1":
        return False
    try:
        a = sc.read_h5ad(OUT, backed="r")
        ok = (
            all(k in a.obsm for k in ("X_pca", "X_harmony", "X_scvi"))
            and a.n_obs == n_obs
        )
        a.file.close()
        return ok
    except Exception:
        return False


def main():
    t0 = time.time()
    joint = sc.read_h5ad(SRC)
    obs = joint.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    tr_m = (obs["split"] == "train").values
    n = joint.n_obs
    if valid_cache(n):
        src = sc.read_h5ad(SRC, backed="r")
        cached = sc.read_h5ad(OUT, backed="r")
        same = np.array_equal(cached.obs_names.astype(str), src.obs_names.astype(str))
        src.file.close(); cached.file.close()
        if same:
            print("valid cache exists ->", OUT)
            return
    print(f"joint {joint.shape}, train {tr_m.sum()}")

    trn = joint[tr_m].copy()
    ten = joint[~tr_m].copy()
    for a in (trn, ten):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)

    # PCA-50 (frozen)
    pca = PCA(n_components=50, random_state=SEED)
    Xp_tr = pca.fit_transform(trn.X.toarray() if hasattr(trn.X, "toarray") else trn.X)
    Xp_te = pca.transform(ten.X.toarray() if hasattr(ten.X, "toarray") else ten.X)

    # Harmony (transductive)
    Xp_all = np.vstack([Xp_tr, Xp_te])
    jo = sc.concat([trn, ten], label="split", keys=["train", "test"])
    H = harmonize(Xp_all, jo.obs, batch_key="fish")
    del Xp_all, jo; gc.collect()

    # scVI (train + fully-unfrozen query surgery, seed 42)
    train_counts = joint[tr_m].copy()
    test_counts = joint[~tr_m].copy()
    scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(train_counts, batch_key="fish")
    vae = scvi.model.SCVI(train_counts, n_latent=30, n_layers=2, gene_likelihood="nb")
    vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
    Xtr_v = vae.get_latent_representation()
    q = scvi.model.SCVI.load_query_data(test_counts, vae, freeze_expression=False,
                                        freeze_decoder_first_layer=False, freeze_dropout=False)
    q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
    Xte_v = q.get_latent_representation()

    out = ad_stack(joint.obs_names.astype(str),
                   {"X_pca": np.vstack([Xp_tr, Xp_te]).astype(np.float32),
                    "X_harmony": H.astype(np.float32),
                    "X_scvi": np.vstack([Xtr_v, Xte_v]).astype(np.float32)})
    out.write_h5ad(OUT, compression="gzip")
    print(f"saved {OUT} ({time.time()-t0:.0f}s)")


def ad_stack(obs_names, matrices):
    """Obs-only AnnData carrying several embedding matrices in obsm."""
    import anndata as ad
    a = ad.AnnData(X=np.zeros((len(obs_names), 1), dtype=np.float32))
    a.obs_names = pd.Index(obs_names)
    for k, m in matrices.items():
        a.obsm[k] = m
    return a


if __name__ == "__main__":
    main()
