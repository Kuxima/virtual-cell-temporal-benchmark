# -*- coding: utf-8 -*-
"""21 P4 小鼠统一协议评测：PCA/Harmony/scVI(×3 种子)（复用 11b 的 eval_embedding）。
与斑马鱼协议同构：无监督嵌入 + kNN(k=15 cosine) + 按样本 bootstrap。"""
import gc
import importlib.util
import json
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
from scipy import sparse
from sklearn.decomposition import PCA

from common import DATA_PROC, RESULTS, SEED, shared_novel_mask

torch.set_float32_matmul_precision("high")
SEEDS = (42, 0, 1)

spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding
to_dense = u11b.to_dense


def lognorm(a):
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def build_ctx(obs, tr_m):
    ytr_lin = obs.loc[tr_m, "lineage"].values
    yte_lin = obs.loc[~tr_m, "lineage"].values
    ytr_clu = obs.loc[tr_m, "cluster"].values
    yte_clu = obs.loc[~tr_m, "cluster"].values
    lin_classes = sorted(set(ytr_lin))
    clu_classes = sorted(set(ytr_clu))
    lin_map = {c: i for i, c in enumerate(lin_classes)}
    clu_map = {c: i for i, c in enumerate(clu_classes)}
    clu_codes_te = np.array([clu_map.get(c, -1) for c in yte_clu], dtype=np.int32)
    shared, novel = shared_novel_mask(ytr_clu, yte_clu)
    assert np.all(clu_codes_te[novel] == -1)
    assert np.all(clu_codes_te[shared] >= 0)
    assert int(shared.sum() + novel.sum()) == len(yte_clu)
    clu2lin = pd.DataFrame({"c": [clu_map[c] for c in ytr_clu], "l": [lin_map[c] for c in ytr_lin]}
                           ).groupby("c")["l"].agg(lambda s: s.value_counts().index[0]).to_dict()
    return dict(
        yte_lin=np.array([lin_map[c] for c in yte_lin]),
        yte_clu=clu_codes_te,
        shared=shared, novel=novel,
        lin_codes_tr=np.array([lin_map[c] for c in ytr_lin]),
        lin_codes_te=np.array([lin_map[c] for c in yte_lin]),
        clu_codes_tr=np.array([clu_map[c] for c in ytr_clu]),
        lin_classes=lin_classes, clu_classes=clu_classes, clu2lin_code=clu2lin,
        tp_te=obs.loc[~tr_m, "stage"].values,
        fish_te_codes=pd.factorize(obs.loc[~tr_m, "sample"].values)[0],
    ), (ytr_lin, ytr_clu)


def main():
    t0 = time.time()
    joint = sc.read_h5ad(DATA_PROC / "mouse_t1_joint.h5ad")
    obs = joint.obs.copy()
    for c in ("stage", "sample", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    tr_m = (obs["split"] == "train").values
    ctx, (ytr_lin, ytr_clu) = build_ctx(obs, tr_m)
    assert int(tr_m.sum()) == 20308
    assert int((~tr_m).sum()) == 88549
    assert len(ctx["clu_classes"]) == 11
    assert len(set(obs.loc[~tr_m, "cluster"])) == 22
    assert int(ctx["shared"].sum()) == 61784
    assert int(ctx["novel"].sum()) == 26765
    assert np.array_equal(ctx["novel"], ~ctx["shared"])
    rep = {
        "dataset": "E-MTAB-6967 mouse gastrulation, train E6.5-E7.0, test E7.25-E8.5",
        "majority_lineage": float(pd.Series(ctx["yte_lin"]).map(pd.Series(ctx["yte_lin"]).value_counts(normalize=True)).max()),
        "n_train_cluster": len(ctx["clu_classes"]),
        "n_test_cluster": int(len(set(obs.loc[~tr_m, "cluster"]))),
        "n_novel_cluster": int(len(set(obs.loc[~tr_m, "cluster"]) - set(ctx["clu_classes"]))),
    }
    # 正确的 majority 基线
    yte_clu_str = obs.loc[~tr_m, "cluster"].values
    maj_lineage = obs.loc[~tr_m, "lineage"].value_counts(normalize=True).iloc[0]
    shared_mask = ctx["shared"]
    maj_cluster = (yte_clu_str[shared_mask] == pd.Series(ytr_clu if isinstance(ytr_clu, np.ndarray) else ytr_clu).value_counts().index[0]).mean()
    rep["majority_lineage"] = float(maj_lineage)
    rep["majority_cluster_shared"] = float(maj_cluster)
    print(f"majority lineage {maj_lineage:.4f}, cluster(shared) {maj_cluster:.4f}")

    # PCA frozen
    trn, ten = lognorm(joint[tr_m].copy()), lognorm(joint[~tr_m].copy())
    pca = PCA(n_components=50, random_state=SEED)
    Xtr = pca.fit_transform(to_dense(trn.X))
    Xte = pca.transform(to_dense(ten.X))
    rep["PCA_frozen"] = eval_embedding("PCA_frozen", "frozen", Xtr, Xte, ytr_lin, ytr_clu, ctx)
    del Xtr, Xte
    gc.collect()

    # Harmony（transductive，batch=sample）
    Xpca = np.vstack([pca.transform(to_dense(trn.X)), pca.transform(to_dense(ten.X))])
    jo = sc.concat([trn, ten], label="split", keys=["train", "test"])
    H = harmonize(Xpca, jo.obs, batch_key="sample")
    n_tr = int(tr_m.sum())
    rep["Harmony_adapted"] = eval_embedding("Harmony_adapted", "transductive", H[:n_tr], H[n_tr:], ytr_lin, ytr_clu, ctx)
    del Xpca, H, jo
    gc.collect()

    # scVI ×3 种子
    train_c, test_c = joint[tr_m].copy(), joint[~tr_m].copy()
    per_seed = {}
    for seed in SEEDS:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(train_c, batch_key="sample")
        vae = scvi.model.SCVI(train_c, n_latent=30, n_layers=2, gene_likelihood="nb")
        vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
        Xtr_v = vae.get_latent_representation()
        q = scvi.model.SCVI.load_query_data(test_c, vae, freeze_expression=False,
                                            freeze_decoder_first_layer=False, freeze_dropout=False)
        q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
        Xte_v = q.get_latent_representation()
        print(f"scVI seed {seed} done ({time.time()-t0:.0f}s)")
        per_seed[seed] = eval_embedding(f"scVI_s{seed}", "query-training-unfrozen",
                                        Xtr_v, Xte_v, ytr_lin, ytr_clu, ctx)
        del vae, q, Xte_v
        gc.collect()
        torch.cuda.empty_cache()
    agg = {}
    for key in ("lineage_all", "cluster_shared", "lineage_on_novel_clusters"):
        vals = [per_seed[s][key]["accuracy"] for s in SEEDS]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    rep["scVI_summary"] = agg
    rep["scVI_s42"] = per_seed[42]
    print("scVI:", " | ".join(f"{k} {v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()))

    rep["runtime_s"] = round(time.time() - t0, 1)
    (RESULTS / "p4_classical_eval.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved results/p4_classical_eval.json")


if __name__ == "__main__":
    main()
