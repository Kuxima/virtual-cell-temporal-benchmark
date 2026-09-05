# -*- coding: utf-8 -*-
"""35 Random cell-split ceiling control（随机切分天花板对照）.

动机：文献中"虚拟细胞/基础模型"评测普遍用随机细胞切分（80/20），同分布
kNN 近邻会大幅高估迁移能力。本脚本在同一批细胞、同一 kNN 协议下用
随机 80/20 重跑全部六个模型，与时间切分（train ≤16hpf → test ≥19hpf）
的官方数字逐模型对比，量化随机切分的乐观偏差。

  STATE / TranscriptFormer / Geneformer — 冻结嵌入换掩码（零重训）；
  PCA-50 / Harmony — 随机切分上重训；
  scVI — 随机切分上重训（frozen 协议，seed 42；随机切分下 train/test
  共享全部 fish 批次，无需批次映射）。

时间切分参考数字直接取自 results/t1_unified_eval.json 与
results/t1_p2_eval.json（论文主表数字），不在本脚本重复计算。

输出 results/random_split_ceiling.json。
"""
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
from sklearn.decomposition import PCA

from common import DATA_PROC, RESULTS, SEED

torch.set_float32_matmul_precision("high")

spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding
to_dense = u11b.to_dense

OUT = RESULTS / "random_split_ceiling.json"
TRAIN_FRAC = 0.8


def build_ctx(obs, tr_m):
    ytr_lin = obs.loc[tr_m, "lineage"].values
    yte_lin = obs.loc[~tr_m, "lineage"].values
    ytr_clu = obs.loc[tr_m, "cluster"].values
    yte_clu = obs.loc[~tr_m, "cluster"].values
    lin_classes = sorted(set(ytr_lin))
    clu_classes = sorted(set(ytr_clu))
    lin_map = {c: i for i, c in enumerate(lin_classes)}
    clu_map = {c: i for i, c in enumerate(clu_classes)}
    clu2lin = pd.DataFrame({"c": [clu_map[c] for c in ytr_clu],
                            "l": [lin_map[c] for c in ytr_lin]}
                           ).groupby("c")["l"].agg(lambda s: s.value_counts().index[0]).to_dict()
    return dict(
        yte_lin=np.array([lin_map[c] for c in yte_lin]),
        yte_clu=np.array([clu_map.get(c, -1) for c in yte_clu]),
        shared=np.isin(yte_clu, clu_classes), novel=~np.isin(yte_clu, clu_classes),
        lin_codes_tr=np.array([lin_map[c] for c in ytr_lin]),
        lin_codes_te=np.array([lin_map[c] for c in yte_lin]),
        clu_codes_tr=np.array([clu_map[c] for c in ytr_clu]),
        lin_classes=lin_classes, clu_classes=clu_classes, clu2lin_code=clu2lin,
        tp_te=obs.loc[~tr_m, "timepoint"].values,
        fish_te_codes=pd.factorize(obs.loc[~tr_m, "fish"].values)[0],
    )


def reference_time_split():
    uni = json.loads((RESULTS / "t1_unified_eval.json").read_text())
    p2 = json.loads((RESULTS / "t1_p2_eval.json").read_text())
    ref = {
        "PCA-50": uni["PCA_frozen"],
        "Harmony": uni["Harmony_adapted"],
        "scVI": uni["scVI_frozen_s42"],
        "STATE SE-600M": uni["STATE_frozen"],
        "TranscriptFormer": p2["TranscriptFormer_frozen"],
        "Geneformer V1": p2["Geneformer_frozen"],
    }
    return {m: {"lineage_all": r["lineage_all"], "cluster_shared": r["cluster_shared"]}
            for m, r in ref.items()}


def main():
    t0 = time.time()
    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    obs = joint.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    n = joint.n_obs
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr = int(round(TRAIN_FRAC * n))
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    tr_m = np.zeros(n, dtype=bool)
    tr_m[tr_idx] = True
    # 排序到文件行序：X[tr_m]/X[~tr_m] 均按 joint 顺序切片，sub 必须同序，
    # 否则几何与标签错位（症状：所有嵌入给出相同的"标签置换"准确率）。
    tr_idx = np.sort(tr_idx)
    te_idx = np.sort(te_idx)
    sub = obs.iloc[np.concatenate([tr_idx, te_idx])].reset_index(drop=True)
    sub_tr = pd.Series(np.arange(len(sub)) < n_tr, index=sub.index)
    ctx = build_ctx(sub, sub_tr)
    ytr_lin = sub.loc[sub_tr, "lineage"].values
    ytr_clu = sub.loc[sub_tr, "cluster"].values
    n_shared_clu = len(set(sub.loc[~sub_tr, "cluster"]) & set(ctx["clu_classes"]))
    print(f"random split: train {n_tr}, test {n - n_tr}, "
          f"clusters shared {n_shared_clu}/{sub.loc[~sub_tr, 'cluster'].nunique()}")

    rep = {
        "protocol": ("random 80/20 cell split (seed 42) on the unified 2581-ortholog "
                     "benchmark, same kNN protocol as the time split; time-split "
                     "reference numbers imported from t1_unified_eval.json / t1_p2_eval.json"),
        "train_frac": TRAIN_FRAC, "seed": SEED,
        "n_train": n_tr, "n_test": n - n_tr,
        "n_shared_clusters": n_shared_clu,
        "time_split_reference": reference_time_split(),
        "models": {},
    }
    rep["majority_lineage"] = float(pd.Series(ytr_lin).value_counts(normalize=True).iloc[0])
    te_clu = sub.loc[~sub_tr, "cluster"].values
    top_tr_clu = pd.Series(ytr_clu).value_counts().index[0]
    rep["majority_cluster_shared"] = float(
        (te_clu[np.isin(te_clu, ctx["clu_classes"])] == top_tr_clu).mean())
    print(f"majority lineage {rep['majority_lineage']:.4f} cluster {rep['majority_cluster_shared']:.4f}")

    # ---- 冻结模型 ----
    for mname, f, key in (("STATE SE-600M", "zebrahub_t1_joint_state.h5ad", "X_state"),
                          ("TranscriptFormer", "zebrahub_t1_joint_tf.h5ad", "X_tf")):
        j = sc.read_h5ad(DATA_PROC / f)
        assert (j.obs_names == joint.obs_names).all(), f"{mname} 行序不一致"
        X = np.asarray(j.obsm[key], dtype=np.float32)
        del j
        gc.collect()
        rep["models"][mname] = eval_embedding(f"{mname} (random)", "frozen",
                                              X[tr_m], X[~tr_m], ytr_lin, ytr_clu, ctx)
        del X
        gc.collect()

    gfj = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_gf.h5ad")
    assert (gfj.obs_names == joint.obs_names).all(), "GF 行序不一致"
    gf_keep = (gfj.obs["gf_covered"].astype(bool).values
               & (np.abs(np.asarray(gfj.obsm["X_gf"])).sum(1) > 0))
    X_gf = np.asarray(gfj.obsm["X_gf"], dtype=np.float32)[gf_keep]
    obs_gf = obs[gf_keep].reset_index(drop=True)
    del gfj
    gc.collect()
    gtr = gf_keep & tr_m
    gte = gf_keep & ~tr_m
    subg_tr = pd.Series(gtr[gf_keep], index=obs_gf.index)
    ctxg = build_ctx(obs_gf, subg_tr)
    rep["models"]["Geneformer V1"] = eval_embedding(
        "Geneformer V1 (random)", "frozen",
        X_gf[gtr], X_gf[gte],
        obs_gf.loc[subg_tr, "lineage"].values,
        obs_gf.loc[subg_tr, "cluster"].values, ctxg)
    del X_gf
    gc.collect()

    # ---- PCA + Harmony（随机切分重训） ----
    a = joint.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    Xa = to_dense(a.X).astype(np.float32)
    del a
    gc.collect()
    pca = PCA(n_components=50, random_state=SEED)
    Xp_tr = pca.fit_transform(Xa[tr_m])
    Xp_te = pca.transform(Xa[~tr_m])
    rep["models"]["PCA-50"] = eval_embedding("PCA-50 (random)", "frozen",
                                             Xp_tr, Xp_te, ytr_lin, ytr_clu, ctx)
    H = harmonize(np.vstack([Xp_tr, Xp_te]), sub, batch_key="fish")
    rep["models"]["Harmony"] = eval_embedding("Harmony (random)", "transductive",
                                              H[:n_tr], H[n_tr:], ytr_lin, ytr_clu, ctx)
    del H, Xp_tr, Xp_te, Xa, pca
    gc.collect()

    # ---- scVI frozen（随机切分重训；train/test 共享批次，直接前向） ----
    tr_counts = joint[tr_m].copy()
    te_counts = joint[~tr_m].copy()
    scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(tr_counts, batch_key="fish")
    vae = scvi.model.SCVI(tr_counts, n_latent=30, n_layers=2, gene_likelihood="nb")
    vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
    Xtr_v = vae.get_latent_representation()
    Xte_v = vae.get_latent_representation(te_counts)
    del vae, tr_counts, te_counts
    gc.collect()
    torch.cuda.empty_cache()
    rep["models"]["scVI"] = eval_embedding("scVI (random)", "frozen",
                                           Xtr_v, Xte_v, ytr_lin, ytr_clu, ctx)
    del Xtr_v, Xte_v
    gc.collect()

    # ---- 汇总偏差 ----
    rep["comparison"] = {}
    print("\nmodel            time-split lin/clu      random-split lin/clu     delta lin")
    for m, rref in rep["time_split_reference"].items():
        rr = rep["models"][m]
        rep["comparison"][m] = {
            "lineage_time": rref["lineage_all"]["accuracy"],
            "lineage_random": rr["lineage_all"]["accuracy"],
            "lineage_delta": rr["lineage_all"]["accuracy"] - rref["lineage_all"]["accuracy"],
            "cluster_time": rref["cluster_shared"]["accuracy"],
            "cluster_random": rr["cluster_shared"]["accuracy"],
            "cluster_delta": rr["cluster_shared"]["accuracy"] - rref["cluster_shared"]["accuracy"],
        }
        c = rep["comparison"][m]
        print(f"{m:<16} {c['lineage_time']:.3f}/{c['cluster_time']:.3f}"
              f"              {c['lineage_random']:.3f}/{c['cluster_random']:.3f}"
              f"          +{c['lineage_delta']:.3f}")

    rep["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
