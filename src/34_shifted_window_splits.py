# -*- coding: utf-8 -*-
"""34 Shifted-window temporal splits（滑窗外推难度曲线 + 内插对照）.

问题：主协议（train ≤16hpf → test ≥19hpf）只给出一个外推距离上的单点。
本脚本把切分点沿时间轴平移，得到"外推难度 vs 训练窗覆盖"曲线：
  E16: train {10,12,14,16}          → test {19hpf..10dpf}   （= 主协议，锚点）
  E19: train {10..19}               → test {24hpf..10dpf}
  E24: train {10hpf..24hpf}         → test {2dpf..10dpf}
  E72: train {10hpf..3dpf}          → test {5dpf, 10dpf}
  I19: train {10,12,14,16,24}       → test {19hpf}          （内插对照）

I19 与 E16 的测试细胞完全相同（19hpf 全部 5,899 个细胞），唯一区别是
训练集是否包含更晚的 24hpf。若 I19 ≫ E16@19hpf，说明失败模式是时间外推
本身而非"晚期阶段天然更难"。

模型与协议：
  STATE / TranscriptFormer / Geneformer — 冻结嵌入直接换掩码评测（零重训）；
  PCA-50 / Harmony — 按窗重训（CPU 分钟级）；
  scVI — 按窗重训（frozen 协议，seed 42；主协议 3 种子 std ≤0.002）。

输出 results/shifted_window_eval.json（逐配置逐模型 lineage_all /
cluster_shared / majority 基线 / shared cluster 数）。
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

HPF = {"10hpf": 10.0, "12hpf": 12.0, "14hpf": 14.0, "16hpf": 16.0, "19hpf": 19.0,
       "24hpf": 24.0, "2dpf": 48.0, "3dpf": 72.0, "5dpf": 120.0, "10dpf": 240.0}

CONFIGS = [
    ("E16", ["10hpf", "12hpf", "14hpf", "16hpf"], ["19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]),
    ("E19", ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf"], ["24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]),
    ("E24", ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf"], ["2dpf", "3dpf", "5dpf", "10dpf"]),
    ("E72", ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf"], ["5dpf", "10dpf"]),
    ("I19", ["10hpf", "12hpf", "14hpf", "16hpf", "24hpf"], ["19hpf"]),
]

OUT = RESULTS / "shifted_window_eval.json"


def build_ctx(obs, tr_m):
    """任意掩码上的 ctx（与 15_eval_tf_gf.build_ctx 相同逻辑）。"""
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


def acc_lin_clu(Xtr, Xte, ytr_lin, ytr_clu, ctx):
    r = eval_embedding("win", "frozen", Xtr, Xte, ytr_lin, ytr_clu, ctx)
    return r


def main():
    t0 = time.time()
    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    obs = joint.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    print(f"joint {joint.shape} ({time.time()-t0:.0f}s)")

    # 冻结模型嵌入（行序与 joint_humanorth 对齐）
    st = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_state.h5ad")
    assert (st.obs_names == joint.obs_names).all(), "STATE 行序不一致"
    X_state = np.asarray(st.obsm["X_state"], dtype=np.float32)
    del st
    tfj = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_tf.h5ad")
    assert (tfj.obs_names == joint.obs_names).all(), "TF 行序不一致"
    X_tf = np.asarray(tfj.obsm["X_tf"], dtype=np.float32)
    del tfj
    gfj = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_gf.h5ad")
    assert (gfj.obs_names == joint.obs_names).all(), "GF 行序不一致"
    gf_keep = (gfj.obs["gf_covered"].astype(bool).values
               & (np.abs(np.asarray(gfj.obsm["X_gf"])).sum(1) > 0))
    X_gf = np.asarray(gfj.obsm["X_gf"], dtype=np.float32)[gf_keep]
    obs_gf = obs[gf_keep].reset_index(drop=True)
    del gfj
    gc.collect()
    print(f"frozen embeddings loaded; GF covered {gf_keep.sum()}")

    frozen_zf = [("STATE SE-600M", X_state), ("TranscriptFormer", X_tf)]
    frozen_gf = ("Geneformer V1", X_gf)

    rep = {
        "protocol": ("shifted-window temporal splits on the unified 2581-ortholog benchmark; "
                     "kNN (cosine, distance-weighted, k=15); PCA/Harmony/scVI retrained per "
                     "window (scVI frozen protocol, seed 42); I19 = interpolation control "
                     "(train includes 24hpf, test = same 19hpf cells as E16)"),
        "configs": {},
        "runtime_s": None,
    }
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            if isinstance(prev.get("configs"), dict):
                rep["configs"] = prev["configs"]
                print(f"resuming: {len(rep['configs'])} configs already done "
                      f"({', '.join(rep['configs'])})")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    for name, train_tps, test_tps in CONFIGS:
        if name in rep["configs"] and rep["configs"][name].get("models"):
            print(f"skip {name} (already evaluated)")
            continue
        tc = time.time()
        tr_m = obs["timepoint"].isin(train_tps).values
        te_m = obs["timepoint"].isin(test_tps).values
        assert tr_m.sum() > 500 and te_m.sum() > 500, (name, tr_m.sum(), te_m.sum())
        # 显式先 train 后 test 拼接，不依赖文件内的原始行序
        idx = np.concatenate([np.where(tr_m)[0], np.where(te_m)[0]])
        sub = obs.iloc[idx].reset_index(drop=True)
        sub_tr = np.arange(len(sub)) < int(tr_m.sum())
        sub_tr = pd.Series(sub_tr, index=sub.index)
        ctx = build_ctx(sub, sub_tr)
        ytr_lin = sub.loc[sub_tr, "lineage"].values
        ytr_clu = sub.loc[sub_tr, "cluster"].values
        n_shared_clu = len(ctx["clu_classes"])
        n_test_clu = int(len(set(sub.loc[~sub_tr, "cluster"])))
        # lineage majority 用 test 侧分布（与 11b majority_lineage_all 同口径）
        maj_lin = float(pd.Series(sub.loc[~sub_tr, "lineage"].values).value_counts(normalize=True).iloc[0])
        te_clu = sub.loc[~sub_tr, "cluster"].values
        top_tr_clu = pd.Series(ytr_clu).value_counts().index[0]
        maj_clu = float((te_clu[np.isin(te_clu, ctx["clu_classes"])] == top_tr_clu).mean())

        entry = {
            "train_tps": train_tps, "test_tps": test_tps,
            "train_cutoff_hpf": max(HPF[t] for t in train_tps),
            "test_min_hpf": min(HPF[t] for t in test_tps),
            "interpolation_control": name == "I19",
            "n_train": int(sub_tr.sum()), "n_test": int((~sub_tr).sum()),
            "n_train_clusters": n_shared_clu, "n_test_clusters": n_test_clu,
            "n_shared_clusters": len(set(te_clu) & set(ctx["clu_classes"])),
            "majority_lineage": maj_lin, "majority_cluster_shared": maj_clu,
            "models": {},
        }
        print(f"\n=== {name}: train {entry['n_train']} ({train_tps}) -> test {entry['n_test']} "
              f"({test_tps}); shared clusters {entry['n_shared_clusters']}/{n_test_clu}")

        # ---- 冻结模型（斑马鱼原生两个 + GF 用 covered 子集） ----
        for mname, X in frozen_zf:
            r = acc_lin_clu(X[tr_m], X[te_m], ytr_lin, ytr_clu, ctx)
            entry["models"][mname] = r
            del r
        subg = obs_gf
        subg_tr = subg["timepoint"].isin(train_tps).values
        ctxg = build_ctx(subg, subg_tr)
        r = acc_lin_clu(X_gf[subg_tr], X_gf[~subg_tr],
                        subg.loc[subg_tr, "lineage"].values,
                        subg.loc[subg_tr, "cluster"].values, ctxg)
        entry["models"]["Geneformer V1"] = r
        del r, ctxg

        # ---- PCA-50（按窗重训） ----
        idx = np.where(tr_m | te_m)[0]
        a = joint[idx].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        n_tr_w = int(sub_tr.sum())
        pca = PCA(n_components=50, random_state=SEED)
        Xa = to_dense(a.X).astype(np.float32)
        del a
        gc.collect()
        Xp_tr = pca.fit_transform(Xa[:n_tr_w])
        Xp_te = pca.transform(Xa[n_tr_w:])
        r = acc_lin_clu(Xp_tr, Xp_te, ytr_lin, ytr_clu, ctx)
        entry["models"]["PCA-50"] = r
        del r

        # ---- Harmony（按窗重训，transductive） ----
        H = harmonize(np.vstack([Xp_tr, Xp_te]), sub, batch_key="fish")
        r = acc_lin_clu(H[:n_tr_w], H[n_tr_w:], ytr_lin, ytr_clu, ctx)
        entry["models"]["Harmony"] = r
        del H, Xp_tr, Xp_te, Xa, pca
        gc.collect()

        # ---- scVI frozen（按窗重训，seed 42） ----
        tr_counts = joint[tr_m].copy()
        te_counts = joint[te_m].copy()
        scvi.settings.seed = SEED
        scvi.model.SCVI.setup_anndata(tr_counts, batch_key="fish")
        vae = scvi.model.SCVI(tr_counts, n_latent=30, n_layers=2, gene_likelihood="nb")
        vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
        Xtr_v = vae.get_latent_representation()
        ref_fish = tr_counts.obs["fish"].astype(str).value_counts().index[0]
        tfwd = te_counts.copy()
        tfwd.obs["fish"] = pd.Categorical(
            [ref_fish] * tfwd.n_obs,
            categories=tr_counts.obs["fish"].astype("category").cat.categories)
        Xte_v = vae.get_latent_representation(tfwd)
        del tfwd, vae, tr_counts, te_counts
        gc.collect()
        torch.cuda.empty_cache()
        r = acc_lin_clu(Xtr_v, Xte_v, ytr_lin, ytr_clu, ctx)
        entry["models"]["scVI"] = r
        del r, Xtr_v, Xte_v
        gc.collect()

        rep["configs"][name] = entry
        OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"--- {name} done ({time.time()-tc:.0f}s, total {time.time()-t0:.0f}s); saved {OUT}")

    rep["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nALL DONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
