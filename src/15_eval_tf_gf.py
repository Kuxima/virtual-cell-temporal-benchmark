# -*- coding: utf-8 -*-
"""15 P2 模型（TranscriptFormer / Geneformer）统一协议评测。
复用 11b 的 eval_embedding（frozen 协议，kNN k=5/15/30 + 按鱼 bootstrap）。
Geneformer：词表未覆盖的细胞（零 token）从 train/test 两侧剔除后评测。"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc

from common import DATA_PROC, RESULTS

SEED = 42

spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding


def build_ctx(obs, tr_m):
    """与 11b 相同的 ctx 构建（可在行子集上）。"""
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


def main():
    t0 = time.time()
    rep = {"protocol": "P2 frozen 前沿模型，与 t1_unified_eval.json 同协议"}

    # ---------- TranscriptFormer ----------
    tf_path = DATA_PROC / "zebrahub_t1_joint_tf.h5ad"
    if tf_path.exists():
        tf = sc.read_h5ad(tf_path)
        obs = tf.obs.copy()
        for c in ("timepoint", "fish", "lineage", "cluster", "split"):
            if c in obs:
                obs[c] = obs[c].astype(str)
        key = [k for k in ("X_tf", "X_transcriptformer", "X_emb") if k in tf.obsm]
        tr_m = (obs["split"] == "train").values
        print(f"TF: {tf.shape}, obsm={list(tf.obsm)}, key={key}, train={tr_m.sum()}")
        X = tf.obsm[key[0]] if key else tf.X
        ctx = build_ctx(obs, tr_m)
        rep["TranscriptFormer_frozen"] = eval_embedding(
            "TranscriptFormer_frozen", "frozen",
            np.asarray(X[tr_m], dtype=np.float32), np.asarray(X[~tr_m], dtype=np.float32),
            obs.loc[tr_m, "lineage"].values, obs.loc[tr_m, "cluster"].values, ctx)
    else:
        print("TF 嵌入不存在，跳过")

    # ---------- Geneformer ----------
    gf_path = DATA_PROC / "zebrahub_t1_joint_gf.h5ad"
    if gf_path.exists():
        gf = sc.read_h5ad(gf_path)
        obs = gf.obs.copy()
        for c in ("timepoint", "fish", "lineage", "cluster", "split"):
            obs[c] = obs[c].astype(str)
        covered = obs.get("gf_covered", pd.Series(True, index=obs.index)).astype(bool).values
        keep = covered & (np.abs(np.asarray(gf.obsm["X_gf"])).sum(1) > 0)
        print(f"GF: 覆盖 {keep.sum()}/{len(keep)}")
        obs_k = obs[keep]
        X = np.asarray(gf.obsm["X_gf"], dtype=np.float32)[keep]
        tr_m = (obs_k["split"] == "train").values
        ctx = build_ctx(obs_k, tr_m)
        rep["Geneformer_frozen"] = eval_embedding(
            "Geneformer_frozen", "frozen", X[tr_m], X[~tr_m],
            obs_k.loc[tr_m, "lineage"].values, obs_k.loc[tr_m, "cluster"].values, ctx)
    else:
        print("GF 嵌入不存在，跳过")

    rep["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS / "t1_p2_eval.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
