# -*- coding: utf-8 -*-
"""42 UCE 斑马鱼统一协议评测:复用 11b 的 eval_embedding(frozen,kNN k=5/15/30+按鱼 bootstrap)。
输入:data/processed/zebrahub_t1_joint_uce.h5ad(obsm['X_uce'],与 joint 行同序)。
输出:results/t1_uce_eval.json(UCE_frozen 条目,并入主表/主图时与 t1_p2_eval 同构)。"""
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

spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding


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


def main():
    t0 = time.time()
    rep = {"protocol": "UCE frozen(4-layer CXG,human-ortholog 路由),与 t1_unified_eval.json 同协议"}

    path = DATA_PROC / "zebrahub_t1_joint_uce.h5ad"
    a = sc.read_h5ad(path)
    obs = a.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        if c in obs:
            obs[c] = obs[c].astype(str)
    tr_m = (obs["split"] == "train").values
    print(f"UCE: {a.shape}, obsm={list(a.obsm)}, train={tr_m.sum()}")
    X = np.asarray(a.obsm["X_uce"], dtype=np.float32)
    ctx = build_ctx(obs, tr_m)
    rep["UCE_frozen"] = eval_embedding(
        "UCE_frozen", "frozen", X[tr_m], X[~tr_m],
        obs.loc[tr_m, "lineage"].values, obs.loc[tr_m, "cluster"].values, ctx)

    rep["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS / "t1_uce_eval.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
