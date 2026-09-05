# -*- coding: utf-8 -*-
"""45 UCE 补丁:把 UCE 加入移窗扫描(shifted_window_eval.json)与随机切分天花板(random_split_ceiling.json)。
冻结嵌入只重探针;随机切分逐位复现 35 的 default_rng(42) 排序索引。"""
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

SEED = 42


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
    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    obs = joint.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    uj = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_uce.h5ad")
    assert (uj.obs_names == joint.obs_names).all(), "UCE 行序不一致"
    X_uce = np.asarray(uj.obsm["X_uce"], dtype=np.float32)
    del uj, joint
    import gc
    gc.collect()
    print(f"UCE zf embedding loaded {X_uce.shape}")

    # ---------- 1) 移窗扫描 ----------
    OUT_W = RESULTS / "shifted_window_eval.json"
    rep = json.loads(OUT_W.read_text())
    # 与 34 相同的窗口定义
    tp = obs["timepoint"].values
    ALL = ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
    idx_of = {t: i for i, t in enumerate(ALL)}
    WINDOWS = {
        "E16": ([t for t in ALL[:4]], [t for t in ALL[4:]]),
        "E19": ([t for t in ALL[:5]], [t for t in ALL[5:]]),
        "E24": ([t for t in ALL[:6]], [t for t in ALL[6:]]),
        "E72": ([t for t in ALL[:8]], [t for t in ALL[8:]]),
        "I19": ([t for t in ALL[:4]] + ["24hpf"], ["19hpf"]),
    }
    for name, (train_tps, test_tps) in WINDOWS.items():
        if name not in rep["configs"]:
            print(f"skip {name}(主 JSON 无此配置)")
            continue
        if "UCE" in rep["configs"][name]["models"]:
            print(f"skip {name}.UCE(已有)")
            continue
        tr_m = np.isin(tp, train_tps)
        te_m = np.isin(tp, test_tps)
        idx = np.concatenate([np.where(tr_m)[0], np.where(te_m)[0]])
        sub = obs.iloc[idx].reset_index(drop=True)
        sub_tr = pd.Series(np.arange(len(sub)) < int(tr_m.sum()), index=sub.index)
        ctx = build_ctx(sub, sub_tr)
        r = eval_embedding("UCE", "frozen", X_uce[tr_m], X_uce[te_m],
                           sub.loc[sub_tr, "lineage"].values, sub.loc[sub_tr, "cluster"].values, ctx)
        rep["configs"][name]["models"]["UCE"] = r
        lin = r["lineage_all"]["accuracy"]
        clu = r["cluster_shared"]["accuracy"]
        print(f"{name}: UCE lineage {lin:.3f} cluster {clu:.3f}")
        OUT_W.write_text(json.dumps(rep, indent=1))
    del rep

    # ---------- 2) 随机切分天花板(逐位复现 35 的 split) ----------
    OUT_R = RESULTS / "random_split_ceiling.json"
    rep = json.loads(OUT_R.read_text())
    if "UCE" not in rep["models"]:
        n = len(obs)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n)
        n_tr = int(round(rep["train_frac"] * n))
        tr_m = np.zeros(n, dtype=bool)
        tr_m[perm[:n_tr]] = True
        idx = np.concatenate([np.sort(np.where(tr_m)[0]), np.sort(np.where(~tr_m)[0])])
        sub = obs.iloc[idx].reset_index(drop=True)
        sub_tr = pd.Series(np.arange(len(sub)) < n_tr, index=sub.index)
        ctx = build_ctx(sub, sub_tr)
        r = eval_embedding("UCE (random)", "frozen", X_uce[tr_m], X_uce[~tr_m],
                           sub.loc[sub_tr, "lineage"].values, sub.loc[sub_tr, "cluster"].values, ctx)
        rep["models"]["UCE"] = r
        # comparison 块
        uce_time = json.loads((RESULTS / "t1_uce_eval.json").read_text())["UCE_frozen"]
        rep["comparison"]["UCE"] = {
            "lineage_time": uce_time["lineage_all"]["accuracy"],
            "lineage_random": r["lineage_all"]["accuracy"],
            "lineage_delta": r["lineage_all"]["accuracy"] - uce_time["lineage_all"]["accuracy"],
            "cluster_time": uce_time["cluster_shared"]["accuracy"],
            "cluster_random": r["cluster_shared"]["accuracy"],
            "cluster_delta": r["cluster_shared"]["accuracy"] - uce_time["cluster_shared"]["accuracy"],
        }
        print("random split UCE:", rep["comparison"]["UCE"])
        OUT_R.write_text(json.dumps(rep, indent=1))

    print("addendum done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
