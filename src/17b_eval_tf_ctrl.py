# -*- coding: utf-8 -*-
"""17b TF 基因集控制（2994 基因 = 人同源可回收集）统一协议评测。
对照主结果：TF 全基因集（3816）lineage 0.864 / cluster 0.040 / novel 0.798。"""
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

tf = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_tf_ctrl.h5ad")
obs = tf.obs.copy()
for c in ("timepoint", "fish", "lineage", "cluster", "split"):
    obs[c] = obs[c].astype(str)
tr_m = (obs["split"] == "train").values
X = np.asarray(tf.obsm["X_tf"], dtype=np.float32)
print(f"TF-ctrl: {tf.shape}, train {tr_m.sum()}")

ytr_lin = obs.loc[tr_m, "lineage"].values
yte_lin = obs.loc[~tr_m, "lineage"].values
ytr_clu = obs.loc[tr_m, "cluster"].values
yte_clu = obs.loc[~tr_m, "cluster"].values
lin_classes = sorted(set(ytr_lin))
clu_classes = sorted(set(ytr_clu))
lin_map = {c: i for i, c in enumerate(lin_classes)}
clu_map = {c: i for i, c in enumerate(clu_classes)}
clu2lin = pd.DataFrame({"c": [clu_map[c] for c in ytr_clu], "l": [lin_map[c] for c in ytr_lin]}
                       ).groupby("c")["l"].agg(lambda s: s.value_counts().index[0]).to_dict()
ctx = dict(
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
t0 = time.time()
out = eval_embedding("TF_ctrl_2994genes", "frozen", X[tr_m], X[~tr_m], ytr_lin, ytr_clu, ctx)
rep = {"protocol": "TF-metazoa on 2994 ortholog-recoverable genes (vs 3816 full)",
       "TF_ctrl": out, "runtime_s": round(time.time() - t0, 1)}
(RESULTS / "t1_tf_ctrl_eval.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
print("saved results/t1_tf_ctrl_eval.json")
