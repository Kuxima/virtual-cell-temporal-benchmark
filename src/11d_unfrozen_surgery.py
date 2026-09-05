# -*- coding: utf-8 -*-
"""11d 真正的测试时适应：scVI surgery 完全解冻版 ×3 种子完整评测。
背景（11c 诊断）：scvi-tools 1.4.2 默认 load_query_data surgery 的查询潜空间
输出与冻结编码器逐位相同（max|diff|=0）——默认"surgery"不是真适应；
全解冻（freeze_expression=False, freeze_decoder_first_layer=False, freeze_dropout=False）
后嵌入真实移动，lineage 71.6%→76.4%。本脚本把它作为正式的 adapted-scVI 行跑全指标。"""
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

from common import DATA_PROC, RESULTS, SEED

torch.set_float32_matmul_precision("high")
SEEDS = (42, 0, 1)

# 复用 11b 的评测函数（模块名以数字开头，需 spec 加载）
spec = importlib.util.spec_from_file_location(
    "u11b", os.path.join(os.path.dirname(__file__), "11b_hard_eval_unified.py"))
u11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11b)
eval_embedding = u11b.eval_embedding

t0 = time.time()
joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
obs = joint.obs.copy()
for c in ("timepoint", "fish", "lineage", "cluster", "split"):
    obs[c] = obs[c].astype(str)
tr_m = (obs["split"] == "train").values

ytr_lin = obs.loc[tr_m, "lineage"].values
yte_lin = obs.loc[~tr_m, "lineage"].values
ytr_clu = obs.loc[tr_m, "cluster"].values
yte_clu = obs.loc[~tr_m, "cluster"].values
lin_classes = sorted(set(ytr_lin))
clu_classes = sorted(set(ytr_clu))
lin_map = {c: i for i, c in enumerate(lin_classes)}
clu_map = {c: i for i, c in enumerate(clu_classes)}
lin_codes_tr = np.array([lin_map[c] for c in ytr_lin])
lin_codes_te = np.array([lin_map.get(c, -1) for c in yte_lin])
clu_codes_tr = np.array([clu_map[c] for c in ytr_clu])
clu_codes_te = np.array([clu_map.get(c, -1) for c in yte_clu])
shared = np.isin(yte_clu, clu_classes)
novel = ~shared
df_c2l = pd.DataFrame({"c": clu_codes_tr, "l": lin_codes_tr}).groupby("c")["l"].agg(lambda s: s.value_counts().index[0])
ctx = dict(yte_lin=lin_codes_te, yte_clu=clu_codes_te, shared=shared, novel=novel,
           lin_codes_tr=lin_codes_tr, lin_codes_te=lin_codes_te, clu_codes_tr=clu_codes_tr,
           lin_classes=lin_classes, clu_classes=clu_classes, clu2lin_code=df_c2l.to_dict(),
           tp_te=obs.loc[~tr_m, "timepoint"].values,
           fish_te_codes=pd.factorize(obs.loc[~tr_m, "fish"].values)[0])

train = joint[tr_m].copy()
test = joint[~tr_m].copy()
rep = {"protocol": "scVI surgery fully-unfrozen (freeze_expression=False, freeze_decoder_first_layer=False, freeze_dropout=False), 30ep query training, x3 seeds",
       "note": "默认 scArches surgery 与 frozen 逐位相同（见 11c 诊断），不单列", "seeds": list(SEEDS)}
per_seed = {}
for seed in SEEDS:
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(train, batch_key="fish")
    vae = scvi.model.SCVI(train, n_latent=30, n_layers=2, gene_likelihood="nb")
    vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
    Xtr_v = vae.get_latent_representation()
    q = scvi.model.SCVI.load_query_data(test, vae, freeze_expression=False,
                                        freeze_decoder_first_layer=False, freeze_dropout=False)
    q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
    Xte = q.get_latent_representation()
    print(f"seed {seed} done ({time.time()-t0:.0f}s)")
    per_seed[seed] = eval_embedding(f"scVI_surgery_unfrozen_s{seed}", "query-training-unfrozen",
                                    Xtr_v, Xte, ytr_lin, ytr_clu, ctx)
    del vae, q, Xte
    gc.collect()
    torch.cuda.empty_cache()

agg = {}
for key in ("lineage_all", "cluster_shared", "lineage_on_novel_clusters"):
    vals = [per_seed[s][key]["accuracy"] for s in SEEDS]
    agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": [float(v) for v in vals]}
rep["summary"] = agg
rep["s42"] = per_seed[42]
print("scVI surgery(unfrozen):", " | ".join(f"{k} {v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()))

rep["runtime_s"] = round(time.time() - t0, 1)
out = RESULTS / "t1_unfrozen_surgery.json"
out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
print("saved", out)
