# -*- coding: utf-8 -*-
"""11c 诊断：为什么 scVI surgery(30ep) 与 frozen 的 kNN 预测逐位相同？
检查：(1) 嵌入是否真的相同；(2) surgery 模型哪些参数可训练（scArches 默认冻结）；
      (3) 完全解冻的 surgery 是否改变结果。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

from common import DATA_PROC, RESULTS, SEED

scvi.settings.seed = SEED
torch.set_float32_matmul_precision("high")

joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
obs = joint.obs.copy()
obs["split"] = obs["split"].astype(str)
tr_m = (obs["split"] == "train").values
train = joint[tr_m].copy()
test = joint[~tr_m].copy()
print(f"train {train.shape} test {test.shape}")

scvi.model.SCVI.setup_anndata(train, batch_key="fish")
vae = scvi.model.SCVI(train, n_latent=30, n_layers=2, gene_likelihood="nb")
vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
Xtr = vae.get_latent_representation()

ref_fish = train.obs["fish"].astype(str).value_counts().index[0]
tf = test.copy()
tf.obs["fish"] = pd.Categorical([ref_fish] * tf.n_obs, categories=train.obs["fish"].astype("category").cat.categories)
Xte_frozen = vae.get_latent_representation(tf)

q = scvi.model.SCVI.load_query_data(test, vae)
n_trainable = sum(p.numel() for p in q.module.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in q.module.parameters())
print(f"surgery 模型可训练参数: {n_trainable:,} / {n_total:,} ({n_trainable/n_total:.2%})")
trainable_names = [n for n, p in q.module.named_parameters() if p.requires_grad]
print("可训练参数示例:", trainable_names[:8], "... 共", len(trainable_names))

q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
Xte_surg = q.get_latent_representation()

d = np.abs(Xte_frozen - Xte_surg)
print(f"frozen vs surgery: max|diff|={d.max():.4g} mean|diff|={d.mean():.4g}")
print(f"逐维相关: {np.corrcoef(Xte_frozen.ravel()[::37], Xte_surg.ravel()[::37])[0,1]:.6f}")

# 完全解冻版 surgery（scvi 1.4.2 的 load_query_data 只接受这几个 freeze 开关）
q2 = scvi.model.SCVI.load_query_data(test, vae, freeze_expression=False,
                                     freeze_decoder_first_layer=False,
                                     freeze_dropout=False)
q2.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
Xte_surg2 = q2.get_latent_representation()
d2 = np.abs(Xte_frozen - Xte_surg2)
print(f"frozen vs surgery(全解冻): max|diff|={d2.max():.4g} mean|diff|={d2.mean():.4g}")

# kNN 预测一致性
from sklearn.metrics import accuracy_score


def knn_pred(Xtr, ytr, Xte, k=15):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


ytr = obs.loc[tr_m, "lineage"].astype(str).values
yte = obs.loc[~tr_m, "lineage"].astype(str).values
for name, Xte in (("frozen", Xte_frozen), ("surgery默认", Xte_surg), ("surgery全解冻", Xte_surg2)):
    p = knn_pred(Xtr, ytr, Xte)
    print(f"{name}: lineage acc={accuracy_score(yte, p):.4f}")
    if name == "frozen":
        p0 = p
    else:
        print(f"  与 frozen 预测不一致的细胞比例: {np.mean(p != p0):.4%}")
print("DIAG DONE")
