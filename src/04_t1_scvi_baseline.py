# -*- coding: utf-8 -*-
"""04 T1 基线评测：scVI/scANVI 时序外推。
协议（无泄漏）：
  1) scVI 在训练时间点（10-16hpf）上训练，batch=fish
  2) scANVI 用 lineage 标签微调
  3) SCANVI.load_query_data 将测试时间点（19hpf-10dpf）映射进参考潜空间
  4) query.predict() 输出细胞类型预测 → 与真值比对（accuracy / macro-F1）
模型与流程对齐 BioMNI create_scvi_embeddings_scRNA + popV 的 SCANVI 路线。"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
import scvi
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from common import DATA_PROC, RESULTS, SEED

scvi.settings.seed = SEED
LABEL = "lineage"  # 主指标：10 个胚层/解剖大类；cluster 为细粒度补充
torch.set_float32_matmul_precision("high")

train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train.h5ad")
test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test.h5ad")
print(f"train {train.shape}, test {test.shape}")

# 测试集中的"新类别"统计（外推难度核心指标）
novel = set(test.obs[LABEL].astype(str)) - set(train.obs[LABEL].astype(str))
print(f"测试集 lineage 类别: {test.obs[LABEL].nunique()}, 其中参考集未见: {len(novel)} -> {sorted(novel)}")

t0 = time.time()

# 1) scVI 参考模型
scvi.model.SCVI.setup_anndata(train, batch_key="fish")
vae = scvi.model.SCVI(train, n_latent=30, n_layers=2, gene_likelihood="nb")
vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
print(f"[1] scVI 训练完成 {time.time()-t0:.0f}s")

# 2) scANVI 标签微调
lvae = scvi.model.SCANVI.from_scvi_model(vae, labels_key=LABEL, unlabeled_category="Unknown")
lvae.train(max_epochs=40, early_stopping=True, check_val_every_n_epoch=5)
print(f"[2] scANVI 训练完成 {time.time()-t0:.0f}s")

# 3) 查询映射（surgery）
query = scvi.model.SCANVI.load_query_data(test, lvae)
query.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
print(f"[3] query 映射完成 {time.time()-t0:.0f}s")

# 4) 预测与指标
pred = query.predict()
test.obs["pred_lineage"] = pred.values if hasattr(pred, "values") else pred
y_true = test.obs[LABEL].astype(str).values
y_pred = test.obs["pred_lineage"].astype(str).values

metrics = {
    "model": "scVI/scANVI (temporal extrapolation)",
    "train_timepoints": "10-16hpf",
    "test_timepoints": "19hpf-10dpf",
    "n_train": int(train.n_obs),
    "n_test": int(test.n_obs),
    "label": LABEL,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true) & set(y_pred)) or None)),
    "novel_types": sorted(novel),
    "runtime_s": round(time.time() - t0, 1),
}

# 分时间点的准确率（看外推如何随时间衰减——论文核心图）
per_tp = {}
tp = test.obs["timepoint"].astype(str)
for t in sorted(tp.unique()):
    m = (tp == t).values
    per_tp[t] = float(accuracy_score(y_true[m], y_pred[m]))
metrics["accuracy_per_timepoint"] = per_tp

out = RESULTS / "t1_scvi_baseline.json"
out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# 保存带预测的测试对象供画图
test.write_h5ad(DATA_PROC / "zebrahub_t1_test_annotated.h5ad", compression="gzip")
print("T1 BASELINE DONE")
