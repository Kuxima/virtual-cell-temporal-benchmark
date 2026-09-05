# -*- coding: utf-8 -*-
"""17 TF 基因集公平性控制输入：把 TF 输入限制为「人同源映射可回收」的斑马鱼基因
（≈2994 个，即 STATE/Geneformer/scVI 实际看到的基因的斑马鱼侧），
验证 TF 优势不来自更大的基因集（3816）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
from common import DATA_PROC, RESULTS

tf_in = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_tfinput.h5ad")
ho = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
ortho = json.loads((RESULTS / "zf2human_orthologs.json").read_text())  # zf 符号 -> 人符号

human_set = set(ho.var_names.astype(str))
zf_kept = [zf for zf, h in ortho.items() if h in human_set]
zf_set = set(zf_kept) & set(tf_in.var_names.astype(str))
print(f"人同源基因 {len(human_set)}，对应斑马鱼基因 ∩ TF 输入词表命中集: {len(zf_set)}")

mask = np.asarray(tf_in.var_names.astype(str).isin(zf_set))
ctrl = tf_in[:, mask].copy()
ctrl.obs["assay"] = "unknown"
out = DATA_PROC / "zebrahub_t1_joint_tfinput_ctrl.h5ad"
ctrl.write_h5ad(out, compression="gzip")
print(f"控制输入 {ctrl.shape} -> {out}")
