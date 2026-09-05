# -*- coding: utf-8 -*-
"""把 train/test 同源 h5ad 拼成 STATE 输入：CSR counts + var['gene_name']。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import anndata as ad
import scanpy as sc
from common import DATA_PROC

train = sc.read_h5ad(DATA_PROC / "zebrahub_t1_train_humanorth.h5ad")
test = sc.read_h5ad(DATA_PROC / "zebrahub_t1_test_humanorth.h5ad")
train.obs["split"] = "train"
test.obs["split"] = "test"
joint = ad.concat([train, test], join="inner", index_unique=None, label=None)
joint.var["gene_name"] = joint.var_names.astype(str)
# STATE 要 CSR
import scipy.sparse as sp
if not sp.isspmatrix_csr(joint.X):
    joint.X = sp.csr_matrix(joint.X)
out = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
joint.write_h5ad(out, compression="gzip")
print(joint.shape, "CSR", sp.isspmatrix_csr(joint.X), "->", out)
