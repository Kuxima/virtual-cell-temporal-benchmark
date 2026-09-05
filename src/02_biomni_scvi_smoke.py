# -*- coding: utf-8 -*-
"""BioMNI 工具链冒烟测试：合成 3000 细胞数据 → create_scvi_embeddings_scRNA。
验证 BioMNI 工具层 + scvi-tools + GPU 全链路可用。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from common import DATA_PROC, SEED


def make_synthetic(n_cells=3000, n_genes=800, n_batches=3, n_types=4):
    rng = np.random.default_rng(SEED)
    # 每个细胞类型有 30 个marker基因，制造可分离结构
    markers = {t: rng.choice(n_genes, 30, replace=False) for t in range(n_types)}
    counts = rng.poisson(0.05, (n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, n_types, n_cells)
    batches = rng.integers(0, n_batches, n_cells)
    for i, t in enumerate(labels):
        counts[i, markers[t]] += rng.poisson(4.0, 30)
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(
            {
                "batch": [f"b{b}" for b in batches],
                "cell_type": [f"type{t}" for t in labels],
            },
            index=[f"cell{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]),
    )
    return adata


adata = make_synthetic()
adata.var_names_make_unique()
adata.write_h5ad(DATA_PROC / "synthetic_smoke.h5ad")
print("合成数据:", adata.shape)

from biomni.tool.genomics import create_scvi_embeddings_scRNA

result = create_scvi_embeddings_scRNA(
    adata_filename="synthetic_smoke.h5ad",
    batch_key="batch",
    label_key="cell_type",
    data_dir=str(DATA_PROC),
)
print(result)

out = sc.read_h5ad(DATA_PROC / "scvi_emb_data.h5ad")
assert "X_scVI" in out.obsm and "X_scANVI" in out.obsm, "嵌入缺失"
print(f"输出嵌入: X_scVI {out.obsm['X_scVI'].shape}, X_scANVI {out.obsm['X_scANVI'].shape}")
print("BIOMNI TOOLCHAIN SMOKE TEST PASSED")
