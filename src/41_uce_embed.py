# -*- coding: utf-8 -*-
"""41 UCE 嵌入驱动(斑马鱼/小鼠通用)。

适配点(相对 external/UCE 官方 CLI):
- filter=False:官方默认会再过滤(min_cells=10 / min_genes=25),会破坏与统一评测的行对齐;
- np.random.seed(42):官方每细胞按表达量加权采样 1024 个基因,无全局种子,补上以保证可复现;
- 路径全部用正斜杠(官方按 "/" 切文件名,Windows 反斜杠会解析失败);
- 环境变量:UCE_SPECIES(zebra|mouse)、UCE_BATCH(默认 16,16GB 显存起步值)。

输出:<work_dir>/{name}_uce_adata.h5ad,obsm['X_uce'](1280 维),细胞与输入完全同序。
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

# 路径全部可移植：默认相对仓库布局解析（src/ 的上一级为项目根），可用环境变量覆盖；
# 仍统一转正斜杠（官方按 "/" 切文件名，Windows 反斜杠会解析失败）。
PROJECT = Path(os.environ.get("VC_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
UCE_ROOT = str((Path(os.environ.get("UCE_ROOT", PROJECT.parent / "external" / "UCE"))).as_posix())
WORK = str((Path(os.environ.get("UCE_WORK", PROJECT / "models" / "uce_work"))).as_posix())
SPECIES = os.environ.get("UCE_SPECIES", "zebra")
BATCH = int(os.environ.get("UCE_BATCH", "16"))

INP = str((PROJECT / "data" / "processed" / {
    "zebra": "zebrahub_t1_joint_humanorth.h5ad",
    "mouse": "mouse_t1_joint_uce.h5ad",
}[SPECIES]).as_posix())
UCE_SPECIES = "human" if SPECIES == "zebra" else "mouse"

os.chdir(UCE_ROOT)
sys.path.insert(0, UCE_ROOT)
os.makedirs(WORK, exist_ok=True)

import numpy as np
import torch
from accelerate import Accelerator

from evaluate import AnndataProcessor

np.random.seed(42)
torch.manual_seed(42)

args = Namespace(
    adata_path=INP,
    dir=WORK + "/",
    species=UCE_SPECIES,
    filter=False,
    skip=True,
    model_loc=None,  # 首次运行自动从 figshare 下载 4 层官方权重
    batch_size=BATCH,
    pad_length=1536,
    pad_token_idx=0,
    chrom_token_left_idx=1,
    chrom_token_right_idx=2,
    cls_token_idx=3,
    CHROM_TOKEN_OFFSET=143574,
    sample_size=1024,
    CXG=True,
    nlayers=4,
    output_dim=1280,
    d_hid=5120,
    token_dim=5120,
    multi_gpu=False,
    spec_chrom_csv_path="./model_files/species_chrom.csv",
    token_file="./model_files/all_tokens.torch",
    protein_embeddings_dir="./model_files/protein_embeddings/",
    offset_pkl_path="./model_files/species_offsets.pkl",
)

accelerator = Accelerator(project_dir=args.dir)
proc = AnndataProcessor(args, accelerator)
proc.preprocess_anndata()
proc.generate_idxs()
proc.run_evaluation()
print("UCE EMBED DONE", SPECIES)
