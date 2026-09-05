# -*- coding: utf-8 -*-
"""低显存 STATE 嵌入：config 建模型 + safetensors 灌权重，不加载 11GB Lightning ckpt。
batch_size 默认 8，适配 RTX 5080 16GB。"""
import os
import sys
import gc

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(__file__))

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from state.emb.inference import Inference
from state.emb.nn.model import StateEmbeddingModel
from state.emb.utils import get_dataset_cfg, get_precision_config

from common import DATA_PROC, STATE_MODEL_DIR

MODEL_DIR = STATE_MODEL_DIR
INP = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
OUT = DATA_PROC / "zebrahub_t1_joint_state.h5ad"
BATCH = int(os.environ.get("STATE_BATCH", "8"))


def build_model(cfg):
    token_dim = int(cfg.tokenizer.token_dim)
    d_model = int(cfg.model.emsize)
    nhead = int(cfg.model.nhead)
    d_hid = int(cfg.model.d_hid)
    nlayers = int(cfg.model.nlayers)
    output_dim = int(cfg.model.output_dim)
    emb_cnt = int(get_dataset_cfg(cfg).num_datasets)
    model = StateEmbeddingModel(
        token_dim=token_dim,
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        output_dim=output_dim,
        dropout=0.0,
        compiled=False,
        emb_cnt=emb_cnt,
        emb_size=token_dim,
        cfg=cfg,
    )
    return model


def main():
    cfg = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
    # 推理时关掉 flash attention 以外的训练开关；保留 rda / dataset_correction 与权重一致
    cfg.model.dropout = 0.0
    cfg.model.batch_size = BATCH
    cfg.model.use_flash_attention = False  # Windows + 无 triton 更稳
    print("building model...")
    model = build_model(cfg)
    print("loading safetensors on CPU...")
    sd = load_file(os.path.join(MODEL_DIR, "se600m_epoch4.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:8]:
        print(" missing sample", missing[:8])
    del sd
    gc.collect()

    protein = torch.load(os.path.join(MODEL_DIR, "protein_embeddings.pt"), map_location="cpu", weights_only=False)
    inferer = Inference(cfg=cfg, protein_embeds=protein)
    inferer.model = model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = get_precision_config(device_type=device)
    inferer.model = inferer.model.to(device=device, dtype=precision)
    if isinstance(protein, dict):
        all_pe = torch.vstack(list(protein.values()))
    else:
        all_pe = protein
    inferer.model.pe_embedding = torch.nn.Embedding.from_pretrained(all_pe.float())
    inferer.model.pe_embedding.to(device=device, dtype=precision)
    inferer.model.binary_decoder.requires_grad_(False)
    inferer.model.eval()
    print("model on", device, "dtype", precision, "batch", BATCH)
    if device == "cuda":
        print("VRAM allocated GB", torch.cuda.memory_allocated() / 1e9)

    print("encoding", INP)
    inferer.encode_adata(
        input_adata_path=str(INP),
        output_adata_path=str(OUT),
        emb_key="X_state",
        batch_size=BATCH,
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
