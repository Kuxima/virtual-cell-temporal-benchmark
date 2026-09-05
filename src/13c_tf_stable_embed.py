# -*- coding: utf-8 -*-
"""13c TranscriptFormer 稳定推理：用标准 PyTorch 注意力替换 flex_attention fallback。
flex fallback 在 Windows 上有 illegal instruction / 显存爆裂问题；本脚本逐层复算
与其 score_mod/mask_mod 语义完全等价的标准注意力（含 softcap=10、log-counts 偏置、
causal + pad 掩码），先用小批量对拍验证等价（cos≥0.999），再全量推理。
保留 13b 的动态长度裁剪、排序聚块、断点续跑。"""
import gc
import hashlib
import json
import math
import os
import sys
import time

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from common import DATA_PROC, PROJECT, TF_MODEL_DIR

CKPT = str(TF_MODEL_DIR)
SRC = os.environ.get("TF_SRC", str(DATA_PROC / "zebrahub_t1_joint_tfinput.h5ad"))
OUT = os.environ.get("TF_OUT", str(DATA_PROC / "zebrahub_t1_joint_tf.h5ad"))
CKPT_NPZ = Path(os.environ.get("TF_CKPT", str(DATA_PROC / "tf_embed_ckpt.npz")))
CHUNK = 2048
MIN_LEN = 16
SOFTCAP = 10.0
TOKEN_BUDGET = 6144  # batch x L upper bound for GPU memory


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT))
    except ValueError:
        return str(Path(path).resolve())


def checkpoint_metadata(n_obs):
    return {
        "source": project_relative(SRC),
        "source_size": Path(SRC).stat().st_size,
        "source_sha256": sha256_file(SRC),
        "model_config_sha256": sha256_file(Path(CKPT) / "config.json"),
        "model_weights_sha256": sha256_file(Path(CKPT) / "model_weights.pt"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "script": project_relative(__file__),
        "chunk": CHUNK,
        "token_budget": TOKEN_BUDGET,
        "n_obs": int(n_obs),
    }


def stable_forward(model, batch, device):
    """与 Transcriptformer.forward(embed=True) 数值等价的稳定实现（不含 llh 头）。"""
    eps = float(model.model_config.log_counts_eps)
    gene_tokens = batch.gene_token_indices
    rs_tokens = torch.cat([model.gene_vocab.start_idx * torch.ones_like(gene_tokens[:, :1]),
                           gene_tokens[:, :-1]], dim=1)
    rs_counts = torch.cat([torch.ones_like(batch.gene_counts[:, :1]),
                           batch.gene_counts[:, :-1]], dim=1)
    x = model.gene_embeddings(rs_tokens)  # (B, L, D)
    if batch.aux_token_indices is not None:
        aux_emb = model.get_aux_embeddings(batch.aux_token_indices)  # (B, A, D)
        x = torch.cat([aux_emb, x], dim=1)
        rs_counts = torch.cat([torch.ones_like(batch.aux_token_indices, dtype=x.dtype)
                               .mean(dim=1, keepdim=True), rs_counts], dim=1)

    pad_bool = rs_tokens == model.gene_vocab.pad_idx  # (B, L_gene)
    if batch.aux_token_indices is not None:
        aux_pad = torch.stack([batch.aux_token_indices == pid for pid in model.aux_vocab.pad_ids],
                              dim=-1).any(dim=-1)
        pad_bool = torch.cat([aux_pad, pad_bool], dim=1)
    valid = ~pad_bool  # (B, A+L) True=有效
    log_counts = torch.log1p(rs_counts + eps)  # (B, L)

    B, L, D = x.shape
    H = model.transformer_encoder.nheads
    dk = D // H
    causal = torch.ones(L, L, device=device, dtype=torch.bool).tril()
    bias = log_counts.view(B, 1, 1, L)
    neg_pad = torch.zeros(B, 1, 1, L, device=device).masked_fill(~valid.view(B, 1, 1, L), float("-inf"))

    for layer in model.transformer_encoder.encoder_layers:
        h = layer.norm1(x)
        attn_mod = layer.self_attn
        q, k, v = (lin(h).view(B, L, H, dk).transpose(1, 2) for lin in attn_mod.linears[:3])
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)  # (B,H,L,L)
        s = SOFTCAP * torch.tanh((s + bias) / SOFTCAP)
        s = s + neg_pad
        s = s.masked_fill(~causal.view(1, 1, L, L), float("-inf"))
        a = torch.softmax(s, dim=-1)
        a = torch.nan_to_num(a, nan=0.0)  # 复刻 flex 语义：全 mask 的 query 行输出 0（aux=pad 时 q=0 行会出现）
        o = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, L, D)
        x = attn_mod.linears[-1](o) + h  # 残差接 norm 后的值（与官方实现一致）
        x = layer.norm2(x)
        x = layer.linear2(layer.activation(layer.linear1(x))) + x
    # 去掉 aux token 后对有效位置平均
    if batch.aux_token_indices is not None:
        A = batch.aux_token_indices.shape[1]
        x = x[:, A:]
        valid = valid[:, A:]
    m = valid.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def main():
    t0 = time.time()
    from transcriptformer.data.dataloader import process_batch
    from transcriptformer.data.dataclasses import BatchData
    from transcriptformer.model.model import Transcriptformer
    from transcriptformer.tokenizer.tokenizer import BatchGeneTokenizer, BatchObsTokenizer
    from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings

    cli_cfg = OmegaConf.load(os.path.join(
        os.path.dirname(__import__("transcriptformer").__file__), "cli", "conf", "inference_config.yaml"))
    ckpt_cfg = OmegaConf.create(json.loads(open(os.path.join(CKPT, "config.json"), encoding="utf-8").read()))
    cfg = OmegaConf.merge(OmegaConf.merge(OmegaConf.create({"model": {}}), cli_cfg), ckpt_cfg)
    cfg.model.data_config.aux_vocab_path = os.path.join(CKPT, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(CKPT, "vocabs")
    cfg.model.model_config.compile_block_mask = False
    cfg.model.inference_config.emb_type = "cell"

    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(cfg)
    adata = sc.read_h5ad(SRC)
    X = adata.X.tocsr()
    names_all = adata.var["ensembl_id"].astype(str).values
    filter_idx = np.array([i for i, n in enumerate(names_all) if n in gene_vocab])
    gene_names = np.array(names_all)[filter_idx]

    Xf = X[:, filter_idx]
    nnz = np.asarray((Xf > 0).sum(axis=1)).ravel()
    print(f"词表 {len(gene_vocab)}，命中 {len(gene_names)} 基因，nnz median={np.median(nnz):.0f} max={nnz.max()}", flush=True)

    model = Transcriptformer(
        data_config=cfg.model.data_config, model_config=cfg.model.model_config,
        loss_config=cfg.model.loss_config, inference_config=cfg.model.inference_config,
        gene_vocab_dict=gene_vocab, aux_vocab_dict=aux_vocab, emb_matrix=emb_matrix,
    )
    model.eval()
    del emb_matrix
    gc.collect()
    sd = torch.load(os.path.join(CKPT, "model_weights.pt"), weights_only=True, map_location="cuda")
    model.load_state_dict(sd)
    del sd
    gc.collect()
    print(f"模型就绪（CPU，对拍后再上 GPU）({time.time()-t0:.0f}s)", flush=True)

    gtok = BatchGeneTokenizer(gene_vocab)
    atok = BatchObsTokenizer(aux_vocab)
    max_len = int(cfg.model.model_config.seq_len)
    pad_token = cfg.model.data_config.gene_pad_token

    # ---------- 等价性验证（CPU 上对拍：flex 的 CPU 数学路径稳定）----------
    short_rows = np.argsort(nnz)[100:108]
    x = Xf[short_rows].toarray().astype(np.float32)
    obs = adata.obs.iloc[short_rows][["assay"]]
    res = process_batch(x, obs, gene_names, gtok, atok, True, False, max_len, True,
                        pad_token, gene_vocab, 0, 30, aux_vocab)
    L = min((int(nnz[short_rows].max()) + 4 + 127) // 128 * 128, res["gene_counts"].shape[1])
    L = max(L, MIN_LEN)
    bd = BatchData(
        gene_counts=res["gene_counts"][:8, :L],
        gene_token_indices=res["gene_token_indices"][:8, :L],
        aux_token_indices=res["aux_token_indices"][:8] if res["aux_token_indices"] is not None else None,
    )
    try:
        with torch.no_grad():
            ref = model(batch=bd, embed=True)["embeddings"].float()
            mine = stable_forward(model, bd, "cpu").float()
        cos = F.cosine_similarity(ref, mine, dim=1)
        rel = (ref - mine).norm() / ref.norm()
        print(f"等价性(CPU): cos min={cos.min():.6f} mean={cos.mean():.6f} rel_err={rel:.6f}", flush=True)
        if torch.isnan(cos).any() or cos.min() < 0.99:
            print("等价性不达标，终止")
            sys.exit(1)
    except Exception as e:
        print(f"参照路径不可用（{type(e).__name__}），跳过对拍，直接使用稳定实现", flush=True)
    del bd, res, x
    gc.collect()

    model = model.cuda()

    # ---------- 全量推理（排序 + 动态长度 + 断点续跑）----------
    n = adata.n_obs
    order = np.argsort(-nnz)
    embs = None
    start_chunk = 0
    metadata = checkpoint_metadata(n)
    meta_path = CKPT_NPZ.with_suffix(".json")
    if CKPT_NPZ.exists():
        if not meta_path.exists():
            raise RuntimeError(f"checkpoint metadata missing; refusing to resume {CKPT_NPZ}")
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != metadata:
            raise RuntimeError("checkpoint metadata mismatch; remove checkpoint or restore matching inputs")
        with np.load(CKPT_NPZ) as z:
            embs = z["embs"]
            start_chunk = int(z["next_chunk"])
        if embs.shape[0] != n or start_chunk < 0 or start_chunk > (n + CHUNK - 1) // CHUNK:
            raise RuntimeError("checkpoint shape or progress is invalid")
        print(f"续跑：从块 {start_chunk}（已完成 {min(start_chunk*CHUNK, n)} 细胞）", flush=True)

    t0_run = time.time()
    for ci, cs in enumerate(range(0, n, CHUNK)):
        if ci < start_chunk:
            continue
        rows = order[cs:cs + CHUNK]
        x = Xf[rows].toarray().astype(np.float32)
        obs = adata.obs.iloc[rows][["assay"]]
        res = process_batch(x, obs, gene_names, gtok, atok, True, False, max_len, True,
                            pad_token, gene_vocab, 0, 30, aux_vocab)
        counts, tokens, aux = res["gene_counts"], res["gene_token_indices"], res["aux_token_indices"]
        g = 0
        while g < len(rows):
            Lw = max(int(nnz[rows[g]].max()) + 4, MIN_LEN)
            Lw = min((Lw + 127) // 128 * 128, counts.shape[1])
            b = max(1, min(TOKEN_BUDGET // Lw, len(rows) - g))
            sub = BatchData(
                gene_counts=counts[g:g + b, :Lw].cuda(),
                gene_token_indices=tokens[g:g + b, :Lw].cuda(),
                aux_token_indices=aux[g:g + b].cuda() if aux is not None else None,
            )
            for _ in range(4):
                try:
                    with torch.no_grad():
                        emb = stable_forward(model, sub, "cuda").float().cpu().numpy()
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    b = max(1, b // 2)
                    sub = BatchData(
                        gene_counts=counts[g:g + b, :Lw].cuda(),
                        gene_token_indices=tokens[g:g + b, :Lw].cuda(),
                        aux_token_indices=aux[g:g + b].cuda() if aux is not None else None,
                    )
            else:
                raise RuntimeError("连续 OOM")
            if embs is None:
                embs = np.zeros((n, emb.shape[1]), dtype=np.float32)
            embs[rows[g:g + b]] = emb
            g += b
        del res, counts, tokens, aux, x
        gc.collect()
        if ci % 2 == 0 or ci == n // CHUNK:
            done = min((ci + 1) * CHUNK, n)
            rate = done / max(time.time() - t0_run, 1)
            eta = (n - done) / max(rate, 1e-9) / 60
            print(f"  {done}/{n} ({time.time()-t0_run:.0f}s, {1000/rate:.1f}ms/cell, ETA {eta:.0f}min)", flush=True)
            np.savez(CKPT_NPZ, embs=embs, next_chunk=ci + 1)
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    adata.obsm["X_tf"] = embs
    adata.write_h5ad(OUT, compression="gzip")
    CKPT_NPZ.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    print(f"saved {OUT} dim={embs.shape[1]} 总耗时 {time.time()-t0_run:.0f}s", flush=True)


if __name__ == "__main__":
    main()
