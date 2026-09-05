# -*- coding: utf-8 -*-
"""13b TranscriptFormer tf-metazoa 自定义推理 v2（动态长度裁剪版）。
相对 v1 的关键优化：
- 细胞按非零基因数全局排序、同长度聚块；GPU 子批裁掉尾部全-pad 列
  （pad 列在注意力中本被 pad_mask 屏蔽，裁剪数学等价，平均序列 2047→~450）
- sort_genes=True：截断时保留表达量最高的基因
- 自适应 GPU batch（OOM 减半），异常兜底重试
输出 obsm['X_tf']。"""
import gc
import json
import os
import sys
import time
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scanpy as sc
import torch
from omegaconf import OmegaConf

from common import DATA_PROC, TF_MODEL_DIR

CKPT = str(TF_MODEL_DIR)
SRC = Path(os.environ.get("TF_SRC", str(DATA_PROC / "zebrahub_t1_joint_tfinput.h5ad")))
OUT = Path(os.environ.get("TF_OUT", str(DATA_PROC / "zebrahub_t1_joint_tf.h5ad")))
CHUNK = 2048
GPU_B_START = 8
MIN_LEN = 16


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
    print(f"词表 {len(gene_vocab)}，emb_matrix {tuple(emb_matrix.shape)} ({time.time()-t0:.0f}s)", flush=True)

    adata = sc.read_h5ad(SRC)
    X = adata.X.tocsr()
    names_all = adata.var["ensembl_id"].astype(str).values
    filter_idx = np.array([i for i, n in enumerate(names_all) if n in gene_vocab])
    gene_names = np.array(names_all)[filter_idx]
    print(f"词表命中基因 {len(gene_names)}/{len(names_all)}", flush=True)

    # 每细胞非零基因数（在词表过滤后的基因空间）
    Xf = X[:, filter_idx]
    nnz = np.asarray((Xf > 0).sum(axis=1)).ravel()
    print(f"非零基因: median={np.median(nnz):.0f} p95={np.percentile(nnz, 95):.0f} max={nnz.max()}", flush=True)
    order = np.argsort(-nnz)  # 长细胞在前，同长度聚块

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
    model = model.cuda()
    print(f"模型就绪 ({time.time()-t0:.0f}s)", flush=True)

    gtok = BatchGeneTokenizer(gene_vocab)
    atok = BatchObsTokenizer(aux_vocab)
    max_len = int(cfg.model.model_config.seq_len)
    pad_token = cfg.model.data_config.gene_pad_token

    n = adata.n_obs
    embs = None
    gpu_b = GPU_B_START
    n_done = 0
    CKPT = DATA_PROC / "tf_embed_ckpt.npz"
    start_chunk = 0
    if CKPT.exists():
        z = np.load(CKPT)
        embs = z["embs"]
        start_chunk = int(z["next_chunk"])
        n_done = min(start_chunk * CHUNK, n)
        print(f"续跑：从第 {start_chunk} 块开始（已完成 {n_done} 细胞）", flush=True)
    t0 -= n_done * 0  # 计时保持总口径
    for cs in range(0, n, CHUNK):
        rows = order[cs:cs + CHUNK]
        x = Xf[rows].toarray().astype(np.float32)
        obs = adata.obs.iloc[rows][["assay"]]
        res = process_batch(
            x, obs, gene_names, gtok, atok,
            True, False, max_len, True, pad_token,
            gene_vocab, 0, 30, aux_vocab,
        )
        counts = res["gene_counts"]
        tokens = res["gene_token_indices"]
        aux = res["aux_token_indices"]
        g = 0
        while g < len(rows):
            b = min(gpu_b, len(rows) - g)
            L = max(int(nnz[rows[g:g + b]].max()) + 4, MIN_LEN)
            L = min((L + 127) // 128 * 128, counts.shape[1])  # 对齐 128，规避 flex fallback 奇异 shape
            sub = BatchData(
                gene_counts=counts[g:g + b, :L].cuda(non_blocking=True),
                gene_token_indices=tokens[g:g + b, :L].cuda(non_blocking=True),
                aux_token_indices=aux[g:g + b].cuda(non_blocking=True) if aux is not None else None,
            )
            ok = False
            for _ in range(4):
                try:
                    with torch.no_grad():
                        out = model(batch=sub, embed=True)
                    emb = out["embeddings"].float().cpu().numpy()
                    if embs is None:
                        embs = np.zeros((n, emb.shape[1]), dtype=np.float32)
                    embs[rows[g:g + b]] = emb
                    ok = True
                    break
                except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                        raise
                    torch.cuda.empty_cache()
                    gpu_b = max(1, gpu_b // 2)
                    b = min(gpu_b, len(rows) - g)
                    L = max(int(nnz[rows[g:g + b]].max()) + 4, MIN_LEN)
                    L = min(L, counts.shape[1])
                    sub = BatchData(
                        gene_counts=counts[g:g + b, :L].cuda(),
                        gene_token_indices=tokens[g:g + b, :L].cuda(),
                        aux_token_indices=aux[g:g + b].cuda() if aux is not None else None,
                    )
                    print(f"  OOM→gpu_b={gpu_b}", flush=True)
            if not ok:
                raise RuntimeError("连续 OOM，放弃")
            g += b
            n_done += b
        del res, counts, tokens, aux, x
        gc.collect()
        torch.cuda.empty_cache()
        if (cs // CHUNK) % 4 == 0:
            r = (time.time() - t0) / max(n_done, 1)
            print(f"  {n_done}/{n} ({time.time()-t0:.0f}s, {r*1000:.1f}ms/cell, gpu_b={gpu_b})", flush=True)

    # 写回（按原顺序）
    adata.obsm["X_tf"] = embs
    adata.write_h5ad(OUT, compression="gzip")
    print(f"saved {OUT} dim={embs.shape[1]} 总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
