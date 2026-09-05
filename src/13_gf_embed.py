# -*- coding: utf-8 -*-
"""13 Geneformer V1-10M（6 层 30M，Genecorpus-30M）嵌入 T1 joint 数据。
管线：humanorth 符号 → ENSG（mygene）→ 官方 TranscriptomeTokenizer(V1) →
BertModel 前向 → 非_pad token 平均池化 → obsm['X_gf']。

Windows 注意：
  - datasets 的 multiprocess 子进程会以 __mp_main__ 重新导入本脚本 →
    一切副作用（构建输入/加载模型）必须放在函数内且由 __main__ 守卫调用，
    否则子进程会并发重写输入文件造成 h5py 锁冲突。
  - tokenizer 经 importlib shim 加载（绕开 geneformer/__init__ 重依赖链）。"""
import importlib.util
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from common import DATA_PROC, RESULTS, GENEFORMER_REPO, GENEFORMER_MODEL_DIR

REPO = str(GENEFORMER_REPO)
MODEL_DIR = str(GENEFORMER_MODEL_DIR)


def paths():
    from common import DATA_PROC, RESULTS
    return DATA_PROC, RESULTS


def build_input(INP):
    import numpy as np
    import scanpy as sc
    DATA_PROC, RESULTS = paths()
    sym2ensg = json.loads((RESULTS / "sym2ensg_mygene.json").read_text())
    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    ens = [sym2ensg.get(s) for s in joint.var_names.astype(str)]
    keep = np.array([e is not None for e in ens])
    adata = joint[:, keep].copy()
    adata.var.index = [e for e in ens if e is not None]
    adata.obs["cell_idx"] = np.arange(adata.n_obs).astype(str)
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        adata.obs[c] = adata.obs[c].astype(str)
    adata.write_h5ad(INP, compression="gzip")
    del joint, adata
    print(f"GF 输入写入 {INP}", flush=True)


def load_tokenizer():
    pkg = types.ModuleType("geneformer")
    pkg.__path__ = [os.path.join(REPO, "geneformer")]
    g30 = os.path.join(REPO, "geneformer", "gene_dictionaries_30m")
    pkg.GENE_MEDIAN_FILE = os.path.join(REPO, "geneformer", "gene_median_dictionary_gc104M.pkl")
    pkg.TOKEN_DICTIONARY_FILE = os.path.join(REPO, "geneformer", "token_dictionary_gc104M.pkl")
    pkg.ENSEMBL_DICTIONARY_FILE = os.path.join(REPO, "geneformer", "gene_name_id_dict_gc104M.pkl")
    pkg.ENSEMBL_MAPPING_FILE = os.path.join(REPO, "geneformer", "ensembl_mapping_dict_gc104M.pkl")
    pkg.GENE_MEDIAN_FILE_30M = os.path.join(g30, "gene_median_dictionary_gc30M.pkl")
    pkg.TOKEN_DICTIONARY_FILE_30M = os.path.join(g30, "token_dictionary_gc30M.pkl")
    pkg.ENSEMBL_DICTIONARY_FILE_30M = os.path.join(g30, "gene_name_id_dict_gc30M.pkl")
    pkg.ENSEMBL_MAPPING_FILE_30M = os.path.join(g30, "ensembl_mapping_dict_gc30M.pkl")
    sys.modules.setdefault("geneformer", pkg)
    spec = importlib.util.spec_from_file_location(
        "geneformer.tokenizer", os.path.join(REPO, "geneformer", "tokenizer.py"))
    tok_mod = importlib.util.module_from_spec(spec)
    sys.modules["geneformer.tokenizer"] = tok_mod
    spec.loader.exec_module(tok_mod)
    return tok_mod


def main():
    import numpy as np
    import scanpy as sc
    import torch
    DATA_PROC, RESULTS = paths()
    GF_DIR = DATA_PROC / "gf"
    GF_DIR.mkdir(exist_ok=True)
    INP = GF_DIR / "gf_input.h5ad"
    OUT = DATA_PROC / "zebrahub_t1_joint_gf.h5ad"
    t0 = time.time()

    if not INP.exists():
        build_input(INP)
    tok_mod = load_tokenizer()
    tk = tok_mod.TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_idx": "cell_idx", "split": "split"},
        nproc=1, model_version="V1", use_h5ad_index=True,
    )
    tk.tokenize_data(data_directory=str(GF_DIR), output_directory=str(GF_DIR),
                     output_prefix="gf_tokens", file_format="h5ad",
                     input_identifier="gf_input")
    print(f"tokenize 完成 {time.time()-t0:.0f}s", flush=True)

    from datasets import load_from_disk
    from transformers import AutoModel

    ds = load_from_disk(str(GF_DIR / "gf_tokens.dataset"))
    cell_idx = np.array([int(x) for x in ds["cell_idx"]])
    lengths = np.array([len(x) for x in ds["input_ids"]])
    print(f"token 数: median={np.median(lengths):.0f} p95={np.percentile(lengths, 95):.0f} "
          f"max={lengths.max()}，细胞 {len(ds)}", flush=True)

    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")
    model = AutoModel.from_pretrained(MODEL_DIR).cuda().eval()
    HID = model.config.hidden_size
    emb = np.zeros((joint.n_obs, HID), dtype=np.float32)

    order = np.argsort(lengths)
    BATCH = 256
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            rows = order[s:s + BATCH]
            seqs = [ds[int(i)]["input_ids"] for i in rows]
            L = max(len(x) for x in seqs)
            ids = np.full((len(seqs), L), 0, dtype=np.int64)
            mask = np.zeros((len(seqs), L), dtype=bool)
            for j, x in enumerate(seqs):
                ids[j, :len(x)] = x
                mask[j, :len(x)] = True
            out = model(input_ids=torch.from_numpy(ids).cuda(),
                        attention_mask=torch.from_numpy(mask).cuda().long()
                        ).last_hidden_state
            m = torch.from_numpy(mask).cuda().unsqueeze(-1).float()
            mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
            emb[cell_idx[rows]] = mean.float().cpu().numpy()
            if (s // BATCH) % 100 == 0:
                print(f"  {s}/{len(order)} ({time.time()-t0:.0f}s)", flush=True)

    hit = np.zeros(joint.n_obs, dtype=bool)
    hit[cell_idx] = True
    joint.obsm["X_gf"] = emb
    joint.obs["gf_covered"] = hit
    joint.write_h5ad(OUT, compression="gzip")
    print(f"saved {OUT}，GF 覆盖 {hit.sum()}/{joint.n_obs}，总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
