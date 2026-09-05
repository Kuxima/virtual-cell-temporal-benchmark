# -*- coding: utf-8 -*-
"""Legacy STATE CLI entry (`state emb`).

NOT used for the reported results: the reference embeddings were produced by
09_state_embed_safetensors.py (batch size 8 via STATE_BATCH). This script is
kept only as a CLI-compatibility path; its batch size now follows the same
STATE_BATCH variable so reruns cannot silently double the batch."""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import scanpy as sc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from common import DATA_PROC, RESULTS, SEED, STATE_MODEL_DIR

MODEL = str(STATE_MODEL_DIR)
INP = DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad"
OUT = DATA_PROC / "zebrahub_t1_joint_state.h5ad"
LABEL = "lineage"
K = 15
STATE_BATCH = os.environ.get("STATE_BATCH", "8")


def validate_existing_output():
    if not OUT.exists():
        return False
    out = sc.read_h5ad(OUT, backed="r")
    inp = sc.read_h5ad(INP, backed="r")
    try:
        if "X_state" not in out.obsm:
            return False
        if out.n_obs != inp.n_obs or not np.array_equal(
            out.obs_names.astype(str), inp.obs_names.astype(str)
        ):
            return False
        if out.obsm["X_state"].shape != (inp.n_obs, 2058):
            return False
        return True
    finally:
        out.file.close()
        inp.file.close()


def run_state():
    if validate_existing_output():
        print("已有有效 STATE 嵌入，跳过推理:", OUT)
        return
    if OUT.exists():
        print("已有 STATE 输出未通过 schema/顺序校验，将重新生成:", OUT)
        OUT.unlink()
    cmd = [
        "state",
        "emb",
        "--model-folder",
        MODEL,
        "--input",
        str(INP),
        "--output",
        str(OUT),
        "--batch-size",
        STATE_BATCH,
    ]
    print("RUN", " ".join(cmd))
    env = os.environ.copy()
    subprocess.check_call(cmd, env=env)


def eval_knn():
    adata = sc.read_h5ad(OUT)
    key = "X_state" if "X_state" in adata.obsm else None
    if key is None:
        # 兼容不同版本的键名
        cands = [k for k in adata.obsm.keys() if "state" in k.lower() or k.startswith("X_")]
        if not cands:
            raise SystemExit(f"找不到嵌入键，obsm={list(adata.obsm.keys())}")
        key = cands[0]
    print("使用嵌入键", key, adata.obsm[key].shape)
    train = adata[adata.obs["split"] == "train"]
    test = adata[adata.obs["split"] == "test"]
    y_train = train.obs[LABEL].astype(str).values
    y_test = test.obs[LABEL].astype(str).values
    clf = KNeighborsClassifier(n_neighbors=K, weights="distance", metric="cosine")
    clf.fit(train.obsm[key], y_train)
    pred = clf.predict(test.obsm[key])
    tp = test.obs["timepoint"].astype(str).values
    per_tp = {str(t): float(accuracy_score(y_test[tp == t], pred[tp == t])) for t in sorted(set(tp))}
    metrics = {
        "model": "STATE SE-600M + kNN (human ortholog genes)",
        "embedding_key": key,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(adata.n_vars),
        "label": LABEL,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy_per_timepoint": per_tp,
        "n_novel_types": int(len(set(y_test) - set(y_train))),
    }
    (RESULTS / "t1_state.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    t0 = time.time()
    run_state()
    eval_knn()
    print(f"STATE T1 DONE in {time.time()-t0:.0f}s")
