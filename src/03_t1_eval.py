# -*- coding: utf-8 -*-
"""03 T1 时序外推基线评测：早期时间点为参考 → 预测晚期细胞类型。
方法无关：任何模型只要产出 obsm 嵌入即可接入（scVI/UCE/STATE/TranscriptFormer/Harmony/PCA）。
用法: python 03_t1_eval.py --h5ad <processed.h5ad> --embedding X_scVI --time_col timepoint --label_col lineage
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, T1_TRAIN_MAX_HPF, T1_TEST_MIN_HPF, SEED


def parse_hpf(val):
    """Convert hpf/dpf labels to a common hpf-scale developmental time."""
    if np.issubdtype(type(val), np.number):
        return float(val)
    import re
    text = str(val).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    value = float(match.group(1))
    if "dpf" in text or "day" in text:
        return value * 24.0
    return value


def main(h5ad_path: str, embedding: str, time_col: str, label_col: str,
         train_max: float, test_min: float, k: int = 15):
    adata = sc.read_h5ad(h5ad_path)
    print(f"数据: {adata.shape}, 嵌入: {embedding}, 时间列: {time_col}")

    hpf = adata.obs[time_col].map(parse_hpf)
    if hpf.isna().any():
        raise ValueError(f"无法解析时间列 {time_col} 中的部分值")
    train_mask = (hpf <= train_max).to_numpy()
    test_mask = (hpf >= test_min).to_numpy()
    if np.any(train_mask & test_mask):
        raise ValueError("train/test time thresholds overlap")
    print(f"训练(≤{train_max}hpf): {train_mask.sum()} cells; "
          f"测试(≥{test_min}hpf): {test_mask.sum()} cells")

    if test_mask.sum() < 100:
        sys.exit("测试集过小，请检查时间切分阈值")

    # 嵌入矩阵（PCA 情形下现场计算）
    if embedding == "X_pca":
        sc.tl.pca(adata, n_comps=50, random_state=SEED)
    X = adata.obsm[embedding]

    y_train = adata.obs[label_col].astype(str).values[train_mask]
    y_test = adata.obs[label_col].astype(str).values[test_mask]

    # 测试集中出现但参考集没有的细胞类型（外推难度的关键统计）
    novel_types = set(y_test) - set(y_train)
    print(f"测试集细胞类型 {len(set(y_test))} 种，其中 {len(novel_types)} 种未在参考集出现")

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(X[train_mask], y_train)
    pred = clf.predict(X[test_mask])

    metrics = {
        "embedding": embedding,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "n_novel_types": len(novel_types),
        # 只在参考集已知类型上的表现（排除"根本无法预测"的新类型后）
        "known_mask": None,
    }
    known = np.isin(y_test, list(set(y_train)))
    if known.sum() > 0:
        metrics["accuracy_known_types_only"] = float(accuracy_score(y_test[known], pred[known]))

    out = RESULTS / f"t1_{embedding.replace('X_', '')}.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"指标已保存: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--embedding", default="X_scVI")
    ap.add_argument("--time_col", default="timepoint")
    ap.add_argument("--label_col", default="lineage")
    ap.add_argument("--train_max", type=float, default=T1_TRAIN_MAX_HPF)
    ap.add_argument("--test_min", type=float, default=T1_TEST_MIN_HPF)
    args = ap.parse_args()
    main(args.h5ad, args.embedding, args.time_col, args.label_col, args.train_max, args.test_min)
