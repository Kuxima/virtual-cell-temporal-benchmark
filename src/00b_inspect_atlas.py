# -*- coding: utf-8 -*-
"""用 h5py 直接检查 Zebrahub h5ad 结构（不经 anndata，避免整层加载）。"""
import h5py
import numpy as np
import pandas as pd
from common import DATA_RAW

path = DATA_RAW / "zebrahub" / "zf_atlas_full_v4_release.h5ad"

with h5py.File(path, "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: {obj.shape} {obj.dtype}")
    print("== 顶层键:", list(f.keys()))
    print("== X:", type(f["X"]))
    try:
        print("   X keys:", list(f["X"].keys()), {k: f["X"][k].shape for k in f["X"]})
    except Exception:
        print("   X:", f["X"])
    print("== layers:", list(f["layers"].keys()) if "layers" in f else None)
    print("== obsm:", list(f["obsm"].keys()) if "obsm" in f else None)

    obs = f["obs"]
    print("\n== obs 列:")
    for k in obs.keys():
        item = obs[k]
        if isinstance(item, h5py.Dataset):
            print(f"  {k}: {item.shape} {item.dtype}")
        else:
            print(f"  {k}: group {list(item.keys())}")

    # 读关键列（类别列在 codes/categories 里）
    def read_col(k):
        g = obs[k]
        if isinstance(g, h5py.Dataset):
            return pd.Series(np.array(g), name=k)
        cats = [c.decode() if isinstance(c, bytes) else c for c in np.array(g["categories"])]
        codes = np.array(g["codes"])
        return pd.Series([cats[c] if c >= 0 else None for c in codes], name=k)

    for cand in ["Time_point", "timepoint", "time_point", "stage", "Stage", "hpf"]:
        if cand in obs:
            print(f"\n[{cand}] 分布:")
            print(read_col(cand).value_counts().to_string())

    for cand in ["cell_type", "Cell_Type", "celltype", "major_celltype", "cell_identity"]:
        if cand in obs:
            s = read_col(cand)
            print(f"\n[{cand}] 类别数: {s.nunique()}, 前 25:")
            print(s.value_counts().head(25).to_string())
