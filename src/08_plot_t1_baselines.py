# -*- coding: utf-8 -*-
"""画出 T1 三个基线的分时间点准确率折线（论文主图雏形）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt

from common import RESULTS

ORDER = ["19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
FILES = {
    "PCA": "t1_pca.json",
    "Harmony": "t1_harmony.json",
    "scVI/scANVI": "t1_scvi_baseline.json",
}


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for name, fn in FILES.items():
        d = json.loads((RESULTS / fn).read_text(encoding="utf-8"))
        ys = [d["accuracy_per_timepoint"][t] for t in ORDER]
        ax.plot(ORDER, ys, marker="o", label=f"{name}  ({d['accuracy']:.1%} overall)")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Lineage accuracy")
    ax.set_xlabel("Held-out developmental stage")
    ax.set_title("T1 temporal extrapolation on Zebrahub (train: 10–16 hpf)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = RESULTS / "t1_baseline_per_timepoint.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
