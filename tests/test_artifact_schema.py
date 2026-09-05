# -*- coding: utf-8 -*-
"""Validate the reference-run artifact contracts without loading model weights."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_result(path, model_keys):
    payload = load(path)
    for key in model_keys:
        d = payload[key]
        for metric in ("lineage_all", "cluster_shared", "lineage_on_novel_clusters"):
            assert metric in d and d[metric]["n"] > 0, (path, key, metric)


def main():
    results = ROOT / "results"
    check_result(results / "t1_unified_eval.json", ["PCA_frozen", "Harmony_adapted", "STATE_frozen"])
    check_result(results / "p4_classical_eval.json", ["PCA_frozen", "Harmony_adapted", "scVI_s42"])
    check_result(results / "p4_tf_eval.json", ["TF_mouse"])

    for path in (ROOT / "paper" / "tables").glob("*.csv"):
        frame = pd.read_csv(path)
        assert len(frame) > 0, path
    assert (ROOT / "paper" / "manuscript.md").exists()
    assert (ROOT / "REPRODUCIBILITY.md").exists()
    print("artifact schema checks passed")


if __name__ == "__main__":
    main()
