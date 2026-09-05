# -*- coding: utf-8 -*-
"""Small, dependency-light regression tests for evaluation contracts."""
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from common import shared_novel_mask


def load_eval_module():
    path = SRC / "11b_hard_eval_unified.py"
    spec = importlib.util.spec_from_file_location("eval11b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_novel_mask_uses_raw_labels():
    shared, novel = shared_novel_mask(
        np.array(["a", "b", "b"]), np.array(["a", "c", "b", "d"])
    )
    assert shared.dtype == bool
    assert np.array_equal(shared, [True, False, True, False])
    assert np.array_equal(novel, ~shared)


def test_shared_novel_mask_handles_numeric_like_labels():
    shared, novel = shared_novel_mask(
        np.array(["1.0", "2.0"]), np.array(["2.0", "3.0"])
    )
    assert np.array_equal(shared, [True, False])
    assert np.array_equal(novel, [False, True])


def test_knn_vote_shape_and_label_range():
    module = load_eval_module()
    idx = np.array([[0, 1], [1, 0]], dtype=np.int32)
    sim = np.array([[0.9, 0.2], [0.9, 0.1]], dtype=np.float32)
    pred = module.vote_predict(idx, sim, np.array([0, 1]), 2, 2)
    assert pred.shape == (2,)
    assert np.all((pred >= 0) & (pred < 2))


def test_eval_embedding_handles_empty_novel_subset():
    """Regression (I19 interpolation control): when every test cluster is shared,
    lineage_on_novel_clusters must be {'n': 0} and eval_embedding must not crash."""
    module = load_eval_module()
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(40, 4)).astype(np.float32)
    Xte = rng.normal(size=(12, 4)).astype(np.float32)
    ctx = dict(
        yte_lin=np.array([0, 1] * 6),
        yte_clu=np.array([0, 1] * 6),
        shared=np.ones(12, dtype=bool),
        novel=np.zeros(12, dtype=bool),
        lin_codes_tr=np.array([0, 1] * 20),
        lin_codes_te=np.array([0, 1] * 6),
        clu_codes_tr=np.array([0, 1] * 20),
        lin_classes=[0, 1], clu_classes=[0, 1], clu2lin_code={0: 0, 1: 1},
        tp_te=np.array(["t1", "t2"] * 6),
        fish_te_codes=np.array([0, 1] * 6),
    )
    out = module.eval_embedding("synthetic", "frozen", Xtr, Xte,
                                np.array(["a", "b"]), np.array(["c1", "c2"]), ctx)
    assert out["lineage_on_novel_clusters"] == {"n": 0}
    assert "accuracy" not in out["lineage_on_novel_clusters"]


def test_index_subset_must_be_sorted_to_match_mask_slice_order():
    """Regression (random-split label misalignment): an embedding sliced with a
    boolean mask keeps file order, so an index-based subset must be sorted
    before building the label frame. Unsorted (permutation-order) indices
    produce a different row order than the mask slice."""
    n = 50
    obs_names = np.array([f"cell_{i:03d}" for i in range(n)])
    tr_m = np.zeros(n, dtype=bool)
    perm = np.random.default_rng(42).permutation(n)
    tr_idx, te_idx = perm[:40], perm[40:]
    tr_m[tr_idx] = True

    unsorted_sub = obs_names[np.concatenate([tr_idx, te_idx])]
    sorted_sub = obs_names[np.concatenate([np.sort(tr_idx), np.sort(te_idx)])]
    mask_slice = obs_names[np.concatenate([np.where(tr_m)[0], np.where(~tr_m)[0]])]

    assert not np.array_equal(unsorted_sub, mask_slice)
    assert np.array_equal(sorted_sub, mask_slice)


def test_result_contracts_are_nonempty_and_consistent():
    results = ROOT / "results"
    checks = {
        "t1_unified_eval.json": (102335, 92482, 9853),
        "p4_classical_eval.json": (88549, 61784, 26765),
        "p4_tf_eval.json": (88549, 61784, 26765),
    }
    for filename, (n_test, n_shared, n_novel) in checks.items():
        payload = json.loads((results / filename).read_text(encoding="utf-8"))
        if filename == "t1_unified_eval.json":
            models = ["PCA_frozen", "Harmony_adapted", "STATE_frozen"]
        elif filename == "p4_classical_eval.json":
            models = ["PCA_frozen", "Harmony_adapted", "scVI_s42"]
        else:
            models = ["TF_mouse"]
        for model in models:
            assert payload[model]["lineage_all"]["n"] == n_test
            assert payload[model]["cluster_shared"]["n"] == n_shared
            assert payload[model]["lineage_on_novel_clusters"]["n"] == n_novel


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("evaluation contract tests passed")
