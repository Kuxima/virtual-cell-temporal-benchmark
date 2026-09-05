# -*- coding: utf-8 -*-
"""Shared project paths and constants.

Paths are derived from the project location and can be overridden with
VC_PROJECT_ROOT, VC_DATA_ROOT, VC_MODEL_ROOT and VC_BIOMNI_ROOT.
"""
import os
from pathlib import Path

import numpy as np


PROJECT = Path(os.environ.get("VC_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("VC_DATA_ROOT", PROJECT / "data")).resolve()
DATA_RAW = DATA_ROOT / "raw"
DATA_PROC = DATA_ROOT / "processed"
RESULTS = Path(os.environ.get("VC_RESULTS_ROOT", PROJECT / "results")).resolve()
LOGS = Path(os.environ.get("VC_LOG_ROOT", PROJECT / "logs")).resolve()
MODEL_ROOT = Path(os.environ.get("VC_MODEL_ROOT", PROJECT / "models")).resolve()
BIOMNI_ROOT = Path(
    os.environ.get("VC_BIOMNI_ROOT", PROJECT / "external" / "biomni")
).resolve()
MODEL_CACHE = Path(os.environ.get("BIOMNI_DATA_PATH", DATA_ROOT / "biomni_cache")).resolve()
STATE_MODEL_DIR = Path(os.environ.get("STATE_MODEL_DIR", MODEL_ROOT / "SE-600M")).resolve()
TF_MODEL_DIR = Path(os.environ.get("TF_MODEL_DIR", MODEL_ROOT / "tf_metazoa")).resolve()
GENEFORMER_REPO = Path(
    os.environ.get("GENEFORMER_REPO", MODEL_ROOT / "geneformer_repo")
).resolve()
GENEFORMER_MODEL_DIR = Path(
    os.environ.get("GENEFORMER_MODEL_DIR", GENEFORMER_REPO / "Geneformer-V1-10M")
).resolve()

# T1 time split used by the legacy generic evaluator.
T1_TRAIN_TP = ("10hpf", "12hpf", "14hpf", "16hpf")
T1_TRAIN_MAX_HPF = 16.0
T1_TEST_MIN_HPF = 19.0

# Random seed fixed for reported experiments.
SEED = 42

# Shared model palette used by all paper figures (keep in sync across panels).
# Okabe-Ito colourblind-safe palette (Nature guideline); adapted = lighter tint of scVI.
MODEL_COLORS = {
    "majority": "#7f7f7f",
    "PCA-50": "#56B4E9",
    "Harmony": "#0072B2",
    "scVI": "#009E73",
    "scVI adapted": "#7fd0a8",
    "STATE SE-600M": "#CC79A7",
    "UCE": "#F0E442",
    "Geneformer V1": "#E69F00",
    "TranscriptFormer": "#D55E00",
}
BASELINE_COLOR = "#111111"  # majority reference line; black keeps it distinct from every model colour


def ensure_project_dirs():
    """Create writable output directories at an explicit pipeline boundary."""
    for directory in (DATA_RAW, DATA_PROC, RESULTS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def shared_novel_mask(train_clusters, test_clusters):
    """Return explicit shared/novel masks using raw cluster labels."""
    train_labels = np.asarray(train_clusters, dtype=str)
    test_labels = np.asarray(test_clusters, dtype=str)
    shared = np.isin(test_labels, np.unique(train_labels))
    novel = ~shared
    if shared.dtype != bool or novel.dtype != bool:
        raise AssertionError("shared/novel masks must be boolean")
    if len(shared) != len(test_labels) or not np.array_equal(novel, ~shared):
        raise AssertionError("shared/novel masks are malformed")
    return shared, novel
