# -*- coding: utf-8 -*-
"""Build a reproducibility manifest with streaming SHA256 hashes."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance" / "manifest.json"


def sha256(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path):
    return str(path.relative_to(ROOT)).replace(os.sep, "/")


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def role(path):
    name = path.name
    text = rel(path)
    if text.startswith("src/") or text.startswith("tests/"):
        return "source"
    if text.startswith("results/") and path.suffix == ".json":
        return "result"
    if text.startswith("results/") and path.suffix in {".png", ".tif", ".tiff", ".eps", ".pdf"}:
        return "figure"
    if text.startswith("paper/tables/"):
        return "table"
    if text.startswith("data/raw/"):
        return "raw_data_or_download"
    if text.startswith("data/processed/"):
        return "processed_data_or_embedding"
    if text.startswith("models/"):
        return "model_or_vocabulary"
    if text.startswith("envs/"):
        return "environment_snapshot"
    if text.startswith("logs/"):
        return "run_log"
    return "documentation_or_metadata"


# Explicitly exclude caches, temporary files, logs and bytecode. Include large assets
# only when they are part of the published/reference run.
include_roots = ["src", "tests", "paper", "results", "envs", "models", "logs", "data/processed"]
include_raw = {
    "data/raw/zebrahub/zf_atlas_full_v4_release.h5ad",
    "data/raw/pijuan_sala/atlas/raw_counts.mtx",
    "data/raw/pijuan_sala/atlas/meta.csv",
    "data/raw/pijuan_sala/atlas/genes.tsv",
    "data/raw/zf2mouse.txt",
}
exclude_parts = {".cache", "__pycache__", "tmp", "torchinductor"}
files = []
for root_name in include_roots:
    root = ROOT / root_name
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = rel(path)
        if any(part in exclude_parts for part in path.parts):
            continue
        if path.name.startswith("tmp_") or path.name.endswith("_ckpt.npz") or path.name.endswith(".sqlite"):
            continue
        if relative.startswith("data/raw/") and relative not in include_raw:
            continue
        files.append(path)
for relative in include_raw:
    path = ROOT / relative
    if path.exists() and path.is_file():
        files.append(path)

unique = {rel(path): path for path in files}
artifacts = []
for relative in sorted(unique):
    path = unique[relative]
    print(f"hashing {relative} ({path.stat().st_size} bytes)", flush=True)
    artifacts.append({
        "path": relative,
        "role": role(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    })

manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_revision": git_revision(),
    "platform": {
        "system": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
    },
    "pipeline_commands": {
        "environment_check": "python src/00_check_env.py",
        "zebrafish_split": "python src/01b_split_zebrahub.py",
        "mouse_split": "python src/20_split_mouse.py",
        "ortholog_mapping": "python src/06_map_zf_to_human.py",
        "symbol_to_ensembl": "python src/12b_mygene_map.py (cached; refresh requires FORCE_MYGENE_REFRESH=1)",
        "state_embedding": "python src/09_state_embed_safetensors.py (biomni_lab env)",
        "geneformer_embedding": "python src/13_gf_embed.py (fm env)",
        "transcriptformer_embedding": "python src/13c_tf_stable_embed.py (fm env)",
        "zebrafish_evaluation": "python src/11b_hard_eval_unified.py; python src/15_eval_tf_gf.py; python src/17b_eval_tf_ctrl.py",
        "mouse_evaluation": "python src/21_p4_eval_unified.py; python src/23_p4_tf_eval.py",
        "diagnostics": "python src/11_in_dist_control.py; python src/11d_indist_tf_gf.py; python src/22_confusion.py",
        "baseline_embedding_cache": "python src/28_cache_baseline_embeddings.py (GPU, ~10 min; PCA+Harmony+scVI seed-42 persisted)",
        "umap_grid": "python src/29_umap_model_grid.py (CPU-heavy first pass; cached afterwards)",
        "supplementary_figures": "python src/30_supp_summary_figs.py",
        "validation_tier1": "python src/31_linear_probe.py; python src/32_geometry_stats.py; python src/33_novel_marker_modulescore.py",
        "validation_tier2": "python src/34_shifted_window_splits.py (resumable per window); python src/35_random_split_ceiling.py",
        "validation_figures": "python src/36_supp_tier2_figs.py (figS3, figS4)",
        "figures": "python src/16_p2_main_figure.py; python src/24_dual_dataset_figure.py; python src/27_design_schematic.py (FIG_DPI/FIG_FORMAT env)",
        "tables": "python src/25_make_tables.py",
        "tests": "python tests/test_evaluation_contracts.py; python tests/test_artifact_schema.py",
    },
    "external_dependencies": {
        "biomni": "external checkout; location via VC_BIOMNI_ROOT; record its revision before rerunning tool-layer steps",
        "state_software": "arc-state 0.11.1 (PyPI)",
        "transcriptformer_software": "transcriptformer 0.6.1 (PyPI)",
        "geneformer": "vendored under models/geneformer_repo (V1-10M weights)",
    },
    "scope": {
        "included_roots": include_roots,
        "included_raw_files": sorted(include_raw),
        "excluded_cache_or_temporary_parts": sorted(exclude_parts),
        "excluded_file_patterns": ["tmp_*", "*_ckpt.npz", "*.sqlite"],
        "note": "Hashes are generated on the reference Windows workstation; paths are repository-relative.",
    },
    "artifacts": artifacts,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved {OUT} ({len(artifacts)} artifacts)")
