# Virtual-cell developmental benchmark: reproducibility guide

This repository contains the temporal-extrapolation benchmark reported in the manuscript. The reported paper scope is T1 temporal extrapolation on Zebrahub and E-MTAB-6967. GEARS/T2 perturbation evaluation is a separate follow-up and is not represented as completed here.

## Environments

Three environments are used:

- `biomni_lab.yml`: PCA/Harmony/scVI/STATE and result aggregation.
- `fm.yml`: TranscriptFormer and Geneformer inference.
- `gears.yml`: GEARS smoke/future perturbation work; the Windows CUDA instability documented below is unresolved.

The YAML files are snapshots of the Windows `win-64` environment used for the reported results. UCE inference uses an external checkout of snap-stanford/UCE (MIT) recorded under `external/UCE`; its official figshare weights were fetched from the Hugging Face mirror `lza1/uce_hf` (figshare is bot-blocked from this network). The `.lock.txt` files preserve exact package exports. The BioMNI source tree is an external dependency; set `VC_BIOMNI_ROOT` to its checkout and record its revision before rerunning.

## Paths

All scripts derive paths from the project directory. Optional environment variables override them:

```text
VC_PROJECT_ROOT   project root (default: repository root)
VC_DATA_ROOT      data root (default: project/data)
VC_MODEL_ROOT     model root (default: project/models)
VC_RESULTS_ROOT   result root (default: project/results)
VC_LOG_ROOT       log root (default: project/logs)
VC_BIOMNI_ROOT    external BioMNI checkout
BIOMNI_DATA_PATH  BioMNI data lake/cache
STATE_MODEL_DIR   STATE model directory
TF_MODEL_DIR      TranscriptFormer model directory
GENEFORMER_REPO   Geneformer repository directory
GENEFORMER_MODEL_DIR  Geneformer checkpoint directory
PYTHON_BIN        Python executable used by models/run_mouse_tf.bat
```

No script should require a drive letter or a user-specific Conda path. A local proxy may be supplied through the normal `HTTP_PROXY`/`HTTPS_PROXY` variables; no proxy is enabled by default.

## Recommended execution order

Run from the project root, after activating the appropriate environment:

```bash
python src/00_check_env.py
python src/01b_split_zebrahub.py
python src/20_split_mouse.py
python src/06_map_zf_to_human.py
python src/12b_mygene_map.py          # uses the saved mapping cache by default
python src/07b_prepare_state_input.py
python src/12_tf_input.py
python src/13_gf_embed.py             # fm environment
python src/13c_tf_stable_embed.py     # fm environment; GPU, resumable
python src/09_state_embed_safetensors.py  # biomni_lab/state environment
python src/11b_hard_eval_unified.py
python src/21_p4_eval_unified.py
python src/23_p4_tf_eval.py
python src/22_confusion.py                # Fig 4 confusion
python src/28_cache_baseline_embeddings.py   # GPU ~10 min; persists PCA/Harmony/scVI
python src/29_umap_model_grid.py             # builds UMAP cache, then Fig 3 grid
python src/30_supp_summary_figs.py           # Fig S1 shift diagnostics (dumbbell + OOD)
python src/31_linear_probe.py                # Tier-1: linear-probe replication (CPU)
python src/32_geometry_stats.py              # Tier-1: permutation null + drift index
python src/33_novel_marker_modulescore.py    # Tier-1: AddModuleScore-style validation
python src/34_shifted_window_splits.py       # Tier-2: shifted windows + interpolation control (GPU for scVI)
python src/35_random_split_ceiling.py        # Tier-2: random 80/20 ceiling control (GPU for scVI)
python src/36_supp_tier2_figs.py             # Fig S2 split robustness (windows + random)
python src/40_uce_input_mouse.py            # UCE mouse input: ENSMUSG -> MGI symbols (mygene cache)
python src/41_uce_embed.py                  # UCE 4-layer embedding (fm env; UCE_SPECIES=zebra|mouse, UCE_BATCH)
python src/42_uce_eval.py                   # UCE zebrafish unified evaluation
python src/43_uce_mouse_eval.py             # UCE mouse evaluation + in-dist + OOD
python src/44_uce_diag.py                   # UCE zebrafish in-dist + OOD diagnostics
python src/45_uce_addendum_splits.py        # UCE into shifted-window + random-split JSONs
python src/16_p2_main_figure.py              # Fig 2 six-panel benchmark
python src/27_design_schematic.py
python src/25_make_tables.py
python tests/test_evaluation_contracts.py
python src/26_build_manifest.py
```

The existing processed h5ad files and result JSONs are retained as the reference run. Model inference is expensive; figure/table generation and contract tests do not require re-embedding.

## Success criteria

- `tests/test_evaluation_contracts.py` passes.
- Mouse contract: train/test = 20,308/88,549 cells; 11 train clusters; 22 test clusters; shared/novel = 61,784/26,765 cells.
- Zebrafish contract: train/test = 18,109/102,335 cells; 29 train/shared clusters; 22 novel clusters.
- Result JSONs contain nonempty `lineage_all`, `cluster_shared`, and `lineage_on_novel_clusters` records with expected sample counts.
- Generated tables and figures complete without warnings that indicate truncation or missing labels.
- `provenance/manifest.json` is regenerated after any input, model, code, or result change.

## Platform notes

- Reported runs used Windows 11, Python 3.11, PyTorch 2.11 with CUDA 12.8, and an RTX 5080 16 GB GPU.
- TranscriptFormer uses `13c_tf_stable_embed.py`, which replaces the unavailable Windows flex-attention path with a numerically equivalent standard-attention implementation. The CPU probe equivalence check must remain >=0.99 cosine.
- TranscriptFormer checkpoints are accompanied by a JSON metadata file. A mismatch in input hash, model hash, script hash, chunk size, token budget, or cell count aborts resume.
- GEARS 0.1.2 currently has a Windows CUDA failure during validation in this project. Use the `gears` snapshot for smoke testing or run the formal T2 work in a supported Linux/WSL environment after separately validating the environment.
- The released STATE YAML contains training-time paths from the upstream checkpoint. The inference scripts use local safetensors and protein embeddings; do not treat upstream training paths as local data dependencies.

## Data and model provenance

`provenance/manifest.json` records relative paths, sizes, SHA256 hashes, roles, and generation notes for the key raw data, model weights, processed inputs, results, tables, figures, run logs (`logs/runs/`), source files. Temporary downloads, HF caches, TorchInductor caches, checkpoint scratch files (`*_ckpt.npz`), and `*.sqlite` are excluded.

The project is a git repository; `source_revision` in the manifest anchors the manifest to the committed code state. Because the manifest embeds the current timestamp it is rebuilt after committing code:

```bash
git add -A && git commit -m "..."   # commit code/state first
python src/26_build_manifest.py     # rebuild with source_revision = HEAD
git add provenance/manifest.json && git commit -m "refresh manifest"
```
