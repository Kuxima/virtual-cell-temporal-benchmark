# Virtual-cell temporal-extrapolation benchmark

Source code and environment snapshots for a benchmark of seven embedding
methods (three generations of single-cell representation learning) evaluated
on two embryonic atlases split along developmental time. Result files,
figures and tables are distributed with the manuscript and its supplementary
material, and will be archived on Zenodo upon publication.

## Key models

| Model | Parameters | Frozen weights |
|---|---|---|
| STATE SE-600M | ~600M | https://huggingface.co/arcinstitute/SE-600M |
| UCE 4-layer CXG | ~4M transformer + frozen ESM2 protein-embedding tokens | Hugging Face mirror `lza1/uce_hf` (`4layer_model.torch` + per-species token files; official figshare is bot-blocked from some networks) |
| Geneformer V1-10M | ~10M | Hugging Face `ctheodoris/Geneformer` |
| TranscriptFormer tf-metazoa | ~1.2B (fp32, 8.6 GB) | official CZ Biohub release |
| scVI / Harmony / PCA-50 | trained from scratch per split | no external weights |

## Datasets

- Zebrahub zebrafish atlas (120,444 cells): https://zebrahub.sf.czbiohub.org/
  (release v4 `zf_atlas_full_v4_release.h5ad`)
- Mouse gastrulation (108,857 cells): ArrayExpress **E-MTAB-6967**,
  https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6967/

## Environments

```bash
conda env create -f envs/biomni_lab.yml   # preprocessing, scVI, evaluation
conda env create -f envs/fm.yml           # STATE / UCE / Geneformer / TranscriptFormer embedding
```

Exact package locks are in `envs/*.lock.txt`. An external checkout of
snap-stanford/UCE (MIT) is used for UCE inference (`external/UCE`, not
included; see `src/41_uce_embed.py` for the expected layout and env vars).

## Code layout

`src/` is a numbered pipeline: 00-19 zebrafish preprocessing and baselines,
20-23 mouse, 25-36 tables and figures, 28-33 diagnostics and controls,
40-45 the UCE arm. `tests/` holds evaluation-contract regression tests
(run with `python -m pytest tests -q` after installing pytest).

## License

MIT (see LICENSE). Model weights and atlases follow their upstream licences.
