# Submission checklist

The analysis and manuscript cleanup are complete. These items require author-provided information or final publisher decisions:

- [ ] Author names, affiliations, corresponding author, email, ORCID
- [ ] Funding statement (confirm whether a specific grant should be named; do not infer from the analysis workspace)
- [ ] Public code repository URL and, ideally, a tagged release/Zenodo DOI
- [ ] Confirm final journal and its abstract/figure/table requirements (Bioinformatics-format manuscript under `submission/bioinformatics/`: structured abstract 199 words, author-year citations, alphabetical OUP reference list)
- [ ] Complete reference records. PubMed-verified (2026-09-04): Zebrahub = Lange et al., *Cell* (2024), PMID 39454574 (in-text "White et al." corrected to Lange); UCE = Rosen et al., *Nature* (2026), PMID 42420460; BioMNI = Huang et al., bioRxiv, PMID 40501924 (first author corrected from "Hu, C."); Bunne virtual-cell perspective = *Cell* 187:7045–7063, PMID 39672099 (2026-09-05: PMID 39398201 turned out to be the arXiv record; the Cell record was verified via Europe PMC). Added: Stuart et al. (2019) for Seurat-v3 HVG selection; Ahlmann-Eltze & Anders (2025) for the linear-baseline distinction. All journal volume/page records filled 2026-09-05 from Europe PMC metadata (`results/_refs_meta.json`); three errors corrected (Theodoris Nature 618:616–624; Cao 566:496–502; scGPT 21:1470–1480). Still open: TranscriptFormer *Science* 2026 full author list/pages (DOI 10.1126/science.aec8514; not in PubMed); a citable preprint record for STATE/SE-600M. Citation style resolved for Bioinformatics: author-year in text + alphabetical reference list (no renumbering needed).
- [ ] Confirm the funding statement: the manuscript currently states that the analysis received no specific external funding; replace if a grant should be credited.
- [x] Presentation upgrades resolved: UMAP grid promoted to main Figure 3 (cross-species and confusion figures shifted to 4 and 5); dumbbell and OOD plots are Supplementary Figures S1-S2. Runtime-Pareto panel remains optional.
- [x] Journal-standard figure formatting applied (2026-09-04, revised same day per author preference): Arial sans-serif, 5-7 pt text at final print size, 183 mm double-column width, BOLD UPPERCASE panel letters (A-G), no in-figure titles/suptitles, Okabe-Ito colourblind-safe model palette, black majority reference lines, RdYlBu (red-green-free) AUROC colourbar, Type 42 fonts embedded in PDFs; all figures passed independent visual acceptance at print size; manuscript legends and body-text panel references kept in sync (`src/fig_style.py`).
- [x] Figure set consolidated (2026-09-04): 9 figures -> 6 (4 main + 2 supplementary). Mouse panels merged into Figure 2 E/F (removing the duplicated zebrafish panels of the former standalone cross-species figure); dumbbell + OOD merged into Fig S1; shifted-window + random-ceiling merged into Fig S2. Output file names now match manuscript numbering; retired script `24_dual_dataset_figure.py` removed.
- [x] UCE seventh model added (2026-09-05): benchmark extended six -> seven methods (UCE 4-layer CXG, human-ortholog routing on zebrafish, MGI-symbol routing on mouse). All layered conclusions replicate within existing ranges (zebrafish 0.725/0.061/0.661; mouse 0.638/0.062/0.602); manuscript, tables, figures and legends re-synced. Structural QA passed; pixel-level judge re-review of the five updated figures is pending re-review.

## Optional validation experiments (author decision; none block submission)

Tier 1 - CPU only, about a day total:
- [x] Linear-probe replication done -> `results/linear_probe_eval.json`, Supplementary Table S7; findings integrated as Results 3.8 (fine-grained failure probe-robust; TF lineage advantage is kNN-specific).
- [x] Permutation null + drift table done -> `results/geometry_stats.json` (all seven models p=0.001; drift index 1.7-16.8, UCE 4.6); cited in Figure 3 legend and Section 3.3.
- [x] Module-score validation done -> `results/novel_cluster_modulescore.csv` (4/4 scorable programs enriched in the predicted novel clusters, BH p<1e-8; intestinal skipped, 1/2 markers in HVG panel); cited in Section 3.4.

Tier 2 - GPU, hours:
- [x] Shifted-window splits + interpolation control done -> `results/shifted_window_eval.json` (E16/E19/E24/E72 + I19; foundation-model lineage placement rises to 0.87-0.96 as the window approaches test while PCA/Harmony collapse below majority; shared-cluster accuracy fails in every window, 0.009-0.024 at the widest window). Integrated as Results 3.9, Supplementary Table S8, Figure S3A-C.
- [x] Random cell-split ceiling control done -> `results/random_split_ceiling.json` (random 80/20 inflates lineage to 0.91-0.97 and shared-cluster accuracy from 0.040-0.069 to 0.59-0.89, a 10-21x inflation). Integrated as Results 3.9, Supplementary Table S9, Figure S4, abstract, and Discussion cautions.
- [x] Timed re-run of the mouse TranscriptFormer embedding done via `models/run_mouse_tf.bat` -> wall time 3,203 s for 108,857 cells (34.4 ms/cell), recorded in `results/p4_tf_run2.log` and Supplementary Table S5; regenerated embedding is bit-identical to the retained original (max |delta| = 0 over the full 108,857 x 2,048 matrix), stale checkpoint leftover removed under the metadata-mismatch guard.

Tier 3 - deferred / next paper:
- [ ] Geneformer 95M variant; additional species atlas; fine-tuned (non-frozen) foundation-model heads; GEARS/T2 perturbation benchmark.
- [ ] Decide whether the late-timepoint confusion matrix is a main figure or supplementary figure
- [x] Review all generated figures at final print dimensions (done 2026-09-04: figures are rendered at the 183 mm double-column final width and passed independent visual acceptance; re-check only if the journal requires a different column width)
- [ ] Upload the validated data/model manifest or provide an archival DOI
- [ ] State explicitly that GEARS/T2 perturbation evaluation is outside the scope of this manuscript and remains future work

## Final artifact map

- Manuscript: `paper/manuscript.md`
- Main tables: `paper/tables/main_table1_zebrafish.csv`, `main_table2_mouse.csv`
- Supplementary tables: `paper/tables/supp_tableS1_full_metrics.csv` through `supp_tableS9_random_ceiling.csv`
- Main figures (file name = manuscript number): `results/fig1_design.png` (Fig 1), `fig2_multimodel_benchmark.png` (Fig 2, six panels A-F incl. mouse E/F), `fig3_umap_grid.png` (Fig 3), `fig4_confusion_late.png` (Fig 4)
- Supplementary figures: `results/figS1_shift_diagnostics.png` (S1, panels A-C), `figS2_split_robustness.png` (S2, panels A-E). The former fig4_dual_dataset / fig5_umap_grid / fig3_confusion_late / fig6-figS4 files were consolidated or renamed on 2026-09-04.
- Validation artifacts: `results/linear_probe_eval.json`, `results/geometry_stats.json`, `results/novel_cluster_modulescore.csv`, `results/shifted_window_eval.json`, `results/random_split_ceiling.json`
- Provenance: `provenance/manifest.json`
- Reproduction guide: `REPRODUCIBILITY.md`
