# -*- coding: utf-8 -*-
"""25 论文表格生成：主表 1/2 + 补充表 S1-S9（CSV，paper/tables/）。
数据全部来自 results/*.json，不手工录入任何指标。"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from common import RESULTS

TAB = RESULTS.parent / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def j(name):
    return json.loads((RESULTS / name).read_text())


uni = j("t1_unified_eval.json")
sur = j("t1_unfrozen_surgery.json")
p2 = j("t1_p2_eval.json")
tfc = j("t1_tf_ctrl_eval.json")
uce = j("t1_uce_eval.json")
p4c = j("p4_classical_eval.json")
p4t = j("p4_tf_eval.json")
p4u = j("p4_uce_eval.json")
ctl = j("in_dist_control.json")
ind2 = j("indist_tf_gf.json")
auroc = j("ood_auroc_models.json")
udiag = j("uce_diag.json")


def row(name, adapt, genes, d, ci_key="bootstrap95_by_fish"):
    ci = d.get(ci_key, {})
    lin, clu, nov = d["lineage_all"], d["cluster_shared"], d["lineage_on_novel_clusters"]
    fmt = lambda m, key: f"{m['accuracy']:.3f} [{ci[key][0]:.3f},{ci[key][1]:.3f}]" if key in ci else f"{m['accuracy']:.3f}"
    return {
        "Model": name, "Adaptation": adapt, "Input genes": genes, "Summary statistic": "single run",
        "Lineage acc (95% CI)": fmt(lin, "lineage_all"),
        "Cluster acc, shared (95% CI)": fmt(clu, "cluster_shared"),
        "Lineage acc on novel clusters (95% CI)": fmt(nov, "lineage_novel"),
        "Balanced acc (lineage)": f"{lin['balanced_accuracy']:.3f}",
        "Macro F1 (lineage)": f"{lin['macro_f1']:.3f}",
    }


# ---------- 主表 1：斑马鱼 ----------
t1 = pd.DataFrame([
    {"Model": "Majority class", "Adaptation": "-", "Input genes": "-", "Summary statistic": "fixed baseline",
     "Lineage acc (95% CI)": f"{uni['majority_lineage_all']:.3f}",
     "Cluster acc, shared (95% CI)": f"{uni['majority_cluster_shared_subset']:.3f}",
     "Lineage acc on novel clusters (95% CI)": "-",
     "Balanced acc (lineage)": "-", "Macro F1 (lineage)": "-"},
    row("PCA-50", "frozen", "2,581", uni["PCA_frozen"]),
    row("Harmony", "transductive", "2,581", uni["Harmony_adapted"]),
    {"Model": "scVI (frozen)", "Adaptation": "frozen", "Input genes": "2,581", "Summary statistic": "3-seed mean ± SD",
     "Lineage acc (95% CI)": f"{uni['scVI_frozen_summary']['lineage_all']['mean']:.3f} ± {uni['scVI_frozen_summary']['lineage_all']['std']:.3f}",
     "Cluster acc, shared (95% CI)": f"{uni['scVI_frozen_summary']['cluster_shared']['mean']:.3f} ± {uni['scVI_frozen_summary']['cluster_shared']['std']:.3f}",
     "Lineage acc on novel clusters (95% CI)": f"{uni['scVI_frozen_summary']['lineage_on_novel_clusters']['mean']:.3f} ± {uni['scVI_frozen_summary']['lineage_on_novel_clusters']['std']:.3f}",
     "Balanced acc (lineage)": "NA",
     "Macro F1 (lineage)": "NA"},
    {"Model": "scVI (surgery, unfrozen)", "Adaptation": "query-training", "Input genes": "2,581", "Summary statistic": "3-seed mean ± SD",
     "Lineage acc (95% CI)": f"{sur['summary']['lineage_all']['mean']:.3f} ± {sur['summary']['lineage_all']['std']:.3f}",
     "Cluster acc, shared (95% CI)": f"{sur['summary']['cluster_shared']['mean']:.3f} ± {sur['summary']['cluster_shared']['std']:.3f}",
     "Lineage acc on novel clusters (95% CI)": f"{sur['summary']['lineage_on_novel_clusters']['mean']:.3f} ± {sur['summary']['lineage_on_novel_clusters']['std']:.3f}",
     "Balanced acc (lineage)": "NA",
     "Macro F1 (lineage)": "NA"},
    row("STATE SE-600M", "frozen", "2,058 / 2,581", uni["STATE_frozen"]),
    row("UCE (4-layer)", "frozen", "2,468 / 2,581", uce["UCE_frozen"]),
    row("Geneformer V1-10M", "frozen", "2,454 ENSG", p2["Geneformer_frozen"]),
    row("TranscriptFormer tf-metazoa", "frozen", "3,816 ENSDARG", p2["TranscriptFormer_frozen"]),
    row("TranscriptFormer (gene-set control)", "frozen", "2,994 ENSDARG", tfc["TF_ctrl"]),
])
t1.to_csv(TAB / "main_table1_zebrafish.csv", index=False)

# ---------- 主表 2：小鼠 ----------
t2 = pd.DataFrame([
    {"Model": "Majority class", "Adaptation": "-", "Input genes": "-", "Summary statistic": "fixed baseline",
     "Lineage acc (95% CI)": f"{p4c['majority_lineage']:.3f}",
     "Cluster acc, shared (95% CI)": f"{p4c['majority_cluster_shared']:.3f}",
     "Lineage acc on novel clusters (95% CI)": "-",
     "Balanced acc (lineage)": "-", "Macro F1 (lineage)": "-"},
    row("PCA-50", "frozen", "4,000 ENSMUSG", p4c["PCA_frozen"]),
    row("Harmony", "transductive", "4,000 ENSMUSG", p4c["Harmony_adapted"]),
    {"Model": "scVI (query-unfrozen)", "Adaptation": "query-training", "Input genes": "4,000 ENSMUSG", "Summary statistic": "3-seed mean ± SD",
     "Lineage acc (95% CI)": f"{p4c['scVI_summary']['lineage_all']['mean']:.3f} ± {p4c['scVI_summary']['lineage_all']['std']:.3f}",
     "Cluster acc, shared (95% CI)": f"{p4c['scVI_summary']['cluster_shared']['mean']:.3f} ± {p4c['scVI_summary']['cluster_shared']['std']:.3f}",
     "Lineage acc on novel clusters (95% CI)": f"{p4c['scVI_summary']['lineage_on_novel_clusters']['mean']:.3f} ± {p4c['scVI_summary']['lineage_on_novel_clusters']['std']:.3f}",
     "Balanced acc (lineage)": "NA",
     "Macro F1 (lineage)": "NA"},
    row("UCE (4-layer)", "frozen", "3,268 / 4,000 ENSMUSG (via MGI symbols)", p4u["UCE_mouse"]),
    row("TranscriptFormer tf-metazoa", "frozen", "3,687 / 4,000 ENSMUSG", p4t["TF_mouse"]),
])
t2.to_csv(TAB / "main_table2_mouse.csv", index=False)

# ---------- S1 full metrics (both datasets) ----------
def metric_row(dataset, model, adaptation, statistic, d):
    return {
        "Dataset": dataset, "Model": model, "Adaptation": adaptation, "Summary statistic": statistic,
        "lineage_acc": d["lineage_all"]["accuracy"], "lineage_bacc": d["lineage_all"]["balanced_accuracy"],
        "lineage_macroF1": d["lineage_all"]["macro_f1"],
        "cluster_acc": d["cluster_shared"]["accuracy"], "cluster_bacc": d["cluster_shared"]["balanced_accuracy"],
        "cluster_macroF1": d["cluster_shared"]["macro_f1"],
        "novel_lineage_acc": d["lineage_on_novel_clusters"]["accuracy"],
        "novel_lineage_via_cluster": d["lineage_via_predicted_cluster_on_novel"]["accuracy"],
    }

def aggregate_row(dataset, model, adaptation, summary, reference):
    return {
        "Dataset": dataset, "Model": model, "Adaptation": adaptation,
        "Summary statistic": "3-seed mean (SD reported in main table)",
        "lineage_acc": summary["lineage_all"]["mean"], "lineage_bacc": "NA", "lineage_macroF1": "NA",
        "cluster_acc": summary["cluster_shared"]["mean"], "cluster_bacc": "NA", "cluster_macroF1": "NA",
        "novel_lineage_acc": summary["lineage_on_novel_clusters"]["mean"],
        "novel_lineage_via_cluster": reference["lineage_via_predicted_cluster_on_novel"]["accuracy"],
    }


rows = [
    metric_row("ZF", "PCA-50", "frozen", "single run", uni["PCA_frozen"]),
    metric_row("ZF", "Harmony", "transductive", "single run", uni["Harmony_adapted"]),
    metric_row("ZF", "scVI", "frozen", "seed 42", uni["scVI_frozen_s42"]),
    aggregate_row("ZF", "scVI", "frozen", uni["scVI_frozen_summary"], uni["scVI_frozen_s42"]),
    metric_row("ZF", "scVI", "fully-unfrozen query-training", "seed 42", sur["s42"]),
    aggregate_row("ZF", "scVI", "fully-unfrozen query-training", sur["summary"], sur["s42"]),
    metric_row("ZF", "STATE SE-600M", "frozen", "single run", uni["STATE_frozen"]),
    metric_row("ZF", "UCE (4-layer)", "frozen", "single run", uce["UCE_frozen"]),
    metric_row("ZF", "Geneformer V1-10M", "frozen", "single run", p2["Geneformer_frozen"]),
    metric_row("ZF", "TranscriptFormer tf-metazoa", "frozen", "single run", p2["TranscriptFormer_frozen"]),
    metric_row("ZF", "TranscriptFormer tf-metazoa (gene-set control)", "frozen", "single run", tfc["TF_ctrl"]),
    metric_row("MS", "PCA-50", "frozen", "single run", p4c["PCA_frozen"]),
    metric_row("MS", "Harmony", "transductive", "single run", p4c["Harmony_adapted"]),
    metric_row("MS", "scVI", "fully-unfrozen query-training", "seed 42", p4c["scVI_s42"]),
    metric_row("MS", "UCE (4-layer)", "frozen", "single run", p4u["UCE_mouse"]),
    metric_row("MS", "TranscriptFormer tf-metazoa", "frozen", "single run", p4t["TF_mouse"]),
]
pd.DataFrame(rows).to_csv(TAB / "supp_tableS1_full_metrics.csv", index=False)

# ---------- S2 in-distribution controls (long format) ----------
tps = ["10hpf", "12hpf", "14hpf", "16hpf", "19hpf", "24hpf", "2dpf", "3dpf", "5dpf", "10dpf"]
rows = []
for t in tps:
    rows.extend([
        {"Dataset": "ZF", "timepoint": t, "Model": "STATE", "cluster_accuracy": ctl["in_dist_control"][t]["cluster_state_acc"], "n_cells_total": ctl["in_dist_control"][t]["n"]},
        {"Dataset": "ZF", "timepoint": t, "Model": "PCA", "cluster_accuracy": ctl["in_dist_control"][t]["cluster_pca_acc"], "n_cells_total": ctl["in_dist_control"][t]["n"]},
        {"Dataset": "ZF", "timepoint": t, "Model": "TranscriptFormer", "cluster_accuracy": ind2["TF"][t], "n_cells_total": ctl["in_dist_control"][t]["n"]},
        {"Dataset": "ZF", "timepoint": t, "Model": "Geneformer", "cluster_accuracy": ind2["Geneformer"][t], "n_cells_total": ctl["in_dist_control"][t]["n"]},
        {"Dataset": "ZF", "timepoint": t, "Model": "UCE", "cluster_accuracy": udiag["UCE"]["in_dist"][t], "n_cells_total": ctl["in_dist_control"][t]["n"]},
    ])
mouse_tps = sorted(p4t["in_dist_control"])
for t in mouse_tps:
    rows.append({"Dataset": "MS", "timepoint": t, "Model": "TranscriptFormer", "cluster_accuracy": p4t["in_dist_control"][t], "n_cells_total": None})
    rows.append({"Dataset": "MS", "timepoint": t, "Model": "UCE", "cluster_accuracy": p4u["in_dist_control"][t], "n_cells_total": None})
pd.DataFrame(rows, columns=["Dataset", "timepoint", "Model", "cluster_accuracy", "n_cells_total"]).to_csv(TAB / "supp_tableS2_in_dist_control.csv", index=False)

# ---------- S3 OOD AUROC ----------
rows = []
for m, d in auroc.items():
    r = {"Dataset": "ZF", "Model": m, "Global AUROC": d["global"]}
    r.update({f"AUROC@{tp}": v for tp, v in d["per_timepoint"].items()})
    rows.append(r)
r = {"Dataset": "MS", "Model": "TranscriptFormer", "Global AUROC": p4t["ood_auroc"]["global"]}
r.update({f"AUROC@{tp}": v for tp, v in p4t["ood_auroc"].items() if tp != "global"})
rows.append(r)
r = {"Dataset": "ZF", "Model": "UCE", "Global AUROC": udiag["UCE"]["ood"]["global"]}
r.update({f"AUROC@{tp}": v for tp, v in udiag["UCE"]["ood"].items() if tp != "global"})
rows.append(r)
r = {"Dataset": "MS", "Model": "UCE", "Global AUROC": p4u["ood_auroc"]["global"]}
r.update({f"AUROC@{tp}": v for tp, v in p4u["ood_auroc"].items() if tp != "global"})
rows.append(r)
s3 = pd.DataFrame(rows).fillna("NA")
s3.to_csv(TAB / "supp_tableS3_ood_auroc.csv", index=False)

# ---------- S4 k sensitivity ----------
rows = []
for name, d in (("ZF PCA", uni["PCA_frozen"]), ("ZF Harmony", uni["Harmony_adapted"]),
                ("ZF scVI(frozen,s42)", uni["scVI_frozen_s42"]), ("ZF scVI(unfrozen,s42)", sur["s42"]),
                ("ZF STATE", uni["STATE_frozen"]), ("ZF UCE", uce["UCE_frozen"]),
                ("ZF Geneformer", p2["Geneformer_frozen"]),
                ("ZF TF", p2["TranscriptFormer_frozen"]), ("MS PCA", p4c["PCA_frozen"]),
                ("MS scVI(s42)", p4c["scVI_s42"]), ("MS UCE", p4u["UCE_mouse"]), ("MS TF", p4t["TF_mouse"])):
    rows.append({"Model": name, "k=5 lineage": d["k_sensitivity"]["lineage"]["5"],
                 "k=15 lineage": d["k_sensitivity"]["lineage"]["15"], "k=30 lineage": d["k_sensitivity"]["lineage"]["30"],
                 "k=5 cluster": d["k_sensitivity"]["cluster_shared"]["5"],
                 "k=15 cluster": d["k_sensitivity"]["cluster_shared"]["15"],
                 "k=30 cluster": d["k_sensitivity"]["cluster_shared"]["30"], "Statistic": "seed 42 or single run"})
rows.extend([
    {"Model": "ZF majority", "k=5 lineage": uni["majority_lineage_all"], "k=15 lineage": uni["majority_lineage_all"], "k=30 lineage": uni["majority_lineage_all"], "k=5 cluster": uni["majority_cluster_shared_subset"], "k=15 cluster": uni["majority_cluster_shared_subset"], "k=30 cluster": uni["majority_cluster_shared_subset"], "Statistic": "fixed baseline"},
    {"Model": "MS majority", "k=5 lineage": p4c["majority_lineage"], "k=15 lineage": p4c["majority_lineage"], "k=30 lineage": p4c["majority_lineage"], "k=5 cluster": p4c["majority_cluster_shared"], "k=15 cluster": p4c["majority_cluster_shared"], "k=30 cluster": p4c["majority_cluster_shared"], "Statistic": "fixed baseline"},
])
pd.DataFrame(rows).to_csv(TAB / "supp_tableS4_k_sensitivity.csv", index=False)

# ---------- S5 resources ----------
import re

tf_mouse_time = "not recorded"
_log = RESULTS / "p4_tf_run2.log"
if _log.exists():
    _m = re.findall(r"总耗时 (\d+)s", _log.read_text(encoding="utf-8", errors="ignore"))
    if _m:
        _s = int(_m[-1])
        tf_mouse_time = f"{_s:,} s for 108,857 cells ({_s * 1000 / 108857:.1f} ms/cell; timed rerun)"

s5 = pd.DataFrame([
    {"Model": "PCA-50", "Parameters": "-", "Inference device": "CPU", "Peak GPU mem": "not applicable", "Wall time": "part of 14-min unified classical run (11b)"},
    {"Model": "Harmony", "Parameters": "-", "Inference device": "CPU", "Peak GPU mem": "not applicable", "Wall time": "part of 14-min unified classical run (11b)"},
    {"Model": "scVI (60ep train + 30ep query, x3 seeds, frozen+surgery)", "Parameters": "~1.5M", "Inference device": "GPU", "Peak GPU mem": "<2 GB", "Wall time": "unified run total 857 s (classical + scVI)"},
    {"Model": "STATE SE-600M (frozen)", "Parameters": "~600M", "Inference device": "GPU (batch 8)", "Peak GPU mem": "<=16 GB (RTX 5080)", "Wall time": "not recorded"},
    {"Model": "UCE 4-layer CXG (frozen, zebrafish)", "Parameters": "~4M transformer + frozen ESM2 protein-embedding tokens (143,574 x 5,120)", "Inference device": "GPU (batch 16)", "Peak GPU mem": "<=16 GB", "Wall time": "1,791 s, 120,444 cells (14.9 ms/cell)"},
    {"Model": "UCE 4-layer CXG (frozen, mouse)", "Parameters": "~4M transformer", "Inference device": "GPU (batch 16)", "Peak GPU mem": "<=16 GB", "Wall time": "2,291 s, 108,857 cells (21.1 ms/cell)"},
    {"Model": "Geneformer V1-10M (frozen)", "Parameters": "~10M", "Inference device": "GPU", "Peak GPU mem": "<4 GB", "Wall time": "455 s forward, 120,444 cells (3.78 ms/cell)"},
    {"Model": "TranscriptFormer tf-metazoa (zebrafish)", "Parameters": "~1.2B (fp32 checkpoint 8.6 GB)", "Inference device": "GPU (chunked, length-sorted)", "Peak GPU mem": "<=16 GB", "Wall time": "4,748 s, 120,444 cells (39.4 ms/cell)"},
    {"Model": "TranscriptFormer tf-metazoa (mouse)", "Parameters": "~1.2B", "Inference device": "GPU (chunked, length-sorted)", "Peak GPU mem": "<=16 GB", "Wall time": tf_mouse_time},
], columns=["Model", "Parameters", "Inference device", "Peak GPU mem", "Wall time"])
s5.to_csv(TAB / "supp_tableS5_resources.csv", index=False)

# ---------- S7 linear probe vs kNN (zebrafish) ----------
probe = json.loads((RESULTS / "linear_probe_eval.json").read_text())
knn_ref = {
    "PCA-50": (uni["PCA_frozen"], None),
    "Harmony": (uni["Harmony_adapted"], None),
    "scVI": (uni["scVI_frozen_s42"], uni["scVI_frozen_summary"]),
    "STATE SE-600M": (uni["STATE_frozen"], None),
    "UCE": (uce["UCE_frozen"], None),
    "Geneformer V1": (p2["Geneformer_frozen"], None),
    "TranscriptFormer": (p2["TranscriptFormer_frozen"], None),
}
rows = []
for model, probe_d in probe["models"].items():
    knn, summary = knn_ref[model.replace(" (seed 42)", "")]
    knn_lin = summary["lineage_all"]["mean"] if summary else knn["lineage_all"]["accuracy"]
    knn_clu = summary["cluster_shared"]["mean"] if summary else knn["cluster_shared"]["accuracy"]
    knn_nov = summary["lineage_on_novel_clusters"]["mean"] if summary else knn["lineage_on_novel_clusters"]["accuracy"]
    display = "scVI (frozen)" if model == "scVI (seed 42)" else model
    rows.append({
        "Model": display,
        "kNN lineage": round(knn_lin, 3), "Linear-probe lineage": round(probe_d["lineage_all"]["accuracy"], 3),
        "kNN cluster (shared)": round(knn_clu, 3), "Linear-probe cluster (shared)": round(probe_d["cluster_shared"]["accuracy"], 3),
        "kNN novel-lineage": round(knn_nov, 3), "Linear-probe novel-lineage": round(probe_d["lineage_on_novel_clusters"]["accuracy"], 3),
    })
pd.DataFrame(rows).to_csv(TAB / "supp_tableS7_linear_probe.csv", index=False)

# ---------- S8 shifted-window splits (zebrafish) ----------
sw = json.loads((RESULTS / "shifted_window_eval.json").read_text())
rows = []
for wname, e in sw["configs"].items():
    kind = "interpolation control" if e.get("interpolation_control") else "extrapolation"
    rows.append({
        "Window": wname, "Type": kind,
        "Train timepoints": "+".join(e["train_tps"]),
        "Test timepoints": "+".join(e["test_tps"]),
        "n_train": e["n_train"], "n_test": e["n_test"],
        "Shared clusters": e["n_shared_clusters"],
        "Majority lineage": round(e["majority_lineage"], 3),
        "Majority cluster (shared)": round(e["majority_cluster_shared"], 3),
    })
    for m, r in e["models"].items():
        rows.append({
            "Window": wname, "Type": f"model: {m}",
            "Train timepoints": "", "Test timepoints": "",
            "n_train": "", "n_test": "", "Shared clusters": "",
            "Majority lineage": "",
            "Majority cluster (shared)": "",
            "Lineage acc": round(r["lineage_all"]["accuracy"], 4),
            "Cluster acc (shared)": round(r["cluster_shared"]["accuracy"], 4),
            "Lineage acc @19hpf": (round(r["accuracy_lineage_per_timepoint"]["19hpf"], 4)
                                   if "19hpf" in r.get("accuracy_lineage_per_timepoint", {}) else ""),
            "Lineage 95% CI": (f"[{r['bootstrap95_by_fish']['lineage_all'][0]:.3f},"
                               f"{r['bootstrap95_by_fish']['lineage_all'][1]:.3f}]"),
        })
pd.DataFrame(rows).to_csv(TAB / "supp_tableS8_shifted_window.csv", index=False)

# ---------- S9 random-split ceiling (zebrafish) ----------
rc = json.loads((RESULTS / "random_split_ceiling.json").read_text())
rows = []
for m, c in rc["comparison"].items():
    rows.append({
        "Model": m,
        "Time-split lineage acc": round(c["lineage_time"], 3),
        "Random-split lineage acc": round(c["lineage_random"], 3),
        "Lineage delta (random - time)": round(c["lineage_delta"], 3),
        "Time-split cluster acc (shared)": round(c["cluster_time"], 3),
        "Random-split cluster acc (shared)": round(c["cluster_random"], 3),
        "Cluster delta (random - time)": round(c["cluster_delta"], 3),
    })
rows.append({"Model": "Majority class",
             "Time-split lineage acc": round(uni["majority_lineage_all"], 3),
             "Random-split lineage acc": round(rc["majority_lineage"], 3),
             "Lineage delta (random - time)": round(rc["majority_lineage"] - uni["majority_lineage_all"], 3),
             "Time-split cluster acc (shared)": round(uni["majority_cluster_shared_subset"], 3),
             "Random-split cluster acc (shared)": round(rc["majority_cluster_shared"], 3),
             "Cluster delta (random - time)": round(rc["majority_cluster_shared"] - uni["majority_cluster_shared_subset"], 3)})
pd.DataFrame(rows).to_csv(TAB / "supp_tableS9_random_ceiling.csv", index=False)

# ---------- S6 novel cluster markers ----------
s6 = pd.read_csv(RESULTS / "novel_cluster_markers.csv", encoding="utf-8-sig")
if s6.columns[0].startswith("Unnamed") or not str(s6.columns[0]).strip():
    s6 = s6.rename(columns={s6.columns[0]: "cluster_id"})
s6.to_csv(TAB / "supp_tableS6_novel_cluster_markers.csv", index=False)

print("saved tables to", TAB)
for f in sorted(TAB.glob("*.csv")):
    print(" -", f.name)
