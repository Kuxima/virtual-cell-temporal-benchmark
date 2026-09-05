# -*- coding: utf-8 -*-
"""11b 统一基因集硬评测（P1 protocol revision, 2026-08）。

修复的四个问题：
  P1 基因集混淆 → 全部模型统一在 2581 人同源基因上评测
  P3 适配机制不对称 → frozen（零前向适应）与 adapted（测试时无标签适应）分开报告
  P4 统计严谨性 → scVI 3 种子 mean±std；按鱼 bootstrap 95% CI（1000 次）
  P8 k 敏感性 → k ∈ {5,15,30}
  P2 基线错位 → cluster majority 基线在 shared 子集上计算

模型（同一数据、同一 kNN 协议：cosine，distance-weighted）：
  PCA-50 (frozen)          — fit 于 train，测试零适应
  Harmony  (transductive)  — 嵌入时见到全部测试细胞
  scVI     (frozen)        — 训练后编码器直接前向测试集，×3 种子
  scVI     (surgery)       — load_query_data 无标签训练 30ep，×3 种子
  STATE    (frozen)        — 复用已有 X_state（本就是 2581 ortholog 基因）
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from harmony import harmonize
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common import DATA_PROC, RESULTS, SEED

torch.set_float32_matmul_precision("high")
KS = (5, 15, 30)
TOPK = 30
K_MAIN = 15
SEEDS = (42, 0, 1)
N_BOOT = 1000


# ---------- 工具 ----------

def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def topk_neighbors(Xtr, Xte, topk=TOPK, chunk=4096):
    """cosine top-k 邻居。分块 matmul，返回 (nb_idx, nb_sim)。"""
    Xr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Xq = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-12)
    n = len(Xq)
    nb_idx = np.empty((n, topk), dtype=np.int32)
    nb_sim = np.empty((n, topk), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = Xq[s:e] @ Xr.T
        idx = np.argpartition(-sim, kth=topk - 1, axis=1)[:, :topk]
        nb_idx[s:e] = idx
        nb_sim[s:e] = np.take_along_axis(sim, idx, axis=1)
    return nb_idx, nb_sim


def vote_predict(nb_idx, nb_sim, ytr_codes, k, n_classes):
    """distance-weighted 投票（与 sklearn KNeighborsClassifier(weights='distance') 等价）。"""
    w = 1.0 / (1.0 - nb_sim[:, :k] + 1e-9)
    cls = ytr_codes[nb_idx[:, :k]]
    votes = np.zeros((len(cls), n_classes), dtype=np.float64)
    for j in range(n_classes):
        votes[:, j] = np.einsum("ij,ij->i", (cls == j).astype(np.float64), w)
    return votes.argmax(1)


def pack(y_true, y_pred, mask=None):
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
    }


def eval_embedding(name, adaptation, Xtr, Xte, ytr_lin, ytr_clu, ctx):
    """统一评测：k=15 主指标 + k 敏感性 + 按鱼 bootstrap CI。"""
    nb_idx, nb_sim = topk_neighbors(Xtr.astype(np.float32), Xte.astype(np.float32))
    pred_lin = {k: vote_predict(nb_idx, nb_sim, ctx["lin_codes_tr"], k, len(ctx["lin_classes"])) for k in KS}
    pred_clu = {k: vote_predict(nb_idx, nb_sim, ctx["clu_codes_tr"], k, len(ctx["clu_classes"])) for k in KS}

    yte_lin, yte_clu = ctx["yte_lin"], ctx["yte_clu"]
    shared, novel = ctx["shared"], ctx["novel"]
    clu2lin_code = ctx["clu2lin_code"]
    # 预测 cluster → 该 cluster 在 train 中的多数 lineage
    pred_lin_via_clu = np.array([clu2lin_code[c] for c in pred_clu[K_MAIN]])

    tp = ctx["tp_te"]
    per_tp = {t: float(accuracy_score(yte_lin[tp == t], pred_lin[K_MAIN][tp == t])) for t in sorted(set(tp))}

    # k 敏感性
    k_sens = {
        "lineage": {int(k): float(accuracy_score(yte_lin, pred_lin[k])) for k in KS},
        "cluster_shared": {int(k): float(accuracy_score(yte_clu[shared], pred_clu[k][shared])) for k in KS},
    }

    # 按鱼 bootstrap（k=15）
    boot = {}
    fish_codes = ctx["fish_te_codes"]
    n_fish = int(fish_codes.max()) + 1
    rng = np.random.default_rng(SEED)
    for metric, correct, mask in (
        ("lineage_all", pred_lin[K_MAIN] == ctx["lin_codes_te"], np.ones(len(yte_lin), dtype=bool)),
        ("cluster_shared", pred_clu[K_MAIN] == ctx["yte_clu"], shared),
        ("lineage_novel", pred_lin[K_MAIN] == ctx["lin_codes_te"], novel),
    ):
        n_by = np.bincount(fish_codes[mask], minlength=n_fish)
        ok_by = np.bincount(fish_codes[mask][correct[mask]], weights=np.ones(int(correct[mask].sum())), minlength=n_fish)
        draws = rng.integers(0, n_fish, size=(N_BOOT, n_fish))
        accs = ok_by[draws].sum(1) / np.maximum(n_by[draws].sum(1), 1)
        boot[metric] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]

    out = {
        "adaptation": adaptation,
        "lineage_all": pack(yte_lin, pred_lin[K_MAIN]),
        "cluster_shared": pack(yte_clu, pred_clu[K_MAIN], shared),
        "lineage_on_novel_clusters": pack(yte_lin, pred_lin[K_MAIN], novel),
        "lineage_via_predicted_cluster_on_novel": pack(ctx["lin_codes_te"], pred_lin_via_clu, novel),
        "accuracy_lineage_per_timepoint": per_tp,
        "k_sensitivity": k_sens,
        "bootstrap95_by_fish": boot,
    }
    assert out["lineage_all"]["n"] == len(yte_lin)
    assert out["cluster_shared"]["n"] == int(shared.sum())
    assert out["lineage_on_novel_clusters"]["n"] == int(novel.sum())
    assert int(shared.sum() + novel.sum()) == len(yte_lin)
    nov_acc = out["lineage_on_novel_clusters"].get("accuracy")
    nov_str = f"{nov_acc:.4f}" if nov_acc is not None else "n/a (no novel cells)"
    print(f"[{name}] lineage {out['lineage_all']['accuracy']:.4f} "
          f"cluster_shared {out['cluster_shared']['accuracy']:.4f} "
          f"novel_lin {nov_str} "
          f"CI(lin) [{boot['lineage_all'][0]:.3f},{boot['lineage_all'][1]:.3f}]")
    return out


def main():
    t0 = time.time()
    joint = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_humanorth.h5ad")  # counts CSR
    obs = joint.obs.copy()
    for c in ("timepoint", "fish", "lineage", "cluster", "split"):
        obs[c] = obs[c].astype(str)
    tr_m = (obs["split"] == "train").values
    print(f"joint {joint.shape}, train {tr_m.sum()}, test {(~tr_m).sum()}")

    ytr_lin = obs.loc[tr_m, "lineage"].values
    yte_lin = obs.loc[~tr_m, "lineage"].values
    ytr_clu = obs.loc[tr_m, "cluster"].values
    yte_clu = obs.loc[~tr_m, "cluster"].values

    lin_classes = sorted(set(ytr_lin))
    clu_classes = sorted(set(ytr_clu))
    lin_map = {c: i for i, c in enumerate(lin_classes)}
    clu_map = {c: i for i, c in enumerate(clu_classes)}
    lin_codes_tr = np.array([lin_map[c] for c in ytr_lin])
    lin_codes_te = np.array([lin_map.get(c, -1) for c in yte_lin])
    clu_codes_tr = np.array([clu_map[c] for c in ytr_clu])
    clu_codes_te = np.array([clu_map.get(c, -1) for c in yte_clu])  # novel cluster = -1（仅经 shared 掩码使用）
    shared = np.isin(yte_clu, clu_classes)
    novel = ~shared
    # train cluster 的多数 lineage（code）
    df_c2l = pd.DataFrame({"c": clu_codes_tr, "l": lin_codes_tr}).groupby("c")["l"].agg(lambda s: s.value_counts().index[0])
    clu2lin_code = df_c2l.to_dict()
    fish_te_codes = pd.factorize(obs.loc[~tr_m, "fish"].values)[0]

    ctx = dict(yte_lin=lin_codes_te, yte_clu=clu_codes_te, shared=shared, novel=novel,
               lin_codes_tr=lin_codes_tr, lin_codes_te=lin_codes_te, clu_codes_tr=clu_codes_tr,
               lin_classes=lin_classes, clu_classes=clu_classes, clu2lin_code=clu2lin_code,
               tp_te=obs.loc[~tr_m, "timepoint"].values, fish_te_codes=fish_te_codes)

    rep = {
        "protocol": ("unified gene set = 2581 human orthologs; unsupervised embedding + kNN "
                     f"(cosine, distance-weighted, k={K_MAIN}); frozen vs test-time-adapted"),
        "majority_lineage_all": float(pd.Series(yte_lin).value_counts(normalize=True).iloc[0]),
        "majority_cluster_shared_subset": float((yte_clu[shared] == pd.Series(ytr_clu).value_counts().index[0]).mean()),
        "n_test_fish": int(fish_te_codes.max() + 1),
        "n_train_cluster": len(clu_classes), "n_test_cluster": int(len(set(yte_clu))),
        "n_novel_cluster": int(len(set(yte_clu) - set(clu_classes))),
        "scvi_seeds": list(SEEDS),
        "scvi_frozen_note": "frozen 协议下未见测试批次映射到参考训练批次（训练集细胞数最多的鱼）",
    }
    print(f"majority lineage {rep['majority_lineage_all']:.4f}, "
          f"majority cluster(shared) {rep['majority_cluster_shared_subset']:.4f}, "
          f"test fish {rep['n_test_fish']}")

    # ---------- PCA (frozen) ----------
    trn = joint[tr_m].copy()
    ten = joint[~tr_m].copy()
    for a in (trn, ten):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    pca = PCA(n_components=50, random_state=SEED)
    Xtr_p = pca.fit_transform(to_dense(trn.X))
    Xte_p = pca.transform(to_dense(ten.X))
    rep["PCA_frozen"] = eval_embedding("PCA_frozen", "frozen", Xtr_p, Xte_p, ytr_lin, ytr_clu, ctx)
    del Xtr_p, Xte_p
    gc.collect()

    # ---------- Harmony (transductive adapted) ----------
    Xpca = np.vstack([pca.transform(to_dense(trn.X)), pca.transform(to_dense(ten.X))])
    jo = sc.concat([trn, ten], label="split", keys=["train", "test"])
    H = harmonize(Xpca, jo.obs, batch_key="fish")
    n_tr = int(tr_m.sum())
    rep["Harmony_adapted"] = eval_embedding("Harmony_adapted", "transductive", H[:n_tr], H[n_tr:], ytr_lin, ytr_clu, ctx)
    del Xpca, H, jo
    gc.collect()

    # ---------- scVI（frozen + surgery，×3 种子） ----------
    train_counts = joint[tr_m].copy()
    test_counts = joint[~tr_m].copy()
    sc_results = {}
    for seed in SEEDS:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(train_counts, batch_key="fish")
        vae = scvi.model.SCVI(train_counts, n_latent=30, n_layers=2, gene_likelihood="nb")
        vae.train(max_epochs=60, early_stopping=True, check_val_every_n_epoch=5)
        Xtr_v = vae.get_latent_representation()
        # 冻结协议：测试鱼批次训练时未见，scVI 拒绝转移 setup。
        # 处理：把全部测试细胞映射到参考批次（训练集细胞数最多的鱼）再前向。
        ref_fish = train_counts.obs["fish"].astype(str).value_counts().index[0]
        tf = test_counts.copy()
        tf.obs["fish"] = pd.Categorical(
            [ref_fish] * tf.n_obs,
            categories=train_counts.obs["fish"].astype("category").cat.categories,
        )
        Xte_frozen = vae.get_latent_representation(tf)
        del tf
        q = scvi.model.SCVI.load_query_data(test_counts, vae)
        q.train(max_epochs=30, plan_kwargs={"weight_decay": 0.0})
        Xte_surg = q.get_latent_representation()
        print(f"  scVI seed {seed} done ({time.time()-t0:.0f}s)")
        sc_results[seed] = {
            "frozen": eval_embedding(f"scVI_frozen_s{seed}", "frozen", Xtr_v, Xte_frozen, ytr_lin, ytr_clu, ctx),
            "surgery": eval_embedding(f"scVI_surgery_s{seed}", "query-training", Xtr_v, Xte_surg, ytr_lin, ytr_clu, ctx),
        }
        del vae, q, Xte_frozen, Xte_surg
        gc.collect()
        torch.cuda.empty_cache()

    # 种子聚合 mean±std
    for mode in ("frozen", "surgery"):
        agg = {}
        for key in ("lineage_all", "cluster_shared", "lineage_on_novel_clusters"):
            vals = [sc_results[s][mode][key]["accuracy"] for s in SEEDS]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": [float(v) for v in vals]}
        rep[f"scVI_{mode}_summary"] = agg
        rep[f"scVI_{mode}_s42"] = sc_results[42][mode]
        print(f"scVI {mode}: " + " | ".join(f"{k} {v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()))

    # ---------- STATE (frozen，复用已算嵌入) ----------
    st = sc.read_h5ad(DATA_PROC / "zebrahub_t1_joint_state.h5ad")
    assert (st.obs.index == joint.obs.index).all(), "joint_state 与 joint_humanorth 顺序不一致"
    Xs = np.asarray(st.obsm["X_state"], dtype=np.float32)
    rep["STATE_frozen"] = eval_embedding("STATE_frozen", "frozen", Xs[tr_m], Xs[~tr_m], ytr_lin, ytr_clu, ctx)

    # ---------- 与旧结果（4000 基因）对比 ----------
    old_p = RESULTS / "t1_hard_eval.json"
    if old_p.exists():
        old = json.loads(old_p.read_text())
        cmp = {}
        for old_k, new_k in (("PCA", "PCA_frozen"), ("Harmony", "Harmony_adapted"),
                             ("scVI", "scVI_surgery_s42"), ("STATE", "STATE_frozen")):
            if old_k in old:
                cmp[old_k] = {
                    "lineage_old4000": old[old_k]["lineage_all"]["accuracy"],
                    "lineage_new2581": rep[new_k]["lineage_all"]["accuracy"],
                    "cluster_old4000": old[old_k]["cluster_shared"]["accuracy"],
                    "cluster_new2581": rep[new_k]["cluster_shared"]["accuracy"],
                }
        rep["vs_old_4000gene"] = cmp
        print("对比(旧4000基因 → 新2581基因):")
        for k, v in cmp.items():
            print(f"  {k}: lineage {v['lineage_old4000']:.3f}->{v['lineage_new2581']:.3f} "
                  f"cluster {v['cluster_old4000']:.3f}->{v['cluster_new2581']:.3f}")

    rep["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS / "t1_unified_eval.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("saved", out)


if __name__ == "__main__":
    main()
