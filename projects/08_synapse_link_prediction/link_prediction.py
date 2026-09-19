"""
Project 8 - Synapse link prediction (hemibrain v1.2)
====================================================

Given two neurons, can a model predict whether one synapses onto the other?
This is the classic graph machine-learning "link prediction" task, done here on
a real brain -- with a leakage-free protocol: node embeddings are learned from a
TRAINING graph only, and the model is scored on held-out edges it never saw.

Protocol
--------
  * hold out 15% of real synaptic connections as positive test edges
  * learn directed node embeddings (SVD of output / input profiles) from the
    remaining training graph ONLY
  * features for an ordered pair (i -> j): compatibility of i's output embedding
    with j's input embedding (Hadamard), plus structural degree/strength/reverse
  * train a gradient-boosted classifier on train edges vs. sampled non-edges
  * evaluate on held-out positives vs. fresh non-edges -> ROC-AUC & PR-AUC

Data: Janelia FlyEM hemibrain v1.2 (CC BY 4.0).
Outputs: metrics + ROC/PR curves in ./outputs/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_hemibrain, outdir

OUT = outdir(__file__)
EMB_DIM = 64
TEST_FRAC = 0.15
N_TRAIN_PAIRS = 120_000   # positive train edges (+ equal negatives)
N_TEST_PAIRS = 40_000     # positive test edges (+ equal negatives)
RANDOM_STATE = 42


def sample_non_edges(n, edge_keys, count, rng, exclude_self=True):
    """Sample `count` ordered pairs (i, j) that are NOT edges, via sorted-key lookup."""
    out_i, out_j = [], []
    need = count
    while need > 0:
        i = rng.integers(0, n, size=need * 2)
        j = rng.integers(0, n, size=need * 2)
        if exclude_self:
            ok = i != j
            i, j = i[ok], j[ok]
        keys = i.astype(np.int64) * n + j.astype(np.int64)
        pos = np.searchsorted(edge_keys, keys)
        pos = np.clip(pos, 0, len(edge_keys) - 1)
        is_edge = edge_keys[pos] == keys
        i, j = i[~is_edge], j[~is_edge]
        take = min(len(i), need)
        out_i.append(i[:take]); out_j.append(j[:take])
        need -= take
    return np.concatenate(out_i), np.concatenate(out_j)


def edge_features(si, tj, emb_out, emb_in, out_str, in_str, out_deg, in_deg, edge_keys, n):
    """Directed pair features: output/input embedding compatibility + structure."""
    had = emb_out[si] * emb_in[tj]                     # (m, EMB_DIM)
    rev_keys = tj.astype(np.int64) * n + si.astype(np.int64)
    rpos = np.clip(np.searchsorted(edge_keys, rev_keys), 0, len(edge_keys) - 1)
    reciprocal = (edge_keys[rpos] == rev_keys).astype(np.float64)
    struct = np.column_stack([
        np.log1p(out_str[si]), np.log1p(in_str[tj]),
        np.log1p(out_deg[si]), np.log1p(in_deg[tj]), reciprocal,
    ])
    return np.hstack([had, struct])


def main() -> None:
    print("Loading hemibrain...")
    conn, neurons = load_hemibrain()
    ids = neurons["bodyId"].to_numpy()
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)

    pre = conn["bodyId_pre"].map(idx).to_numpy()
    post = conn["bodyId_post"].map(idx).to_numpy()
    w = conn["weight"].to_numpy(dtype=np.float64)
    m_edges = len(pre)

    # full edge key set (for non-edge sampling / reciprocity lookups)
    all_keys = np.sort(pre.astype(np.int64) * n + post.astype(np.int64))

    rng = np.random.default_rng(RANDOM_STATE)
    test_mask = rng.random(m_edges) < TEST_FRAC
    tr = ~test_mask
    print(f"  {m_edges:,} edges -> {tr.sum():,} train / {test_mask.sum():,} test")

    # ---- embeddings from TRAINING graph only --------------------------------
    print("Learning node embeddings from the training graph...")
    A_tr = sparse.coo_matrix((w[tr], (pre[tr], post[tr])), shape=(n, n)).tocsr()
    emb_out = TruncatedSVD_fit(normalize(A_tr, norm="l1", axis=1))
    emb_in = TruncatedSVD_fit(normalize(A_tr.T.tocsr(), norm="l1", axis=1))

    out_str = np.asarray(A_tr.sum(axis=1)).ravel()
    in_str = np.asarray(A_tr.sum(axis=0)).ravel()
    out_deg = np.diff(A_tr.indptr)
    A_tr_csc = A_tr.tocsc()
    in_deg = np.diff(A_tr_csc.indptr)

    def feats(si, tj):
        return edge_features(si, tj, emb_out, emb_in, out_str, in_str,
                             out_deg, in_deg, all_keys, n)

    # ---- build train / test sets --------------------------------------------
    print("Sampling positive / negative pairs...")
    tr_pos = rng.choice(np.where(tr)[0], size=min(N_TRAIN_PAIRS, tr.sum()), replace=False)
    te_pos = rng.choice(np.where(test_mask)[0], size=min(N_TEST_PAIRS, test_mask.sum()), replace=False)
    neg_tr_i, neg_tr_j = sample_non_edges(n, all_keys, len(tr_pos), rng)
    neg_te_i, neg_te_j = sample_non_edges(n, all_keys, len(te_pos), rng)

    X_tr = np.vstack([feats(pre[tr_pos], post[tr_pos]), feats(neg_tr_i, neg_tr_j)])
    y_tr = np.r_[np.ones(len(tr_pos)), np.zeros(len(neg_tr_i))]
    X_te = np.vstack([feats(pre[te_pos], post[te_pos]), feats(neg_te_i, neg_te_j)])
    y_te = np.r_[np.ones(len(te_pos)), np.zeros(len(neg_te_i))]

    # ---- train classifier ---------------------------------------------------
    print(f"Training HistGradientBoosting on {len(y_tr):,} pairs "
          f"({X_tr.shape[1]} features each)...")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]

    roc = roc_auc_score(y_te, prob)
    ap = average_precision_score(y_te, prob)
    metrics = {"train_pairs": int(len(y_tr)), "test_pairs": int(len(y_te)),
               "features": int(X_tr.shape[1]), "roc_auc": round(roc, 4),
               "pr_auc_average_precision": round(ap, 4)}
    print("\n===== Link-prediction performance (held-out edges) =====")
    for k, v in metrics.items():
        print(f"  {k:<26} {v}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- ROC + PR curves ----------------------------------------------------
    fpr, tpr, _ = roc_curve(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, color="#d1495b", lw=2, label=f"AUC = {roc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC curve"); ax[0].legend(loc="lower right")
    ax[1].plot(rec, prec, color="#30638e", lw=2, label=f"AP = {ap:.3f}")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-Recall curve"); ax[1].legend(loc="lower left")
    fig.suptitle("Predicting whether one neuron synapses onto another")
    fig.tight_layout(); fig.savefig(OUT / "roc_pr_curves.png", dpi=140); plt.close(fig)

    print(f"\nWrote metrics + curves to {OUT}")


def TruncatedSVD_fit(mat):
    from sklearn.decomposition import TruncatedSVD
    return TruncatedSVD(EMB_DIM, random_state=RANDOM_STATE).fit_transform(mat)


if __name__ == "__main__":
    main()
