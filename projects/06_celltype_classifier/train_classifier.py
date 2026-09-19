"""
Project 6 - Connectivity-based cell-type classifier (hemibrain v1.2)
====================================================================

*The* canonical connectomics machine-learning task: can we predict what TYPE a
neuron is purely from *who it talks to*? Each neuron is described by a
"connectivity fingerprint" -- the fraction of its input/output synapses shared
with every major cell-type family -- and a model learns to recover the neuron's
own family from that fingerprint.

Pipeline
--------
  * build per-neuron features: output- and input-synapse fractions across cell
    families (+ degree/strength summaries)
  * targets: cell-type families with >= MIN_MEMBERS neurons
  * train two models -- RandomForest and a neural network (MLP) -- with a
    stratified train/test split, and report accuracy + macro-F1
  * save a confusion matrix and the most informative connectivity features

Data: Janelia FlyEM hemibrain v1.2 (CC BY 4.0).
Outputs: metrics + figures in ./outputs/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_hemibrain, outdir

OUT = outdir(__file__)
MIN_MEMBERS = 40   # a family needs this many neurons to be a target class
RANDOM_STATE = 42


def partner_fraction_matrix(conn, neurons, self_col, partner_col, vocab):
    """Rows = neurons (self_col), cols = partner family fractions of synapse weight."""
    fam = neurons.set_index("bodyId")["fam_feat"]
    df = conn[[self_col, partner_col, "weight"]].copy()
    df["weight"] = df["weight"].astype("float64")
    df["pfam"] = df[partner_col].map(fam).fillna("other")
    agg = df.groupby([self_col, "pfam"])["weight"].sum().unstack(fill_value=0.0)
    agg = agg.reindex(columns=vocab, fill_value=0.0)
    totals = agg.sum(axis=1).replace(0, 1.0)
    return agg.div(totals, axis=0)


def main() -> None:
    print("Loading hemibrain + building cell-type families...")
    conn, neurons = load_hemibrain()

    fam_counts = neurons["family"].value_counts()
    kept = [f for f in fam_counts.index if f != "other" and fam_counts[f] >= MIN_MEMBERS]
    vocab = kept + ["other"]
    neurons["fam_feat"] = neurons["family"].where(neurons["family"].isin(kept), "other")
    print(f"  {len(kept)} target families (>= {MIN_MEMBERS} neurons each), "
          f"covering {fam_counts[kept].sum():,} neurons")

    # ---- features: output & input synapse-fraction fingerprints -------------
    print("Building connectivity fingerprints...")
    out_frac = partner_fraction_matrix(conn, neurons, "bodyId_pre", "bodyId_post", vocab)
    in_frac = partner_fraction_matrix(conn, neurons, "bodyId_post", "bodyId_pre", vocab)
    out_frac = out_frac.add_prefix("out->")
    in_frac = in_frac.add_prefix("in<-")

    feats = out_frac.join(in_frac, how="outer").fillna(0.0)
    # degree / strength summaries
    deg = pd.DataFrame(index=feats.index)
    deg["log_out_str"] = np.log1p(conn.groupby("bodyId_pre")["weight"].sum())
    deg["log_in_str"] = np.log1p(conn.groupby("bodyId_post")["weight"].sum())
    deg["log_out_deg"] = np.log1p(conn.groupby("bodyId_pre").size())
    deg["log_in_deg"] = np.log1p(conn.groupby("bodyId_post").size())
    feats = feats.join(deg).fillna(0.0)

    # ---- assemble supervised dataset ----------------------------------------
    lab = neurons.set_index("bodyId")["family"]
    y_all = lab.reindex(feats.index)
    mask = y_all.isin(kept)
    X = feats[mask.values]
    y = y_all[mask.values].astype(str)
    print(f"  dataset: {X.shape[0]:,} neurons x {X.shape[1]} features, "
          f"{y.nunique()} classes")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    results = {}

    # ---- model 1: RandomForest ---------------------------------------------
    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                random_state=RANDOM_STATE, class_weight="balanced_subsample")
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    results["random_forest"] = {
        "accuracy": round(accuracy_score(y_te, rf_pred), 4),
        "macro_f1": round(f1_score(y_te, rf_pred, average="macro"), 4),
    }

    # ---- model 2: neural network (MLP) --------------------------------------
    print("Training neural network (MLP)...")
    scaler = StandardScaler().fit(X_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                        early_stopping=True, random_state=RANDOM_STATE)
    mlp.fit(scaler.transform(X_tr), y_tr)
    mlp_pred = mlp.predict(scaler.transform(X_te))
    results["mlp"] = {
        "accuracy": round(accuracy_score(y_te, mlp_pred), 4),
        "macro_f1": round(f1_score(y_te, mlp_pred, average="macro"), 4),
    }

    # baseline: always predict the majority class
    majority = y_tr.value_counts().idxmax()
    results["baseline_majority"] = {
        "accuracy": round((y_te == majority).mean(), 4), "macro_f1": 0.0}

    print("\n===== Test-set performance =====")
    for name, m in results.items():
        print(f"  {name:<18} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")

    best_name, best_pred = ("random_forest", rf_pred) if \
        results["random_forest"]["accuracy"] >= results["mlp"]["accuracy"] else ("mlp", mlp_pred)

    # ---- confusion matrix (top-15 classes by support) -----------------------
    top = y_te.value_counts().head(15).index.tolist()
    cm = confusion_matrix(y_te, best_pred, labels=top, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=8)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Cell-type confusion matrix ({best_name}, top 15 families)")
    fig.colorbar(im, ax=ax, label="row-normalized")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=140); plt.close(fig)

    # ---- most informative connectivity features (RF importance) -------------
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.sort_values().plot.barh(ax=ax, color="#3c887e")
    ax.set_xlabel("RandomForest importance")
    ax.set_title("Top 20 connectivity features for predicting cell type")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=140); plt.close(fig)

    # ---- persist metrics ----------------------------------------------------
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Best model: {best_name}\n\n")
        fh.write(classification_report(y_te, best_pred, zero_division=0))

    print(f"\nBest model: {best_name} "
          f"(accuracy {results[best_name]['accuracy']:.1%})")
    print(f"Wrote metrics + figures to {OUT}")


if __name__ == "__main__":
    main()
