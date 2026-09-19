"""
Project 7 - Neuron embeddings & unsupervised cell typing (hemibrain v1.2)
========================================================================

Can cell types emerge from connectivity *without* any labels? We learn a
low-dimensional embedding of every neuron purely from the wiring, cluster the
embeddings, and then check -- using the held-out expert annotations only as a
scorecard -- how well the unsupervised clusters recover real cell-type families.

Method
------
  * build the sparse synapse adjacency matrix A (neurons x neurons)
  * embed each neuron by truncated SVD of its row-normalized OUTPUT profile and
    INPUT profile (A and A^T) -> a 2*k dimensional connectivity embedding
  * KMeans-cluster the embeddings
  * score against annotated families with Adjusted Rand Index (ARI) and
    Normalized Mutual Information (NMI)
  * visualize with a 2-D t-SNE colored by true family

Data: Janelia FlyEM hemibrain v1.2 (CC BY 4.0).
Outputs: metrics + t-SNE figure in ./outputs/
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
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_hemibrain, outdir

OUT = outdir(__file__)
EMB_DIM = 64          # SVD components per direction (embedding = 2 * EMB_DIM)
MIN_MEMBERS = 40      # families used for scoring / coloring
TSNE_SAMPLE = 4000    # neurons drawn for the 2-D visualization
RANDOM_STATE = 42


def main() -> None:
    print("Loading hemibrain...")
    conn, neurons = load_hemibrain()

    ids = neurons["bodyId"].to_numpy()
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)

    pre = conn["bodyId_pre"].map(idx).to_numpy()
    post = conn["bodyId_post"].map(idx).to_numpy()
    w = conn["weight"].to_numpy(dtype=np.float64)
    A = sparse.coo_matrix((w, (pre, post)), shape=(n, n)).tocsr()
    print(f"  adjacency: {n:,} neurons, {A.nnz:,} nonzero synaptic links")

    # ---- embed: SVD of row-normalized output and input profiles -------------
    print(f"Learning {2*EMB_DIM}-d connectivity embeddings (truncated SVD)...")
    A_out = normalize(A, norm="l1", axis=1)          # "who I output to"
    A_in = normalize(A.T.tocsr(), norm="l1", axis=1)  # "who inputs to me"
    svd_out = TruncatedSVD(EMB_DIM, random_state=RANDOM_STATE).fit_transform(A_out)
    svd_in = TruncatedSVD(EMB_DIM, random_state=RANDOM_STATE).fit_transform(A_in)
    emb = normalize(np.hstack([svd_out, svd_in]))

    # ---- cluster ------------------------------------------------------------
    fam_counts = neurons["family"].value_counts()
    kept = [f for f in fam_counts.index if f != "other" and fam_counts[f] >= MIN_MEMBERS]
    n_clusters = len(kept)
    print(f"Clustering into {n_clusters} groups (= number of annotated families)...")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(emb)

    # ---- score vs annotations (on labelled neurons only) --------------------
    fam = neurons["family"].to_numpy()
    labelled = np.isin(fam, kept)
    ari = adjusted_rand_score(fam[labelled], clusters[labelled])
    nmi = normalized_mutual_info_score(fam[labelled], clusters[labelled])

    # cluster purity: for each cluster, fraction that is its dominant family
    dfc = pd.DataFrame({"cluster": clusters[labelled], "family": fam[labelled]})
    purity = (dfc.groupby("cluster")["family"]
              .agg(lambda s: s.value_counts().iloc[0] / len(s)))
    mean_purity = float(purity.mean())

    metrics = {"embedding_dim": 2 * EMB_DIM, "n_clusters": n_clusters,
               "labelled_neurons": int(labelled.sum()),
               "adjusted_rand_index": round(ari, 4),
               "normalized_mutual_info": round(nmi, 4),
               "mean_cluster_purity": round(mean_purity, 4)}
    print("\n===== Unsupervised recovery of cell types =====")
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- t-SNE visualization (subsample, colored by family) -----------------
    print(f"\nRendering t-SNE ({TSNE_SAMPLE} neurons)...")
    rng = np.random.default_rng(RANDOM_STATE)
    pool = np.where(labelled)[0]
    sample = rng.choice(pool, size=min(TSNE_SAMPLE, len(pool)), replace=False)
    xy = TSNE(n_components=2, init="pca", perplexity=30,
              random_state=RANDOM_STATE).fit_transform(emb[sample])

    top_fams = pd.Series(fam[sample]).value_counts().head(12).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    other = ~np.isin(fam[sample], top_fams)
    ax.scatter(xy[other, 0], xy[other, 1], s=6, color="#dddddd", label="other")
    for i, f in enumerate(top_fams):
        m = fam[sample] == f
        ax.scatter(xy[m, 0], xy[m, 1], s=10, color=cmap(i % 20), label=f)
    ax.legend(fontsize=7, ncol=2, markerscale=1.5, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Neuron connectivity embedding (t-SNE)\n"
                 f"clusters recover cell types: ARI={ari:.2f}, NMI={nmi:.2f}")
    fig.tight_layout(); fig.savefig(OUT / "tsne_embedding.png", dpi=140); plt.close(fig)

    print(f"Wrote metrics + figure to {OUT}")


if __name__ == "__main__":
    main()
