"""
Project 1 - Connectome graph analysis (Janelia hemibrain v1.2)
==============================================================

Treats the fly brain as a directed, weighted network of neurons and measures
its large-scale structure:

  * basic graph statistics (neurons, connections, density, reciprocity)
  * in/out degree distributions (the connectome is heavy-tailed / scale-free-ish)
  * the strongest "hub" neurons, annotated with their cell types
  * weakly-connected components of the strong-connection backbone

Heavy degree/strength work is done in pandas (fast, low memory over 3.5M edges);
NetworkX is used only on a pruned backbone so it stays light.

Data: Janelia FlyEM hemibrain v1.2 traced adjacencies (CC-BY-4.0).
Outputs: PNG figures + CSV tables in ./outputs/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import HEMIBRAIN_CONNECTIONS, HEMIBRAIN_NEURONS, outdir, require

OUT = outdir(__file__)
BACKBONE_MIN_WEIGHT = 10  # keep connections of >=10 synapses for the graph backbone


def main() -> None:
    require(HEMIBRAIN_CONNECTIONS)
    require(HEMIBRAIN_NEURONS)

    print("Loading hemibrain connectivity (3.5M edges)...")
    conn = pd.read_csv(HEMIBRAIN_CONNECTIONS)          # bodyId_pre, bodyId_post, weight
    neurons = pd.read_csv(HEMIBRAIN_NEURONS)           # bodyId, type, instance

    n_neurons = neurons["bodyId"].nunique()
    n_edges = len(conn)
    total_synapses = int(conn["weight"].sum())
    density = n_edges / (n_neurons * (n_neurons - 1))

    # ---- degree & strength via pandas groupby (cheap) -----------------------
    out_deg = conn.groupby("bodyId_pre").size().rename("out_degree")
    in_deg = conn.groupby("bodyId_post").size().rename("in_degree")
    out_str = conn.groupby("bodyId_pre")["weight"].sum().rename("out_strength")
    in_str = conn.groupby("bodyId_post")["weight"].sum().rename("in_strength")

    stats = (
        pd.concat([out_deg, in_deg, out_str, in_str], axis=1)
        .fillna(0)
        .astype(np.int64)
    )
    stats["total_degree"] = stats["out_degree"] + stats["in_degree"]
    stats["total_strength"] = stats["out_strength"] + stats["in_strength"]
    stats = stats.merge(
        neurons.set_index("bodyId")[["type", "instance"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    # ---- reciprocity: fraction of connected pairs that are bidirectional ----
    fwd = set(map(tuple, conn[["bodyId_pre", "bodyId_post"]].to_numpy()))
    recip_edges = sum(1 for a, b in fwd if (b, a) in fwd)
    reciprocity = recip_edges / n_edges

    print("\n===== Hemibrain connectome summary =====")
    print(f"  neurons (traced)      : {n_neurons:,}")
    print(f"  connections (edges)   : {n_edges:,}")
    print(f"  total synapses        : {total_synapses:,}")
    print(f"  graph density         : {density:.6f}")
    print(f"  edge reciprocity      : {reciprocity:.3f}")
    print(f"  mean out-degree       : {stats['out_degree'].mean():.1f}")
    print(f"  max total strength    : {stats['total_strength'].max():,}")

    # ---- top hub neurons ----------------------------------------------------
    top_hubs = stats.sort_values("total_strength", ascending=False).head(25).copy()
    top_hubs.index.name = "bodyId"
    top_hubs.to_csv(OUT / "top_hubs.csv")
    stats.to_csv(OUT / "neuron_degree_table.csv")
    print(f"\nTop 5 hub neurons by total synaptic strength:")
    for bid, r in top_hubs.head(5).iterrows():
        print(f"  {bid}  {str(r['type']):>10}  strength={int(r['total_strength']):,}")

    # ---- figure 1: degree distribution (log-log) ----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for series, label, color, a in [
        (stats["out_degree"], "out-degree", "#d1495b", ax[0]),
        (stats["in_degree"], "in-degree", "#30638e", ax[1]),
    ]:
        vals = series[series > 0]
        counts = vals.value_counts().sort_index()
        a.loglog(counts.index, counts.values, "o", ms=3, color=color, alpha=0.6)
        a.set_xlabel(f"{label} (partners)")
        a.set_ylabel("number of neurons")
        a.set_title(f"Hemibrain {label} distribution")
        a.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle("Fly-brain connectivity is heavy-tailed: a few neurons are massive hubs")
    fig.tight_layout()
    fig.savefig(OUT / "degree_distribution.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: top hubs bar chart ---------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{t if isinstance(t, str) else '?'} ({bid})"
              for bid, t in zip(top_hubs.index[:20], top_hubs["type"][:20])]
    ax.barh(range(20), top_hubs["total_strength"][:20], color="#3c887e")
    ax.set_yticks(range(20))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("total synaptic strength (in + out)")
    ax.set_title("Top 20 hub neurons in the hemibrain")
    fig.tight_layout()
    fig.savefig(OUT / "top_hubs.png", dpi=140)
    plt.close(fig)

    # ---- backbone graph (pruned) -> connected components --------------------
    strong = conn[conn["weight"] >= BACKBONE_MIN_WEIGHT]
    print(f"\nBuilding backbone graph (weight >= {BACKBONE_MIN_WEIGHT}): "
          f"{len(strong):,} edges")
    G = nx.from_pandas_edgelist(
        strong, "bodyId_pre", "bodyId_post",
        edge_attr="weight", create_using=nx.DiGraph,
    )
    wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    print(f"  backbone neurons          : {G.number_of_nodes():,}")
    print(f"  weakly-connected comps    : {len(wccs)} (largest {len(wccs[0]):,})")
    print(f"  strongly-connected comps  : {len(sccs)} (largest {len(sccs[0]):,})")

    with open(OUT / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("Hemibrain v1.2 connectome graph summary\n")
        fh.write(f"neurons={n_neurons} edges={n_edges} synapses={total_synapses}\n")
        fh.write(f"density={density:.6f} reciprocity={reciprocity:.3f}\n")
        fh.write(f"backbone(weight>={BACKBONE_MIN_WEIGHT}): nodes="
                 f"{G.number_of_nodes()} edges={G.number_of_edges()}\n")
        fh.write(f"largest weakly-connected component={len(wccs[0])}\n")
        fh.write(f"largest strongly-connected component={len(sccs[0])}\n")

    print(f"\nWrote figures + tables to {OUT}")


if __name__ == "__main__":
    main()
