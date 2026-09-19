"""
Project 3 - Synaptic pathway tracer (Janelia hemibrain v1.2)
============================================================

Finds the strongest multi-synapse routes that carry a signal from one cell
type to another. Edge "distance" is 1 / synapse-weight, so a shortest path is
the route that uses the *heaviest* connections -- the path information is most
likely to actually travel along.

Default trace: KC -> MBON, i.e. Kenyon cells (mushroom-body input) to
mushroom-body output neurons -- the core learning-and-memory circuit of the
insect brain.

Examples
--------
    python trace.py                          # KC -> MBON (default)
    python trace.py --source KC --target MBON --k 5
    python trace.py --source ER --target EPG  # central-complex compass circuit

Data: Janelia FlyEM hemibrain v1.2 traced adjacencies (CC-BY-4.0).
Outputs: printed paths + a PNG circuit diagram in ./outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import HEMIBRAIN_CONNECTIONS, HEMIBRAIN_NEURONS, outdir, require

OUT = outdir(__file__)
BACKBONE_MIN_WEIGHT = 5  # ignore weak/noisy 1-4 synapse connections when routing


def type_bodyids(neurons: pd.DataFrame, prefix: str) -> list[int]:
    """bodyIds whose hemibrain type starts with `prefix` (case-insensitive)."""
    m = neurons["type"].fillna("").str.upper().str.startswith(prefix.upper())
    return neurons.loc[m, "bodyId"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="KC", help="source cell-type prefix (default KC)")
    ap.add_argument("--target", default="MBON", help="target cell-type prefix (default MBON)")
    ap.add_argument("--k", type=int, default=5, help="number of paths to report")
    args = ap.parse_args()

    require(HEMIBRAIN_CONNECTIONS)
    require(HEMIBRAIN_NEURONS)

    print("Loading hemibrain connectivity...")
    conn = pd.read_csv(HEMIBRAIN_CONNECTIONS)
    neurons = pd.read_csv(HEMIBRAIN_NEURONS)
    id2type = dict(zip(neurons["bodyId"], neurons["type"].fillna("?")))

    src = set(type_bodyids(neurons, args.source))
    dst = set(type_bodyids(neurons, args.target))
    if not src:
        raise SystemExit(f"No neurons with type starting '{args.source}'. "
                         "Try KC, MBON, ER, EPG, PEN, DA, PPL, LC...")
    if not dst:
        raise SystemExit(f"No neurons with type starting '{args.target}'.")
    print(f"  source '{args.source}': {len(src):,} neurons | "
          f"target '{args.target}': {len(dst):,} neurons")

    strong = conn[conn["weight"] >= BACKBONE_MIN_WEIGHT].copy()
    strong["cost"] = 1.0 / strong["weight"]  # heavier synapse -> shorter distance
    G = nx.from_pandas_edgelist(
        strong, "bodyId_pre", "bodyId_post",
        edge_attr=["weight", "cost"], create_using=nx.DiGraph,
    )

    # Virtual super-source / super-sink so one Dijkstra finds the best type->type route.
    S, T = "SOURCE", "TARGET"
    G.add_node(S); G.add_node(T)
    for b in src:
        if b in G:
            G.add_edge(S, b, weight=0, cost=0.0)
    for b in dst:
        if b in G:
            G.add_edge(b, T, weight=0, cost=0.0)

    try:
        length, path = nx.single_source_dijkstra(G, S, T, weight="cost")
    except nx.NetworkXNoPath:
        raise SystemExit(f"No pathway found from {args.source} to {args.target} "
                         f"through connections of >={BACKBONE_MIN_WEIGHT} synapses.")

    core = path[1:-1]  # strip virtual endpoints
    hops = [(core[i], core[i + 1]) for i in range(len(core) - 1)]

    print(f"\n===== Strongest {args.source} -> {args.target} pathway =====")
    print(f"  {len(hops)} synaptic hops:")
    for a, b in hops:
        w = int(G[a][b]["weight"])
        print(f"    {id2type[a]:>10} ({a})  --{w:>4} syn-->  {id2type[b]:>10} ({b})")

    # A few alternative shortest simple paths (bounded) for context.
    print(f"\n  up to {args.k} alternative strong routes (by type sequence):")
    seen, rows = set(), []
    try:
        for p in nx.shortest_simple_paths(G, S, T, weight="cost"):
            c = p[1:-1]
            seq = " -> ".join(dict.fromkeys(id2type[x] for x in c))  # dedupe repeats
            if seq in seen:
                continue
            seen.add(seq)
            cost = sum(G[c[i]][c[i + 1]]["cost"] for i in range(len(c) - 1))
            rows.append((seq, len(c) - 1, round(cost, 4)))
            print(f"    [{len(c)-1} hops]  {seq}")
            if len(rows) >= args.k:
                break
    except nx.NetworkXNoPath:
        pass
    pd.DataFrame(rows, columns=["type_path", "hops", "path_cost"]).to_csv(
        OUT / f"paths_{args.source}_to_{args.target}.csv", index=False)

    # ---- draw the primary pathway ------------------------------------------
    P = nx.DiGraph()
    for a, b in hops:
        P.add_edge(f"{id2type[a]}\n{a}", f"{id2type[b]}\n{b}", weight=int(G[a][b]["weight"]))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(hops)), 4))
    pos = {n: (i, 0) for i, n in enumerate(dict.fromkeys(
        [f"{id2type[a]}\n{a}" for a, _ in hops] + [f"{id2type[hops[-1][1]]}\n{hops[-1][1]}"]))}
    nx.draw_networkx_nodes(P, pos, ax=ax, node_color="#3c887e",
                           node_size=2600, alpha=0.9)
    nx.draw_networkx_labels(P, pos, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edges(P, pos, ax=ax, arrowsize=22, node_size=2600,
                           edge_color="#555", width=2, connectionstyle="arc3,rad=0.0")
    nx.draw_networkx_edge_labels(
        P, pos, ax=ax,
        edge_labels={(u, v): f"{d['weight']} syn" for u, v, d in P.edges(data=True)},
        font_size=8)
    ax.set_title(f"Strongest hemibrain pathway: {args.source} -> {args.target}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"pathway_{args.source}_to_{args.target}.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote diagram + CSV to {OUT}")


if __name__ == "__main__":
    main()
