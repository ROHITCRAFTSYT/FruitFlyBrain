"""
Project 4 - Connectome-constrained neural-network simulation (hemibrain v1.2)
=============================================================================

Most artificial networks invent their wiring. Here the wiring IS the real fly
brain: we build a recurrent network whose synaptic weight matrix is taken
directly from the hemibrain connectome, then watch a stimulus propagate through
it.

Model (deliberately simple, NOT biophysical):
    x(t+1) = ReLU( gain * W_norm @ x(t) + stimulus(t) )
where W_norm[i, j] is the (normalised) synapse count from neuron j onto neuron
i. W_norm is scaled so the dynamics stay bounded, a seed cell type is stimulated
for the first few steps, and we record how activity cascades through the circuit.

Default subsystem: the central complex (the fly's navigation / "compass" hub -
types EPG, PEN, PEG, ER, EL, FB, PB, Delta7, ...).

Data: Janelia FlyEM hemibrain v1.2 traced adjacencies (CC-BY-4.0).
Outputs: activity heatmap + per-type response curves (PNG) and CSV in ./outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import HEMIBRAIN_CONNECTIONS, HEMIBRAIN_NEURONS, outdir, require

OUT = outdir(__file__)

# Central-complex cell-type prefixes (the classic navigation circuit).
CX_PREFIXES = ("EPG", "PEN", "PEG", "ER", "EL", "FB", "PB", "Delta7", "FC", "FS", "PFN")
MAX_NODES = 3000  # cap subnetwork size for speed / memory


def select_subset(neurons: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    up = tuple(p.upper() for p in prefixes)
    m = neurons["type"].fillna("").str.upper().str.startswith(up)
    return neurons.loc[m, ["bodyId", "type"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", nargs="+", default=list(CX_PREFIXES),
                    help="cell-type prefixes defining the subnetwork")
    ap.add_argument("--seed-type", default="ER",
                    help="cell-type prefix to stimulate (default ER = ring neurons)")
    ap.add_argument("--steps", type=int, default=40, help="simulation time steps")
    ap.add_argument("--stim-steps", type=int, default=5, help="steps the stimulus is on")
    ap.add_argument("--gain", type=float, default=0.9, help="recurrent gain (<1 = stable)")
    args = ap.parse_args()

    require(HEMIBRAIN_CONNECTIONS)
    require(HEMIBRAIN_NEURONS)

    print("Loading hemibrain connectivity...")
    conn = pd.read_csv(HEMIBRAIN_CONNECTIONS)
    neurons = pd.read_csv(HEMIBRAIN_NEURONS)

    sub = select_subset(neurons, tuple(args.prefixes))
    if sub.empty:
        raise SystemExit(f"No neurons matched prefixes {args.prefixes}.")

    # Keep only edges internal to the subset.
    ids = set(sub["bodyId"])
    e = conn[conn["bodyId_pre"].isin(ids) & conn["bodyId_post"].isin(ids)]
    if e.empty:
        raise SystemExit("Subset has no internal connections.")

    # If the subset is huge, keep the most-connected neurons.
    if len(sub) > MAX_NODES:
        strength = (e.groupby("bodyId_pre")["weight"].sum()
                    .add(e.groupby("bodyId_post")["weight"].sum(), fill_value=0))
        keep = set(strength.sort_values(ascending=False).head(MAX_NODES).index)
        sub = sub[sub["bodyId"].isin(keep)]
        ids = set(sub["bodyId"])
        e = conn[conn["bodyId_pre"].isin(ids) & conn["bodyId_post"].isin(ids)]

    body_ids = sub["bodyId"].tolist()
    idx = {b: i for i, b in enumerate(body_ids)}
    types = sub.set_index("bodyId")["type"].fillna("?").to_dict()
    n = len(body_ids)
    print(f"  subnetwork: {n:,} neurons, {len(e):,} internal connections")

    # ---- build weight matrix W[i, j] = synapses from j -> i -----------------
    W = np.zeros((n, n), dtype=np.float32)
    pre = e["bodyId_pre"].map(idx).to_numpy()
    post = e["bodyId_post"].map(idx).to_numpy()
    np.add.at(W, (post, pre), e["weight"].to_numpy(dtype=np.float32))

    # Normalise so the spectral radius is bounded by 1 (max row sum), then apply gain.
    max_row = float(W.sum(axis=1).max()) or 1.0
    Wn = W / max_row

    # ---- stimulus: pulse into the seed population ---------------------------
    seed_pref = args.seed_type.upper()
    seed_idx = [idx[b] for b in body_ids if str(types[b]).upper().startswith(seed_pref)]
    if not seed_idx:  # fall back to the most numerous type in the subset
        top = sub["type"].value_counts().idxmax()
        seed_pref = str(top)
        seed_idx = [idx[b] for b in body_ids if types[b] == top]
        print(f"  (seed '{args.seed_type}' not in subset; using '{seed_pref}')")
    print(f"  stimulating {len(seed_idx)} '{seed_pref}' neurons for "
          f"{args.stim_steps} steps")

    stim = np.zeros(n, dtype=np.float32)
    stim[seed_idx] = 1.0

    # ---- run the dynamics ---------------------------------------------------
    x = np.zeros(n, dtype=np.float32)
    hist = np.zeros((args.steps, n), dtype=np.float32)
    for t in range(args.steps):
        drive = stim if t < args.stim_steps else 0.0
        x = np.maximum(0.0, args.gain * (Wn @ x) + drive)
        hist[t] = x

    # ---- per-type mean response over time -----------------------------------
    type_arr = np.array([str(types[b]) for b in body_ids])
    uniq = pd.unique(type_arr)
    type_curves = {}
    for tp in uniq:
        cols = np.where(type_arr == tp)[0]
        type_curves[tp] = hist[:, cols].mean(axis=1)
    curves_df = pd.DataFrame(type_curves)
    curves_df.index.name = "step"
    curves_df.to_csv(OUT / "type_activity_over_time.csv")

    # Most responsive downstream types (exclude the stimulated seed type group).
    peak = curves_df.max().sort_values(ascending=False)
    downstream = [t for t in peak.index if not str(t).upper().startswith(seed_pref)][:8]
    print("\n  most strongly activated downstream types (peak activity):")
    for tp in downstream[:6]:
        print(f"      {tp:<10} peak={peak[tp]:.3f}")

    # ---- figure 1: activity heatmap (neurons sorted by peak time) -----------
    order = np.argsort(np.argmax(hist, axis=0))
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(hist[:, order].T, aspect="auto", cmap="inferno",
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("time step")
    ax.set_ylabel("neuron (sorted by activation time)")
    ax.set_title(f"Activity cascade through the fly central complex "
                 f"(stimulus: {seed_pref})")
    fig.colorbar(im, ax=ax, label="activity")
    fig.tight_layout()
    fig.savefig(OUT / "activity_heatmap.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: per-type response curves ---------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for tp in [t for t in peak.index if str(t).upper().startswith(seed_pref)][:1] + downstream:
        ax.plot(curves_df.index, curves_df[tp], label=str(tp), lw=2)
    ax.axvspan(0, args.stim_steps - 1, color="grey", alpha=0.15, label="stimulus on")
    ax.set_xlabel("time step")
    ax.set_ylabel("mean activity")
    ax.set_title("How the stimulus spreads across cell types")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "type_response_curves.png", dpi=140)
    plt.close(fig)

    with open(OUT / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("Connectome-constrained simulation (hemibrain central complex)\n")
        fh.write(f"neurons={n} internal_connections={len(e)}\n")
        fh.write(f"stimulated={seed_pref} ({len(seed_idx)} neurons) "
                 f"steps={args.steps} gain={args.gain}\n")
        fh.write("top downstream types by peak activity:\n")
        for tp in downstream:
            fh.write(f"  {tp}={peak[tp]:.4f}\n")

    print(f"\nWrote figures + tables to {OUT}")


if __name__ == "__main__":
    main()
