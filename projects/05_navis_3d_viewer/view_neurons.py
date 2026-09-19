"""
Project 5 - 3D neuron morphology viewer (NAVis)
===============================================

Renders the actual 3D shape of fly-brain neurons -- dendrites, axons and all --
using NAVis (Neuron Analysis and Visualization).

Two data sources:

  * example  (default, zero config): real *Drosophila* hemibrain skeletons that
    ship bundled inside NAVis. No download, no login.

  * neuprint (--neuprint): fetch specific neurons by bodyId straight from the
    Janelia hemibrain server. Needs a FREE token from
    https://neuprint.janelia.org (log in -> Account -> Auth Token), exported as:
        setx NEUPRINT_TOKEN "your.token.here"     (Windows, new shell after)
        export NEUPRINT_TOKEN="your.token.here"   (macOS/Linux)
    With no --bodyids given it reads the top hub neurons straight from Project
    1's output (projects/01_.../outputs/top_hubs.csv), so this ties the viewer
    back to the connectome analysis. Use --n-hubs to change how many.

Each run writes:
  * a static multi-view PNG   (outputs/neurons_3d.png)
  * an INTERACTIVE 3D HTML    (outputs/neurons_3d.html)  <- open in a browser,
    then drag to rotate, scroll to zoom.

Examples
--------
    python view_neurons.py                       # 5 bundled example neurons
    python view_neurons.py --n 3
    python view_neurons.py --neuprint            # top hubs from Project 1 (needs token)
    python view_neurons.py --neuprint --bodyids 612371421 300972942
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT, outdir

OUT = outdir(__file__)

# Project 1 writes its ranked hub table here.
TOP_HUBS_CSV = ROOT / "projects" / "01_connectome_graph_analysis" / "outputs" / "top_hubs.csv"
# Fallback if Project 1 hasn't been run yet (APL, DPM, MBON01).
FALLBACK_HEMIBRAIN_BODYIDS = [425790257, 5813105172, 612371421]


def hub_bodyids(n: int) -> list[int]:
    """Read the top-N hub bodyIds from Project 1's output, else use the fallback."""
    if not TOP_HUBS_CSV.exists():
        print(f"  (Project 1 output not found at {TOP_HUBS_CSV};\n"
              f"   run analyze.py first for live hubs - using built-in fallback.)")
        return FALLBACK_HEMIBRAIN_BODYIDS[:n]
    import pandas as pd
    hubs = pd.read_csv(TOP_HUBS_CSV)
    ids = hubs["bodyId"].head(n).astype("int64").tolist()
    labels = ", ".join(f"{b} ({t})" for b, t in zip(ids, hubs["type"].head(n)))
    print(f"  top {n} hubs from Project 1: {labels}")
    return ids
PALETTE = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1",
           "#76b7b2", "#edc948", "#ff9da7"]


def load_example(n: int):
    import navis
    nl = navis.example_neurons(n)
    # NAVis may return a single neuron; always work with a NeuronList.
    return navis.NeuronList(nl)


def load_neuprint(bodyids: list[int]):
    import navis
    import navis.interfaces.neuprint as neu

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit(
            "\n--neuprint needs a free auth token.\n"
            "  1. Sign in at https://neuprint.janelia.org\n"
            "  2. Account menu (top-right) -> copy 'Auth Token'\n"
            "  3. setx NEUPRINT_TOKEN \"<token>\"   (then open a NEW terminal)\n"
            "Then re-run.  (Or just run without --neuprint to use bundled neurons.)\n"
        )
    client = neu.Client("neuprint.janelia.org", dataset="hemibrain:v1.2", token=token)
    print(f"  fetching {len(bodyids)} skeleton(s) from neuPrint...")
    nl = neu.fetch_skeletons(bodyids, client=client)
    return navis.NeuronList(nl)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuprint", action="store_true",
                    help="fetch real hemibrain neurons by bodyId (needs NEUPRINT_TOKEN)")
    ap.add_argument("--bodyids", type=int, nargs="+", default=None,
                    help="explicit hemibrain bodyIds (default: top hubs from Project 1)")
    ap.add_argument("--n-hubs", type=int, default=3,
                    help="how many top hubs to pull from Project 1 (when --bodyids omitted)")
    ap.add_argument("--n", type=int, default=5, help="how many bundled example neurons")
    args = ap.parse_args()

    try:
        import navis  # noqa: F401
    except ImportError:
        raise SystemExit("NAVis is not installed. Run:  pip install navis plotly\n")
    import navis

    print(f"NAVis {navis.__version__} - loading neurons...")
    if args.neuprint:
        bodyids = args.bodyids if args.bodyids is not None else hub_bodyids(args.n_hubs)
        nl = load_neuprint(bodyids)
        source = "neuPrint hemibrain:v1.2"
    else:
        nl = load_example(args.n)
        source = "NAVis bundled example neurons (hemibrain)"

    colors = {n.id: PALETTE[i % len(PALETTE)] for i, n in enumerate(nl)}

    print(f"\n===== Loaded {len(nl)} neuron(s) from {source} =====")
    for i, n in enumerate(nl):
        try:
            cable = n.cable_length / 1000.0  # nm -> um
            print(f"  [{i}] id={n.id}  nodes={n.n_nodes:,}  "
                  f"branches={n.n_branches:,}  cable={cable:,.1f} um")
        except Exception:
            print(f"  [{i}] id={n.id}")

    # ---- static multi-view PNG (matplotlib 3D) ------------------------------
    print("\nRendering static PNG (3 viewing angles)...")
    fig = plt.figure(figsize=(15, 5))
    views = [("frontal", (-90, -90)), ("dorsal", (0, -90)), ("lateral", (0, 0))]
    for j, (name, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        navis.plot2d(nl, method="3d_complex", ax=ax, color=colors, linewidth=0.75)
        ax.set_title(name)
        ax.azim, ax.elev = azim, elev
        ax.set_axis_off()
    fig.suptitle(f"Fly-brain neuron morphology - {source}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "neurons_3d.png", dpi=140)
    plt.close(fig)

    # ---- interactive 3D HTML (plotly) ---------------------------------------
    print("Rendering interactive 3D HTML (open in a browser, drag to rotate)...")
    try:
        pfig = navis.plot3d(nl, backend="plotly", color=colors, inline=False)
        pfig.update_layout(title=f"Interactive fly neurons - {source}")
        pfig.write_html(str(OUT / "neurons_3d.html"), include_plotlyjs="cdn")
    except Exception as e:  # plotly missing or backend issue -> PNG still exists
        print(f"  (interactive HTML skipped: {e})")

    print(f"\nWrote:\n  {OUT / 'neurons_3d.png'}\n  {OUT / 'neurons_3d.html'}")
    print("Open the .html file in any browser to rotate the neurons in 3D.")


if __name__ == "__main__":
    main()
