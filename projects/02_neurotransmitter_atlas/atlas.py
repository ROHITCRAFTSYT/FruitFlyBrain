"""
Project 2 - Whole-brain neurotransmitter & cell-type atlas (FlyWire FAFB)
========================================================================

The FlyWire consortium proofread and annotated *every* neuron in a complete
adult female fly brain. This project turns that ~140k-neuron catalogue into a
census:

  * how many neurons of each super-class (central brain, optic lobe, sensory,
    ascending, descending, ...)
  * the predicted fast neurotransmitter mix across the brain
  * left / right / center symmetry
  * the most numerous cell types
  * a neurotransmitter x super-class cross-tabulation

Data: Schlegel et al. 2024, Nature - FlyWire annotations (CC-BY-4.0).
Outputs: PNG figures + CSV tables in ./outputs/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import FLYWIRE_NEURONS, outdir, require

OUT = outdir(__file__)

NT_COLORS = {
    "acetylcholine": "#e15759", "gaba": "#4e79a7", "glutamate": "#59a14f",
    "dopamine": "#f28e2b", "serotonin": "#b07aa1", "octopamine": "#76b7b2",
    "unknown": "#bab0ac",
}


def main() -> None:
    require(FLYWIRE_NEURONS)
    print("Loading FlyWire neuron annotations (~140k neurons)...")
    df = pd.read_csv(FLYWIRE_NEURONS, sep="\t", low_memory=False)
    print(f"  loaded {len(df):,} annotated neurons, {df.shape[1]} attributes")

    # ---- super-class census -------------------------------------------------
    sc = df["super_class"].fillna("unassigned").value_counts()
    sc.to_csv(OUT / "counts_by_super_class.csv", header=["n_neurons"])

    # ---- neurotransmitter mix ----------------------------------------------
    nt = df["top_nt"].fillna("unknown").str.lower().value_counts()
    nt.to_csv(OUT / "counts_by_neurotransmitter.csv", header=["n_neurons"])

    # ---- side symmetry ------------------------------------------------------
    side = df["side"].fillna("unknown").value_counts()

    # ---- top cell types -----------------------------------------------------
    top_types = df["cell_type"].dropna().value_counts().head(25)
    top_types.to_csv(OUT / "top_cell_types.csv", header=["n_neurons"])

    print("\n===== FlyWire whole-brain census =====")
    print(f"  neurons                : {len(df):,}")
    print(f"  super-classes          : {df['super_class'].nunique()}")
    print(f"  distinct cell types    : {df['cell_type'].nunique():,}")
    print("  neurotransmitter mix   :")
    for k, v in nt.items():
        print(f"      {k:<14} {v:>7,}  ({100*v/len(df):4.1f}%)")

    # ---- figure 1: super-class bar -----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sc.sort_values().plot.barh(ax=ax, color="#4e79a7")
    ax.set_xlabel("number of neurons")
    ax.set_title("FlyWire adult brain: neurons per super-class")
    fig.tight_layout()
    fig.savefig(OUT / "super_class_census.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: neurotransmitter pie ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [NT_COLORS.get(k, "#bab0ac") for k in nt.index]
    ax.pie(nt.values, labels=[f"{k}\n{v:,}" for k, v in nt.items()],
           colors=colors, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
    ax.set_title("Predicted fast neurotransmitter across the whole fly brain")
    fig.tight_layout()
    fig.savefig(OUT / "neurotransmitter_mix.png", dpi=140)
    plt.close(fig)

    # ---- figure 3: top cell types ------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    top_types.sort_values().plot.barh(ax=ax, color="#59a14f")
    ax.set_xlabel("number of neurons")
    ax.set_title("25 most numerous cell types (FlyWire)")
    fig.tight_layout()
    fig.savefig(OUT / "top_cell_types.png", dpi=140)
    plt.close(fig)

    # ---- figure 4: NT x super-class heatmap --------------------------------
    ct = (
        df.assign(nt=df["top_nt"].fillna("unknown").str.lower(),
                  scl=df["super_class"].fillna("unassigned"))
        .pivot_table(index="scl", columns="nt", values="root_id",
                     aggfunc="count", fill_value=0)
    )
    ct.to_csv(OUT / "nt_by_super_class.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(ct.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_title("Neurotransmitter usage by super-class (neuron counts)")
    fig.colorbar(im, ax=ax, label="neurons")
    fig.tight_layout()
    fig.savefig(OUT / "nt_by_super_class.png", dpi=140)
    plt.close(fig)

    with open(OUT / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("FlyWire FAFB whole-brain census\n")
        fh.write(f"total neurons={len(df)}\n")
        fh.write(f"super_classes={df['super_class'].nunique()} "
                 f"cell_types={df['cell_type'].nunique()}\n")
        fh.write("side counts:\n")
        for k, v in side.items():
            fh.write(f"  {k}={v}\n")

    print(f"\nWrote figures + tables to {OUT}")


if __name__ == "__main__":
    main()
