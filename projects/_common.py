"""
Shared helpers for the FruitFlyBrain example projects.

Resolves paths to the locally downloaded connectome datasets so every project
can be run from anywhere with `python <project>/<script>.py`.
"""
from __future__ import annotations

from pathlib import Path

# projects/_common.py -> projects/ -> FruitFlyBrain/
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

HEMIBRAIN_DIR = DATASETS / "hemibrain" / "exported-traced-adjacencies-v1.2"
HEMIBRAIN_NEURONS = HEMIBRAIN_DIR / "traced-neurons.csv"
HEMIBRAIN_CONNECTIONS = HEMIBRAIN_DIR / "traced-total-connections.csv"

FLYWIRE_DIR = DATASETS / "flywire_fafb"
FLYWIRE_NEURONS = FLYWIRE_DIR / "Supplemental_file1_neuron_annotations.tsv"


def require(path: Path) -> Path:
    """Fail loudly with guidance if a dataset file is missing."""
    if not path.exists():
        raise SystemExit(
            f"\nMissing dataset file:\n  {path}\n\n"
            "Re-download the datasets (see datasets/README.md) before running "
            "this project.\n"
        )
    return path


def outdir(script_file: str) -> Path:
    """Return (and create) an `outputs/` folder next to the calling script."""
    d = Path(script_file).resolve().parent / "outputs"
    d.mkdir(exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Shared helpers for the machine-learning projects (06-08)
# --------------------------------------------------------------------------
import re

_LEADING_UPPER = re.compile(r"^[A-Z]{2,}")
_LEADING_ALPHA = re.compile(r"[A-Za-z]{2,}")


def type_family(name: object) -> str:
    """
    Collapse a fine hemibrain type into a coarse cell-type *family*.

    Examples:
        MBON01     -> MBON
        KCg-s2     -> KC
        PEN_a(PEN1)-> PEN
        PVLP011    -> PVLP
        FS4C       -> FS
    Names without a clear uppercase code fall back to their first alpha token,
    else 'other'.
    """
    if not isinstance(name, str) or not name:
        return "other"
    m = _LEADING_UPPER.match(name)
    if m:
        return m.group(0)
    m = _LEADING_ALPHA.search(name)
    return m.group(0).upper() if m else "other"


def load_hemibrain():
    """Return (connections_df, neurons_df) with a 'family' column on neurons."""
    import pandas as pd
    require(HEMIBRAIN_CONNECTIONS)
    require(HEMIBRAIN_NEURONS)
    conn = pd.read_csv(HEMIBRAIN_CONNECTIONS)
    neurons = pd.read_csv(HEMIBRAIN_NEURONS)
    neurons["family"] = neurons["type"].map(type_family)
    return conn, neurons
