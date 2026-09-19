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
