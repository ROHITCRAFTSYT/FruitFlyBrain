"""
One-command downloader for the FruitFlyBrain datasets.

Fetches both officially released connectome datasets into datasets/ so a fresh
clone is fully working. Idempotent: skips files that already exist with the
right size. See datasets/README.md for provenance, licenses and checksums.

    python download_datasets.py
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

HEMIBRAIN_URL = ("https://storage.googleapis.com/hemibrain/v1.2/"
                 "exported-traced-adjacencies-v1.2.tar.gz")
FLYWIRE_BASE = ("https://raw.githubusercontent.com/flyconnectome/"
                "flywire_annotations/main/supplemental_files")
FLYWIRE_FILES = [
    "README.md",
    "Supplemental_file1_neuron_annotations.tsv",
    "Supplemental_file2_non_neuron_annotations.tsv",
    "Supplemental_file3_summary_with_ngl_links.csv",
]


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    print(f"    saved {dest.stat().st_size:,} bytes")


def main() -> None:
    print("== Janelia hemibrain v1.2 (CC BY 4.0) ==")
    tgz = DATASETS / "hemibrain" / "exported-traced-adjacencies-v1.2.tar.gz"
    fetch(HEMIBRAIN_URL, tgz)
    extracted = DATASETS / "hemibrain" / "exported-traced-adjacencies-v1.2"
    if not extracted.exists():
        print("  extracting ...")
        with tarfile.open(tgz) as tf:
            tf.extractall(DATASETS / "hemibrain")
    else:
        print("  skip (extracted)")

    print("\n== FlyWire FAFB annotations (CC BY 4.0) ==")
    for f in FLYWIRE_FILES:
        fetch(f"{FLYWIRE_BASE}/{f}", DATASETS / "flywire_fafb" / f)

    print("\nDone. Datasets ready in", DATASETS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
