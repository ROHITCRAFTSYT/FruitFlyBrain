"""
One-command downloader for the FruitFlyBrain datasets.

Fetches both officially released connectome datasets into datasets/ so a fresh
clone is fully working. Every file is checked against the SHA-256 recorded in
datasets/README.md, so the numbers in the READMEs reproduce exactly.

The FlyWire annotations are fetched from a **pinned commit** of
flyconnectome/flywire_annotations. Its `main` branch keeps changing (on
2026-09-29 it gained Tastekin et al. 2026 annotations, which moves the census
from 8,840 to 8,846 cell types), so downloading from `main` silently changed
Project 02's results.

Idempotent: a file that already exists with the right checksum is skipped; one
with the wrong checksum is reported and re-downloaded.

    python download_datasets.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

HEMIBRAIN_URL = ("https://storage.googleapis.com/hemibrain/v1.2/"
                 "exported-traced-adjacencies-v1.2.tar.gz")
HEMIBRAIN_SHA256 = "07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7"

# flyconnectome/flywire_annotations commit 17fc577 (2026-05-04, "minor fixes for a
# few annotations"): the version every committed FlyWire result was computed from.
FLYWIRE_COMMIT = "17fc57722002e1a7d38cdd0c89ac382bf92718da"
FLYWIRE_BASE = ("https://raw.githubusercontent.com/flyconnectome/"
                f"flywire_annotations/{FLYWIRE_COMMIT}/supplemental_files")
FLYWIRE_FILES = {
    "README.md": "49e884a1273105427ded3c110d48266702a3f90148006cf02c1a0fa50919eb71",
    "Supplemental_file1_neuron_annotations.tsv":
        "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
    "Supplemental_file2_non_neuron_annotations.tsv":
        "2349fd5b4b9e53fd1b4a84aa73230bb7bebe66faa5d6a7490155aa2128861de1",
    "Supplemental_file3_summary_with_ngl_links.csv":
        "8e9d92c0263c19863da3d537400bc2fe5ab6ed19e8327b0237e1e8a60ef9cfa5",
}


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str) -> None:
    """Download `url` to `dest` unless it is already there with the right checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if sha256(dest) == expected_sha256:
            print(f"  skip (verified): {dest.name}")
            return
        print(f"  {dest.name} exists but its checksum does not match; re-downloading")
    print(f"  downloading {dest.name} ...")
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    got = sha256(part)
    if got != expected_sha256:
        part.unlink(missing_ok=True)
        raise SystemExit(f"\nChecksum mismatch for {dest.name}:\n  expected {expected_sha256}\n"
                         f"  got      {got}\nThe upstream file changed; see datasets/README.md.")
    os.replace(part, dest)
    print(f"    saved {dest.stat().st_size:,} bytes (SHA-256 verified)")


def main() -> None:
    print("== Janelia hemibrain v1.2 (CC BY 4.0) ==")
    tgz = DATASETS / "hemibrain" / "exported-traced-adjacencies-v1.2.tar.gz"
    fetch(HEMIBRAIN_URL, tgz, HEMIBRAIN_SHA256)
    extracted = DATASETS / "hemibrain" / "exported-traced-adjacencies-v1.2"
    if not extracted.exists():
        print("  extracting ...")
        with tarfile.open(tgz) as tf:
            # "data" filter: refuse absolute paths, "..", links outside the target, devices.
            tf.extractall(DATASETS / "hemibrain", filter="data")
    else:
        print("  skip (extracted)")

    print(f"\n== FlyWire FAFB annotations (CC BY 4.0), commit {FLYWIRE_COMMIT[:7]} ==")
    for name, digest in FLYWIRE_FILES.items():
        fetch(f"{FLYWIRE_BASE}/{name}", DATASETS / "flywire_fafb" / name, digest)

    print("\nDone. Datasets ready in", DATASETS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
