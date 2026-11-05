"""Data loading: shared helpers in projects/_common.py and the dataset downloader."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import _common
import download_datasets as DL


@pytest.mark.parametrize("name, family", [
    ("MBON01", "MBON"),
    ("KCg-s2", "KC"),
    ("PEN_a(PEN1)", "PEN"),
    ("PVLP011", "PVLP"),
    ("FS4C", "FS"),
    ("Delta7", "DELTA"),      # no leading 2-letter capitals: first alpha token, upper-cased
    ("", "other"),
    (None, "other"),
    (float("nan"), "other"),
    ("123", "other"),
])
def test_type_family(name, family):
    assert _common.type_family(name) == family


def test_require_passes_existing_and_fails_missing(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a\n1\n")
    assert _common.require(f) == f
    with pytest.raises(SystemExit, match="Missing dataset file"):
        _common.require(tmp_path / "missing.csv")


def test_outdir_creates_outputs_next_to_script(tmp_path):
    script = tmp_path / "proj" / "run.py"
    script.parent.mkdir()
    out = _common.outdir(str(script))
    assert out == script.parent / "outputs" and out.is_dir()


def test_load_hemibrain_reads_both_tables_and_adds_family(monkeypatch, hemibrain_paths):
    conn_p, neur_p = hemibrain_paths
    monkeypatch.setattr(_common, "HEMIBRAIN_CONNECTIONS", conn_p)
    monkeypatch.setattr(_common, "HEMIBRAIN_NEURONS", neur_p)
    conn, neurons = _common.load_hemibrain()
    assert list(conn.columns) == ["bodyId_pre", "bodyId_post", "weight"]
    assert len(conn) == 10 and int(conn["weight"].sum()) == 103
    assert neurons.set_index("bodyId")["family"].to_dict() == {
        1: "KC", 2: "KC", 3: "MBON", 4: "APL", 5: "ER", 6: "EPG", 7: "other"}


def test_load_hemibrain_missing_files_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(_common, "HEMIBRAIN_CONNECTIONS", tmp_path / "nope.csv")
    with pytest.raises(SystemExit):
        _common.load_hemibrain()


# --- download_datasets.py ---------------------------------------------------
def test_pinned_flywire_source_and_checksums_are_well_formed():
    assert len(DL.FLYWIRE_COMMIT) == 40 and DL.FLYWIRE_COMMIT in DL.FLYWIRE_BASE
    assert "/main/" not in DL.FLYWIRE_BASE          # main moves; results must not
    for digest in [DL.HEMIBRAIN_SHA256, *DL.FLYWIRE_FILES.values()]:
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "blob.bin"
    data = bytes(range(256)) * 9000          # > 1 MiB chunk boundary
    f.write_bytes(data)
    assert DL.sha256(f) == hashlib.sha256(data).hexdigest()


def test_fetch_verifies_skips_and_rejects(tmp_path):
    src = tmp_path / "upstream.tsv"
    src.write_text("root_id\tcell_type\n1\tKC\n")
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "data" / "neurons.tsv"

    DL.fetch(src.as_uri(), dest, good)                    # downloads + verifies
    assert dest.read_bytes() == src.read_bytes()
    mtime = dest.stat().st_mtime_ns
    DL.fetch(src.as_uri(), dest, good)                    # already there: skipped
    assert dest.stat().st_mtime_ns == mtime

    other = tmp_path / "data" / "other.tsv"
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        DL.fetch(src.as_uri(), other, "0" * 64)          # upstream changed: refuse
    assert not other.exists() and not other.with_name("other.tsv.part").exists()


def test_fixture_tables_have_real_column_names(tiny_hemibrain, tiny_flywire):
    conn, neurons = tiny_hemibrain
    assert {"bodyId", "type", "instance"} <= set(neurons.columns)
    assert {"root_id", "super_class", "top_nt", "side", "cell_type"} <= set(tiny_flywire.columns)
    assert isinstance(conn, pd.DataFrame)
