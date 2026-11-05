"""Projects 06-08: connectivity fingerprints, clustering metrics, link-prediction features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import load_script

p06 = load_script("06_celltype_classifier/train_classifier.py")
p07 = load_script("07_neuron_embeddings/embed_and_cluster.py")
p08 = load_script("08_synapse_link_prediction/link_prediction.py")


def _with_families(neurons):
    import _common
    neurons = neurons.copy()
    neurons["family"] = neurons["type"].map(_common.type_family)
    return neurons


# --- 06 ---------------------------------------------------------------------------
def test_target_families_threshold_and_excludes_other(tiny_hemibrain):
    _, neurons = tiny_hemibrain
    fam = _with_families(neurons)["family"]
    assert p06.target_families(fam, min_members=2) == ["KC"]
    assert set(p06.target_families(fam, min_members=1)) == {"KC", "MBON", "APL", "ER", "EPG"}
    assert p06.target_families(fam, min_members=3) == []


def test_partner_fraction_matrix_rows_are_synapse_fractions(tiny_hemibrain):
    conn, neurons = tiny_hemibrain
    neurons = _with_families(neurons)
    vocab = ["KC", "MBON", "APL", "other"]
    neurons["fam_feat"] = neurons["family"].where(neurons["family"].isin(vocab[:-1]), "other")
    out = p06.partner_fraction_matrix(conn, neurons, "bodyId_pre", "bodyId_post", vocab)
    assert list(out.columns) == vocab
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    # APL (4) sends 15 + 6 synapses to KCs and 5 to the MBON
    assert out.loc[4, "KC"] == pytest.approx(21 / 26) and out.loc[4, "MBON"] == pytest.approx(5 / 26)
    inp = p06.partner_fraction_matrix(conn, neurons, "bodyId_post", "bodyId_pre", vocab)
    assert inp.loc[3, "KC"] == pytest.approx(16 / 21) and inp.loc[3, "APL"] == pytest.approx(5 / 21)


# --- 07 ---------------------------------------------------------------------------
def test_adjacency_matrix(tiny_hemibrain):
    conn, neurons = tiny_hemibrain
    ids = neurons["bodyId"].to_numpy()
    A = p07.adjacency_matrix(conn, ids)
    assert A.shape == (7, 7) and A.nnz == 10 and A.sum() == conn["weight"].sum()
    assert A[0, 2] == 12 and A[3, 0] == 15 and A[2, 0] == 0          # directed


def test_cluster_purity():
    clusters = np.array([0, 0, 0, 1, 1])
    fams = np.array(["a", "a", "b", "c", "c"])
    assert p07.cluster_purity(clusters, fams) == pytest.approx((2 / 3 + 1) / 2)
    assert p07.cluster_purity(np.zeros(4), np.array(list("abcd"))) == pytest.approx(0.25)


# --- 08 ---------------------------------------------------------------------------
def _edge_keys(pairs, n):
    return np.sort(np.array([i * n + j for i, j in pairs], dtype=np.int64))


def test_sample_non_edges_never_returns_edges_or_self_loops():
    n = 6
    pairs = [(0, 1), (1, 0), (2, 3), (4, 5), (5, 4), (1, 2)]
    keys = _edge_keys(pairs, n)
    i, j = p08.sample_non_edges(n, keys, 200, np.random.default_rng(0))
    assert len(i) == len(j) == 200
    assert not (i == j).any()
    assert not np.isin(i.astype(np.int64) * n + j, keys).any()


def test_edge_features_shape_and_reciprocity_flag():
    n, dim = 4, 3
    rng = np.random.default_rng(1)
    emb_out, emb_in = rng.normal(size=(n, dim)), rng.normal(size=(n, dim))
    keys = _edge_keys([(0, 1), (1, 0), (2, 3)], n)
    strength = np.array([1.0, 2.0, 3.0, 4.0])
    deg = np.array([1, 1, 1, 0])
    si, tj = np.array([0, 2, 3]), np.array([1, 3, 2])
    X = p08.edge_features(si, tj, emb_out, emb_in, strength, strength, deg, deg, keys, n)
    assert X.shape == (3, dim + 5)
    np.testing.assert_allclose(X[:, :dim], emb_out[si] * emb_in[tj])        # Hadamard part
    # pair (0,1) has a reverse edge; (2,3) does not; (3,2) does (2->3 exists)
    assert X[:, dim + 4].tolist() == [1.0, 0.0, 1.0]
    np.testing.assert_allclose(X[:, dim], np.log1p(strength[si]))


def test_truncated_svd_is_deterministic():
    from scipy import sparse
    M = sparse.random(80, 80, density=0.1, random_state=0, format="csr")
    a, b = p08.TruncatedSVD_fit(M), p08.TruncatedSVD_fit(M)
    assert a.shape == (80, p08.EMB_DIM) and np.allclose(a, b)


def test_metrics_files_are_consistent_with_readme():
    """The committed metrics are the README headline numbers (no silent drift)."""
    import json

    from conftest import PROJECTS
    m06 = json.loads((PROJECTS / "06_celltype_classifier/outputs/metrics.json").read_text())
    m07 = json.loads((PROJECTS / "07_neuron_embeddings/outputs/metrics.json").read_text())
    m08 = json.loads((PROJECTS / "08_synapse_link_prediction/outputs/metrics.json").read_text())
    assert m06["random_forest"]["accuracy"] == pytest.approx(0.882, abs=1e-3)       # README: 88.2%
    assert m06["mlp"]["accuracy"] == pytest.approx(0.876, abs=1e-3)                 # README: 87.6%
    assert m06["baseline_majority"]["accuracy"] == pytest.approx(0.113, abs=1e-3)   # README: 11.3%
    assert m07["normalized_mutual_info"] == pytest.approx(0.59, abs=5e-3) and m07["n_clusters"] == 53
    assert m08["roc_auc"] == pytest.approx(0.971, abs=5e-4)
    assert m08["pr_auc_average_precision"] == pytest.approx(0.968, abs=5e-4)
    assert isinstance(pd.Series(m08), pd.Series)
