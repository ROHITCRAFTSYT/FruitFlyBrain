"""Projects 01-05: graph statistics, census, pathway search, dynamics, hub lookup."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conftest import load_script

p01 = load_script("01_connectome_graph_analysis/analyze.py")
p02 = load_script("02_neurotransmitter_atlas/atlas.py")
p03 = load_script("03_shortest_path_circuits/trace.py")
p04 = load_script("04_connectome_neural_network/simulate.py")
p05 = load_script("05_navis_3d_viewer/view_neurons.py")


# --- 01: graph statistics -----------------------------------------------------
def test_graph_density():
    assert p01.graph_density(7, 10) == pytest.approx(10 / 42)
    assert p01.graph_density(2, 2) == 1.0


def test_edge_reciprocity(tiny_hemibrain):
    conn, _ = tiny_hemibrain
    # reciprocal pairs 1<->4, 2<->4, 5<->6: 6 of 10 edges have a reverse edge
    assert p01.edge_reciprocity(conn) == pytest.approx(0.6)


def test_degree_table(tiny_hemibrain):
    conn, neurons = tiny_hemibrain
    st = p01.degree_table(conn, neurons)
    apl = st.loc[4]
    assert (apl.out_degree, apl.in_degree, apl.out_strength, apl.in_strength) == (3, 2, 26, 28)
    assert apl.total_degree == 5 and apl.total_strength == 54 and apl.type == "APL"
    leaf = st.loc[7]                         # only an input, no type
    assert (leaf.out_degree, leaf.in_degree, leaf.total_strength) == (0, 1, 1) and pd.isna(leaf.type)
    assert st["out_strength"].sum() == st["in_strength"].sum() == conn["weight"].sum()
    assert st.sort_values("total_strength", ascending=False).index[0] == 4   # the hub


# --- 02: census -----------------------------------------------------------------
def test_census_counts_and_missing_values(tiny_flywire):
    c = p02.census(tiny_flywire)
    assert c["super_class"].to_dict() == {"central": 2, "optic": 2, "unassigned": 1, "descending": 1}
    assert c["neurotransmitter"].to_dict() == {"acetylcholine": 3, "gaba": 1, "glutamate": 1, "unknown": 1}
    assert c["side"].sum() == 6 and c["side"]["unknown"] == 1
    assert c["top_cell_types"].iloc[0] == 2 and c["top_cell_types"].index[0] == "KCg-m"
    assert c["top_cell_types"].sum() == 5             # one neuron has no cell type


def test_nt_by_super_class(tiny_flywire):
    ct = p02.nt_by_super_class(tiny_flywire)
    assert ct.loc["central", "acetylcholine"] == 2
    assert ct.loc["unassigned", "unknown"] == 1
    assert int(ct.to_numpy().sum()) == len(tiny_flywire)


# --- 03: strongest pathway ---------------------------------------------------------
def test_type_bodyids_is_case_insensitive_prefix(tiny_hemibrain):
    _, neurons = tiny_hemibrain
    assert sorted(p03.type_bodyids(neurons, "kc")) == [1, 2]
    assert p03.type_bodyids(neurons, "MBON") == [3]
    assert p03.type_bodyids(neurons, "XYZ") == []


def test_build_cost_graph_drops_weak_edges(tiny_hemibrain):
    conn, _ = tiny_hemibrain
    G = p03.build_cost_graph(conn, min_weight=5)
    assert G.number_of_edges() == 7 and not G.has_edge(2, 3) and not G.has_edge(3, 7)
    assert G[1][3]["cost"] == pytest.approx(1 / 12) and G[1][3]["weight"] == 12


def test_strongest_path_prefers_heavy_synapses(tiny_hemibrain):
    conn, neurons = tiny_hemibrain
    G = p03.build_cost_graph(conn, min_weight=5)
    p03.add_terminals(G, set(p03.type_bodyids(neurons, "KC")), set(p03.type_bodyids(neurons, "MBON")))
    # 1->3 costs 1/12; the detours via APL (1->4->3: 1/20 + 1/5) cost more
    assert p03.strongest_path(G) == [1, 3]


def test_strongest_path_raises_when_unreachable(tiny_hemibrain):
    conn, _ = tiny_hemibrain
    G = p03.build_cost_graph(conn, min_weight=13)      # keeps 1->4, 4->1, 5->6 only
    p03.add_terminals(G, {1, 2}, {3})
    with pytest.raises(nx.NetworkXNoPath):
        p03.strongest_path(G)


# --- 04: connectome-constrained dynamics -----------------------------------------
def test_select_subset_and_weight_matrix(tiny_hemibrain):
    conn, neurons = tiny_hemibrain
    sub = p04.select_subset(neurons, ("ER", "EPG"))
    assert sub["bodyId"].tolist() == [5, 6]
    ids = sub["bodyId"].tolist()
    e = conn[conn["bodyId_pre"].isin(ids) & conn["bodyId_post"].isin(ids)]
    W = p04.weight_matrix(e, ids)
    # W[i, j] = synapses from j onto i
    assert W.dtype == np.float32 and W.tolist() == [[0.0, 2.0], [30.0, 0.0]]


def test_normalize_by_max_row():
    W = np.array([[0.0, 2.0], [30.0, 0.0]], dtype=np.float32)
    Wn = p04.normalize_by_max_row(W)
    assert Wn.sum(axis=1).max() == pytest.approx(1.0)
    Z = np.zeros((3, 3), dtype=np.float32)
    assert not np.isnan(p04.normalize_by_max_row(Z)).any()      # no divide-by-zero


def test_run_dynamics_propagates_a_pulse():
    Wn = np.array([[0.0, 2.0], [30.0, 0.0]], dtype=np.float32) / 30.0
    hist = p04.run_dynamics(Wn, np.array([1.0, 0.0], dtype=np.float32), steps=3, stim_steps=1, gain=1.0)
    np.testing.assert_allclose(hist, [[1.0, 0.0], [0.0, 1.0], [2 / 30, 0.0]], rtol=1e-6)
    assert (hist >= 0).all()                                     # ReLU


# --- 05: hub lookup ------------------------------------------------------------------
def test_hub_bodyids_reads_project01_output(tmp_path):
    csv = tmp_path / "top_hubs.csv"
    pd.DataFrame({"bodyId": [425790257, 5813105172, 7], "type": ["APL", "DPM", "x"]}).to_csv(csv, index=False)
    assert p05.hub_bodyids(2, csv) == [425790257, 5813105172]


def test_hub_bodyids_falls_back_without_project01(tmp_path):
    assert p05.hub_bodyids(2, tmp_path / "missing.csv") == p05.FALLBACK_HEMIBRAIN_BODYIDS[:2]
