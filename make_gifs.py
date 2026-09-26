"""
Build the animated GIFs shown in the README -- one per project, from the
projects' real data and code (nothing is mocked up).

    .venv\\Scripts\\python make_gifs.py                    # projects 01-08 + FlyLab training curves
    .venv-sim\\Scripts\\python make_gifs.py flylab_videos  # FlyLab brain films (needs imageio)
    .venv\\Scripts\\python make_gifs.py p04 p07            # just some

Output: docs/gifs/*.gif
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "gifs"
PROJ = ROOT / "projects"
sys.path.insert(0, str(PROJ))
DPI = 80
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})


def save(anim, name, fps):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gif"
    anim.save(path, writer=PillowWriter(fps=fps), dpi=DPI)
    plt.close(anim._fig)
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)", flush=True)


def frames_with_hold(n, hold):
    """Frame indices 0..n-1, then repeat the last frame `hold` times."""
    return list(range(n)) + [n - 1] * hold


def hemibrain():
    from _common import load_hemibrain
    return load_hemibrain()


# --- 01: the connectome backbone assembling -------------------------------
def p01():
    import networkx as nx
    conn, neurons = hemibrain()
    strength = (conn.groupby("bodyId_pre")["weight"].sum()
                .add(conn.groupby("bodyId_post")["weight"].sum(), fill_value=0))
    hubs = set(strength.sort_values(ascending=False).head(120).index)
    e = conn[conn.bodyId_pre.isin(hubs) & conn.bodyId_post.isin(hubs) & (conn.weight >= 20)]
    e = e.sort_values("weight", ascending=False).head(900)
    G = nx.from_pandas_edgelist(e, "bodyId_pre", "bodyId_post", edge_attr="weight")
    pos = nx.spring_layout(G, seed=3, k=0.35, weight=None)
    types = neurons.set_index("bodyId")["type"].to_dict()
    top = strength.loc[list(G.nodes)].sort_values(ascending=False).index[:6]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    steps = 30
    cut = np.linspace(1, len(e), steps).astype(int)

    def draw(i):
        ax.clear(); ax.axis("off")
        sub = e.iloc[:cut[i]]
        deg = sub.groupby("bodyId_pre")["weight"].sum().add(
            sub.groupby("bodyId_post")["weight"].sum(), fill_value=0)
        for a, b, w in sub[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], color="#3c887e", alpha=min(0.9, 0.08 + w / 900), lw=0.6)
        xs = [pos[n][0] for n in G.nodes]; ys = [pos[n][1] for n in G.nodes]
        sz = [8 + 0.004 * deg.get(n, 0) for n in G.nodes]
        ax.scatter(xs, ys, s=sz, c="#d1495b", alpha=0.85, zorder=3)
        if i > steps * 0.6:
            for n in top:
                ax.annotate(types.get(n, "?"), pos[n], fontsize=9, weight="bold", zorder=4)
        ax.set_title(f"Project 01 - hemibrain backbone: strongest {cut[i]} of {len(e)} "
                     f"hub-to-hub connections\n(120 biggest hubs; node size = synapses so far)",
                     fontsize=10)
    save(FuncAnimation(fig, draw, frames=frames_with_hold(steps, 12)), "p01_connectome_backbone", 8)


# --- 02: whole-brain census counting up -----------------------------------
def p02():
    import pandas as pd
    from _common import FLYWIRE_NEURONS
    df = pd.read_csv(FLYWIRE_NEURONS, sep="\t", usecols=["super_class", "top_nt"], low_memory=False)
    sc = df["super_class"].fillna("unassigned").value_counts().sort_values()
    nt = df["top_nt"].fillna("unknown").str.lower().value_counts().sort_values()
    colors = {"acetylcholine": "#e15759", "gaba": "#4e79a7", "glutamate": "#59a14f",
              "dopamine": "#f28e2b", "serotonin": "#b07aa1", "octopamine": "#76b7b2"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    steps = 25

    def draw(i):
        f = (i + 1) / steps
        for a in ax:
            a.clear()
        ax[0].barh(sc.index, sc.values * f, color="#4e79a7")
        ax[0].set_xlim(0, sc.max() * 1.05); ax[0].set_title("neurons per super-class")
        ax[1].barh(nt.index, nt.values * f, color=[colors.get(k, "#bab0ac") for k in nt.index])
        ax[1].set_xlim(0, nt.max() * 1.05); ax[1].set_title("predicted neurotransmitter")
        for a in ax:
            a.tick_params(labelsize=8)
        fig.suptitle(f"Project 02 - FlyWire whole-brain census: {int(len(df) * f):,} of "
                     f"{len(df):,} neurons counted", fontsize=11, weight="bold")
        fig.tight_layout()
    save(FuncAnimation(fig, draw, frames=frames_with_hold(steps, 12)), "p02_brain_census", 10)


# --- 03: a signal pulse travelling the strongest KC -> MBON routes --------
def p03():
    import networkx as nx
    conn, neurons = hemibrain()
    id2type = dict(zip(neurons.bodyId, neurons.type.fillna("?")))
    up = neurons.type.fillna("").str.upper()
    src = set(neurons.bodyId[up.str.startswith("KC")]); dst = set(neurons.bodyId[up.str.startswith("MBON")])
    strong = conn[conn.weight >= 5].copy(); strong["cost"] = 1.0 / strong.weight
    G = nx.from_pandas_edgelist(strong, "bodyId_pre", "bodyId_post", ["weight", "cost"], create_using=nx.DiGraph)
    for b in src & set(G):
        G.add_edge("S", b, cost=0.0, weight=0)
    for b in dst & set(G):
        G.add_edge(b, "T", cost=0.0, weight=0)
    paths, seen = [], set()
    for p in nx.shortest_simple_paths(G, "S", "T", weight="cost"):
        core = p[1:-1]
        key = tuple(id2type[x] for x in core)
        if key in seen:
            continue
        seen.add(key); paths.append(core)
        if len(paths) == 5:
            break
    fig, ax = plt.subplots(figsize=(9, 4.8))
    lanes = []
    for k, p in enumerate(paths):
        y = len(paths) - k
        lanes.append([((j, y), n) for j, n in enumerate(p)])
    per_hop = 6
    total = sum(max(1, len(p) - 1) * per_hop for p in paths)

    def draw(i):
        ax.clear(); ax.axis("off")
        t = i
        for k, lane in enumerate(lanes):
            hops = max(1, len(lane) - 1)
            prog = np.clip(t / per_hop, 0, hops)
            t -= hops * per_hop
            for j, ((x, y), n) in enumerate(lane):
                lit = j <= prog
                ax.scatter(x, y, s=900, c="#f2b134" if lit else "#dddddd", edgecolors="#555", zorder=3)
                ax.text(x, y, id2type[n], ha="center", va="center", fontsize=7, zorder=4)
                if j < len(lane) - 1:
                    (x2, y2), n2 = lane[j + 1]
                    w = int(G[n][n2]["weight"])
                    ax.annotate("", (x2 - 0.18, y2), (x + 0.18, y),
                                arrowprops=dict(arrowstyle="->", lw=1 + min(w, 600) / 60,
                                                color="#d1495b" if j + 1 <= prog else "#bbbbbb"))
                    ax.text((x + x2) / 2, y + 0.18, f"{w} syn", ha="center", fontsize=7)
                    if j < prog < j + 1:
                        f = prog - j
                        ax.scatter(x + f * (x2 - x), y, s=160, c="#d1495b", zorder=5)
        ax.set_xlim(-0.6, max(len(l) for l in lanes) - 0.4); ax.set_ylim(0.4, len(lanes) + 0.6)
        ax.set_title("Project 03 - a signal travelling the strongest Kenyon-cell -> MBON routes\n"
                     "(the mushroom-body learning circuit; shortest path with cost = 1/synapses)", fontsize=10)
    save(FuncAnimation(fig, draw, frames=frames_with_hold(total + 1, 12)), "p03_signal_pathways", 10)


# --- 04: activity cascade through the central complex ---------------------
def p04():
    conn, neurons = hemibrain()
    pref = ("EPG", "PEN", "PEG", "ER", "EL", "FB", "PB", "DELTA7", "FC", "FS", "PFN")
    sub = neurons[neurons.type.fillna("").str.upper().str.startswith(pref)]
    ids = sub.bodyId.tolist(); idx = {b: i for i, b in enumerate(ids)}
    e = conn[conn.bodyId_pre.isin(idx) & conn.bodyId_post.isin(idx)]
    n = len(ids)
    W = np.zeros((n, n), np.float32)
    np.add.at(W, (e.bodyId_post.map(idx).values, e.bodyId_pre.map(idx).values), e.weight.values.astype(np.float32))
    W /= W.sum(1).max()
    types = sub.type.fillna("?").values
    fam = np.array([t.split("_")[0][:4] for t in types])
    seed = np.array([t.upper().startswith("ER") for t in types])
    steps, x, hist = 16, np.zeros(n, np.float32), []
    for t in range(steps):
        x = np.maximum(0, 0.9 * W @ x + (seed * 1.0 if t < 5 else 0)); hist.append(x.copy())
    hist = np.array(hist)
    order = np.lexsort((np.argmax(hist, 0), fam))
    groups = ["ER", "EL", "EPG", "PEN", "PEG", "PFN", "FB", "FS"]
    curves = {g: hist[:, np.char.startswith(np.char.upper(fam), g)].mean(1) for g in groups
              if np.any(np.char.startswith(np.char.upper(fam), g))}
    side = int(np.ceil(np.sqrt(n)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))

    def draw(t):
        for a in ax:
            a.clear()
        img = np.zeros(side * side); img[:n] = hist[t][order]
        ax[0].imshow(img.reshape(side, side), cmap="inferno", norm=PowerNorm(0.35, vmin=0, vmax=hist.max()))
        ax[0].axis("off"); ax[0].set_title(f"{n:,} central-complex neurons, step {t}")
        for g, c in curves.items():
            ax[1].plot(c[:t + 1], lw=2, label=g)
        ax[1].set_xlim(0, steps - 1); ax[1].set_ylim(0, max(c.max() for c in curves.values()) * 1.1)
        ax[1].axvspan(0, 4, color="grey", alpha=0.15)
        ax[1].set_xlabel("time step"); ax[1].set_ylabel("mean activity"); ax[1].legend(fontsize=7, ncol=2)
        ax[1].set_title("which cell types light up")
        fig.suptitle("Project 04 - stimulating ER ring neurons: activity cascades through the real wiring "
                     "(ER -> EL/EPG compass)", fontsize=10, weight="bold")
        fig.tight_layout()
    save(FuncAnimation(fig, draw, frames=frames_with_hold(steps, 10)), "p04_activity_cascade", 5)


# --- 05: real neurons rotating in 3-D ---------------------------------------
def p05():
    import navis
    nl = navis.NeuronList(navis.example_neurons(5))
    nl = navis.downsample_neuron(nl, 4, inplace=False)
    cols = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"]
    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for n, c in zip(nl, cols):
        nodes = n.nodes.set_index("node_id")
        for nid, par in zip(n.nodes.node_id, n.nodes.parent_id):
            if par >= 0:
                a, b = nodes.loc[nid], nodes.loc[par]
                ax.plot([a.x, b.x], [a.y, b.y], [a.z, b.z], color=c, lw=0.6)
    ax.set_axis_off()
    ax.set_title("Project 05 - five real hemibrain neurons (NAVis)", fontsize=10)

    def draw(i):
        ax.view_init(elev=-70 + 20 * np.sin(i / 36 * 2 * np.pi), azim=-90 + i * 10)
    save(FuncAnimation(fig, draw, frames=36), "p05_neurons_3d", 10)


# --- 06: the cell-type classifier's neural net training --------------------
def p06():
    import importlib.util
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    spec = importlib.util.spec_from_file_location("tc", PROJ / "06_celltype_classifier" / "train_classifier.py")
    tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
    conn, neurons = hemibrain()
    fc = neurons.family.value_counts()
    kept = [f for f in fc.index if f != "other" and fc[f] >= tc.MIN_MEMBERS]
    vocab = kept + ["other"]
    neurons["fam_feat"] = neurons.family.where(neurons.family.isin(kept), "other")
    X = (tc.partner_fraction_matrix(conn, neurons, "bodyId_pre", "bodyId_post", vocab).add_prefix("o")
         .join(tc.partner_fraction_matrix(conn, neurons, "bodyId_post", "bodyId_pre", vocab).add_prefix("i"),
               how="outer").fillna(0.0))
    y = neurons.set_index("bodyId").family.reindex(X.index)
    m = y.isin(kept).values
    X, y = X[m].to_numpy(dtype=float), y[m].astype(str).to_numpy(dtype=str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, learning_rate_init=1e-3)
    classes = np.unique(ytr)
    loss, acc_tr, acc_te = [], [], []
    rng = np.random.default_rng(0)
    for ep in range(40):
        p = rng.permutation(len(Xtr))
        for k in range(0, len(p), 256):
            mlp.partial_fit(Xtr[p[k:k + 256]], ytr[p[k:k + 256]], classes=classes)
        loss.append(mlp.loss_); acc_tr.append(mlp.score(Xtr, ytr)); acc_te.append(mlp.score(Xte, yte))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

    def draw(i):
        for a in ax:
            a.clear()
        ep = np.arange(1, i + 2)
        ax[0].plot(ep, loss[:i + 1], color="#d1495b", lw=2); ax[0].set_xlim(1, 40)
        ax[0].set_ylim(0, max(loss) * 1.05); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
        ax[1].plot(ep, np.array(acc_tr[:i + 1]) * 100, lw=2, label="train")
        ax[1].plot(ep, np.array(acc_te[:i + 1]) * 100, lw=2, label="held-out test")
        ax[1].axhline(11.3, ls="--", color="grey", lw=1, label="majority baseline")
        ax[1].set_xlim(1, 40); ax[1].set_ylim(0, 100); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
        ax[1].set_title(f"accuracy: test {acc_te[i]:.1%} (53 cell-type families)")
        fig.suptitle(f"Project 06 - neural net learning to name a neuron from its wiring  |  epoch {i + 1}",
                     fontsize=11, weight="bold")
        fig.tight_layout()
    save(FuncAnimation(fig, draw, frames=frames_with_hold(40, 10)), "p06_classifier_training", 6)


# --- 07: t-SNE unfolding the connectivity embedding ------------------------
def p07():
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
    conn, neurons = hemibrain()
    ids = neurons.bodyId.values; idx = {b: i for i, b in enumerate(ids)}; n = len(ids)
    A = sparse.coo_matrix((conn.weight.values.astype(float),
                           (conn.bodyId_pre.map(idx).values, conn.bodyId_post.map(idx).values)), (n, n)).tocsr()
    emb = normalize(np.hstack([TruncatedSVD(64, random_state=42).fit_transform(normalize(A, "l1", axis=1)),
                               TruncatedSVD(64, random_state=42).fit_transform(normalize(A.T.tocsr(), "l1", axis=1))]))
    fam = neurons.family.to_numpy(dtype=str)
    top = neurons.family[neurons.family != "other"].value_counts().head(10).index
    pool = np.where(np.isin(fam, top))[0]
    pick = np.random.default_rng(42).choice(pool, 2500, replace=False)
    snaps = []
    for it in (250, 280, 320, 380, 460, 560, 700, 1000):
        snaps.append((it, TSNE(2, init="pca", perplexity=30, max_iter=it, random_state=42).fit_transform(emb[pick])))
        print(f"    t-SNE {it} iterations done", flush=True)
    cmap = plt.get_cmap("tab10")
    col = np.array([cmap(list(top).index(f)) for f in fam[pick]])
    fig, ax = plt.subplots(figsize=(7, 6))

    def draw(i):
        k = min(i // 3, len(snaps) - 1)
        it, xy = snaps[k]
        if k + 1 < len(snaps) and i < 3 * (len(snaps) - 1):
            f = (i % 3) / 3; xy = (1 - f) * xy + f * snaps[k + 1][1]
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=col)
        for j, f in enumerate(top):
            ax.scatter([], [], c=[cmap(j)], label=f)
        ax.legend(fontsize=7, ncol=2, loc="lower right", markerscale=2)
        ax.set_title(f"Project 07 - cell types emerging from wiring alone (no labels)\n"
                     f"t-SNE of learned 128-d neuron embeddings, iteration {it}", fontsize=10)
    save(FuncAnimation(fig, draw, frames=frames_with_hold(3 * (len(snaps) - 1) + 1, 10)),
         "p07_embedding_unfolding", 6)


# --- 08: link-prediction model training (ROC sharpening) -------------------
def p08():
    import importlib.util
    from scipy import sparse
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import normalize
    spec = importlib.util.spec_from_file_location("lp", PROJ / "08_synapse_link_prediction" / "link_prediction.py")
    lp = importlib.util.module_from_spec(spec); spec.loader.exec_module(lp)
    conn, neurons = hemibrain()
    ids = neurons.bodyId.values; idx = {b: i for i, b in enumerate(ids)}; n = len(ids)
    pre = conn.bodyId_pre.map(idx).values; post = conn.bodyId_post.map(idx).values
    w = conn.weight.values.astype(float)
    keys = np.sort(pre.astype(np.int64) * n + post)
    rng = np.random.default_rng(lp.RANDOM_STATE)
    test = rng.random(len(pre)) < lp.TEST_FRAC; tr = ~test
    A = sparse.coo_matrix((w[tr], (pre[tr], post[tr])), (n, n)).tocsr()
    eo = lp.TruncatedSVD_fit(normalize(A, "l1", axis=1)); ei = lp.TruncatedSVD_fit(normalize(A.T.tocsr(), "l1", axis=1))
    os_, is_ = np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel()
    od, idg = np.diff(A.indptr), np.diff(A.tocsc().indptr)
    F = lambda s, t: lp.edge_features(s, t, eo, ei, os_, is_, od, idg, keys, n)
    tp = rng.choice(np.where(tr)[0], lp.N_TRAIN_PAIRS, replace=False)
    te = rng.choice(np.where(test)[0], lp.N_TEST_PAIRS, replace=False)
    ni, nj = lp.sample_non_edges(n, keys, len(tp), rng); ti, tj = lp.sample_non_edges(n, keys, len(te), rng)
    Xtr = np.vstack([F(pre[tp], post[tp]), F(ni, nj)]); ytr = np.r_[np.ones(len(tp)), np.zeros(len(ni))]
    Xte = np.vstack([F(pre[te], post[te]), F(ti, tj)]); yte = np.r_[np.ones(len(te)), np.zeros(len(ti))]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=lp.RANDOM_STATE).fit(Xtr, ytr)
    stages = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 200, 300]
    probs = {s: p[:, 1] for s, p in enumerate(clf.staged_predict_proba(Xte), 1) if s in stages}
    stages = [s for s in stages if s in probs]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    def draw(i):
        s = stages[i]; fpr, tpr, _ = roc_curve(yte, probs[s])
        ax.clear(); ax.plot([0, 1], [0, 1], "--", color="grey")
        for s0 in stages[:i]:
            f0, t0, _ = roc_curve(yte, probs[s0]); ax.plot(f0, t0, color="#d1495b", alpha=0.12)
        ax.plot(fpr, tpr, color="#d1495b", lw=2.5)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(f"Project 08 - learning to predict synapses\n{s} boosted trees: "
                     f"ROC-AUC {roc_auc_score(yte, probs[s]):.3f} on held-out connections", fontsize=10)
    save(FuncAnimation(fig, draw, frames=frames_with_hold(len(stages), 8)), "p08_linkpred_training", 4)


# --- 09a: all seven FlyLab brains training over rounds ----------------------
def p09_training():
    sys.path.insert(0, str(PROJ / "09_flylab"))
    tasks = ["walk_forward", "turn", "goto", "odor_seek", "odor_avoid", "light_seek", "light_avoid"]
    pretty = {"walk_forward": "Walk forward", "turn": "Turn in place", "goto": "Go to a location",
              "odor_seek": "Find food by smell", "odor_avoid": "Escape a bad smell",
              "light_seek": "Walk toward light", "light_avoid": "Hide from light"}
    runs = {}
    for t in tasks:
        p = PROJ / "09_flylab" / "training" / t / "log.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            runs[t] = rows
    maxr = max(len(r) for r in runs.values())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    steps = 40
    cmap = plt.get_cmap("tab10")

    def draw(i):
        ax.clear()
        upto = int(np.ceil((i + 1) / steps * maxr))
        for k, (t, rows) in enumerate(runs.items()):
            rr = rows[:upto]
            if not rr:
                continue
            x = [r["round"] for r in rr]
            ys = np.maximum.accumulate([100 * r["physics_val"] for r in rr])
            ax.plot(x, [100 * r["surrogate"] for r in rr], color=cmap(k), alpha=0.25, lw=1)
            ax.step(x, ys, where="post", color=cmap(k), lw=2.2, label=pretty[t])
        ax.set_xscale("symlog", linthresh=10); ax.set_xlim(1, maxr); ax.set_ylim(-3, 103)
        ax.set_xlabel("training round (100 ES generations + physics validation each)")
        ax.set_ylabel("success (%)")
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.set_title(f"Project 09 - FlyLab: 7 fly brains training continuously, physics in the loop "
                     f"(round {upto})\nbold = best physics validation so far, faint = surrogate", fontsize=10)
        fig.tight_layout()
    save(FuncAnimation(fig, draw, frames=frames_with_hold(steps, 12)), "p09_flylab_training", 8)


# --- 09b: each brain's physics film (needs .venv-sim for imageio) ----------
def flylab_videos():
    import imageio.v2 as imageio
    from PIL import Image
    vdir = PROJ / "09_flylab" / "dashboard" / "videos"
    OUT.mkdir(parents=True, exist_ok=True)
    for mp4 in sorted(vdir.glob("*.mp4")):
        reader = imageio.get_reader(str(mp4))
        fps = reader.get_meta_data().get("fps", 25)
        step = max(1, int(round(fps / 8)))
        frames = []
        for i, fr in enumerate(reader):
            if i % step or len(frames) >= 8 * 10:
                continue
            im = Image.fromarray(fr)
            im = im.resize((480, int(im.height * 480 / im.width)), Image.LANCZOS)
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=96))
        reader.close()
        out = OUT / f"flylab_{mp4.stem}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=125, loop=0, optimize=True)
        print(f"  wrote {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)", flush=True)


JOBS = {"p01": p01, "p02": p02, "p03": p03, "p04": p04, "p05": p05, "p06": p06,
        "p07": p07, "p08": p08, "p09": p09_training, "flylab_videos": flylab_videos}

if __name__ == "__main__":
    wanted = sys.argv[1:] or [k for k in JOBS if k != "flylab_videos"]
    for k in wanted:
        t0 = time.time()
        print(f"[{k}]", flush=True)
        JOBS[k]()
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
