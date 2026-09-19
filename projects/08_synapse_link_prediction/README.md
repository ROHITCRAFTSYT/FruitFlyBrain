# Project 8 — Synapse link prediction 🔗

**Graph machine learning.** Given two neurons, predict whether one synapses onto
the other — the classic *link-prediction* task, on a real brain, with a
**leakage-free** protocol.

**Run**
```bash
../../.venv/Scripts/python link_prediction.py
```

**Protocol (no leakage)**
1. hold out 15% of real connections as positive test edges
2. learn directed node embeddings (SVD of output/input profiles) from the
   **training graph only**
3. pair features: compatibility of source's output embedding with target's input
   embedding (Hadamard) + degree/strength/reciprocity (69 features)
4. train HistGradientBoosting on 240k train pairs; evaluate on 80k held-out pairs

**Results** (held-out edges vs. sampled non-edges)

| Metric | Value |
|---|---|
| ROC-AUC | **0.971** |
| PR-AUC (average precision) | 0.968 |

The model recovers unseen synaptic connections with 97% ROC-AUC — the learned
embeddings capture real wiring rules, not memorized edges (they never saw the
test edges).

**Outputs** (`outputs/`): `roc_pr_curves.png`, `metrics.json`.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
