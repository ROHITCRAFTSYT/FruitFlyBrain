# Project 6 — Connectivity-based cell-type classifier 🧠🤖

**Supervised machine learning.** Predicts a neuron's cell-type family purely from
its *connectivity fingerprint* — the fraction of its input/output synapses shared
with each major cell-type family. This is the canonical connectomics ML task
(connectivity-based cell typing).

**Run**
```bash
../../.venv/Scripts/python train_classifier.py
```

**Pipeline**
- features: per-neuron output- and input-synapse fractions across 53 cell
  families + degree/strength summaries (112 features)
- targets: 53 families with ≥40 neurons (18,590 neurons)
- models: RandomForest **and** a neural network (MLP), stratified 75/25 split

**Results** (held-out test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| RandomForest | **88.2%** | 0.847 |
| Neural net (MLP) | 87.6% | 0.848 |
| Majority baseline | 11.3% | 0.00 |

**Outputs** (`outputs/`): `confusion_matrix.png`, `feature_importance.png`,
`classification_report.txt`, `metrics.json`.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
