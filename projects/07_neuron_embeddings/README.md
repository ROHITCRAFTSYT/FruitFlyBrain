# Project 7 — Neuron embeddings & unsupervised cell typing 🧬

**Unsupervised / representation learning.** Learns a 128-dimensional embedding
of every neuron *from the wiring alone*, clusters them, and asks: do cell types
emerge without any labels? Expert annotations are used **only as a scorecard**.

**Run**
```bash
../../.venv/Scripts/python embed_and_cluster.py
```

**Method**
- sparse synapse adjacency → truncated-SVD of row-normalized **output** and
  **input** profiles → 128-d connectivity embedding
- KMeans into 53 groups (= number of annotated families)
- score vs. annotations: Adjusted Rand Index, Normalized Mutual Info, purity
- 2-D t-SNE colored by true family

**Results** (18,590 labelled neurons)

| Metric | Value |
|---|---|
| Normalized Mutual Info (NMI) | **0.59** |
| Adjusted Rand Index (ARI) | 0.27 |
| Mean cluster purity | 0.63 |

Fully unsupervised recovery of cell identity is hard, so an NMI of ~0.6 and
clusters that are 63% pure on average is a genuinely strong result — the wiring
alone carries most of the cell-type signal.

**Outputs** (`outputs/`): `tsne_embedding.png`, `metrics.json`.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
