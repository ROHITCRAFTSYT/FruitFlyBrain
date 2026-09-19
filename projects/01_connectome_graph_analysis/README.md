# Project 1 — Connectome graph analysis

Treats the hemibrain as a directed, weighted network and measures its
large-scale structure.

**Run**
```bash
../../.venv/Scripts/python analyze.py     # Windows: ..\..\.venv\Scripts\python.exe analyze.py
```

**What it computes**
- neurons, connections, total synapses, density, edge reciprocity
- in/out **degree distributions** (log-log) — the connectome is heavy-tailed
- top **hub neurons** by synaptic strength (annotated with cell type)
- weakly/strongly-connected components of the strong-connection backbone

**Outputs** (`outputs/`)
- `degree_distribution.png`, `top_hubs.png`
- `top_hubs.csv`, `neuron_degree_table.csv`, `summary.txt`

**Sanity check:** the top hubs come out as **APL** and **DPM** — the two giant
mushroom-body neurons known from the literature to contact enormous numbers of
partners. If you see those names, the pipeline is wired correctly.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
