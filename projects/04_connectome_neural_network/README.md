# Project 4 — Connectome-constrained neural-network simulation

Builds a recurrent network whose weight matrix **is the real fly connectome**,
then watches a stimulus cascade through it.

**Model** (simple, *not* biophysical):
```
x(t+1) = ReLU( gain · W_norm · x(t) + stimulus(t) )
```
`W_norm[i,j]` = normalised synapse count from neuron *j* onto neuron *i*, scaled
so the dynamics stay bounded. A seed cell type is stimulated for the first few
steps and we record how activity spreads.

**Run**
```bash
../../.venv/Scripts/python simulate.py                     # central complex, stimulate ER
../../.venv/Scripts/python simulate.py --seed-type EPG --steps 60
../../.venv/Scripts/python simulate.py --prefixes KC MBON APL DPM --seed-type KC
```

**What it does**
- selects a subnetwork by cell-type prefix (default = the central complex, the
  fly's navigation/"compass" hub); caps at 3,000 most-connected neurons
- assembles the weighted adjacency matrix straight from the hemibrain edges
- runs threshold-linear recurrent dynamics and records per-neuron activity

**Outputs** (`outputs/`)
- `activity_heatmap.png` — the cascade over time (neurons sorted by activation time)
- `type_response_curves.png` — which cell types light up, and when
- `type_activity_over_time.csv`, `summary.txt`

**Sanity check:** stimulating **ER** (ring neurons — visual input to the compass)
most strongly drives **EPG/EL** (the compass neurons), reproducing the known
ER→EPG pathway.

> This is a didactic dynamical model on real wiring, not a validated
> biophysical simulation — it ignores neurotransmitter sign, cell biophysics and
> timing. It shows *structure-driven signal flow*, which is genuinely
> informative and a common first-pass connectome analysis.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
