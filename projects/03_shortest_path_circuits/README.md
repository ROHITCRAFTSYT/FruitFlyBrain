# Project 3 — Synaptic pathway tracer

Finds the strongest multi-synapse routes carrying a signal from one cell type to
another. Edge distance = 1 / synapse-weight, so the "shortest" path is the one
built from the **heaviest** connections.

**Run**
```bash
../../.venv/Scripts/python trace.py                        # KC -> MBON (default)
../../.venv/Scripts/python trace.py --source ER --target EPG
../../.venv/Scripts/python trace.py --source LC --target MBON --k 8
```
Useful type prefixes: `KC` (Kenyon cells), `MBON` (mushroom-body output),
`ER`/`EPG`/`PEN`/`PEG` (central complex / compass), `LC` (visual projection),
`DA`/`PPL` (dopaminergic), `oviIN`.

**What it does**
- builds a backbone graph (connections ≥5 synapses)
- adds a virtual super-source/super-sink so one Dijkstra finds the best
  *type → type* route
- reports the primary path (with per-hop synapse counts) plus alternative routes

**Outputs** (`outputs/`)
- `pathway_<src>_to_<dst>.png` — circuit diagram
- `paths_<src>_to_<dst>.csv`

**Sanity check:** the default KC→MBON trace returns a **direct KCg→MBON05
synapse** (the fundamental learning connection) and an alternative route via
**APL/DPM** (the modulatory neurons) — the expected mushroom-body wiring.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
