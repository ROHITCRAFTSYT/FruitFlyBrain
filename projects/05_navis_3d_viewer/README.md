# Project 5 — 3D neuron morphology viewer (NAVis)

Renders the actual 3D shape of fly-brain neurons using
[NAVis](https://navis-org.github.io/navis/) — the standard Python library for
neuron analysis and visualization.

**Run**
```bash
../../.venv/Scripts/python view_neurons.py            # 5 bundled example neurons (no login)
../../.venv/Scripts/python view_neurons.py --n 3
```

**Outputs** (`outputs/`)
- `neurons_3d.png` — static render from 3 viewing angles (frontal / dorsal / lateral)
- `neurons_3d.html` — **interactive 3D**: open in any browser, drag to rotate, scroll to zoom

## Two data sources

### `example` (default — zero config)
Uses real *Drosophila* hemibrain skeletons **bundled inside NAVis**. No download,
no account. Always works — good for confirming the viewer runs.

### `--neuprint` (real neurons by bodyId)
Fetches specific neurons straight from the Janelia hemibrain server. With no
`--bodyids`, it reads the **top hub neurons from Project 1** directly out of
`01_connectome_graph_analysis/outputs/top_hubs.csv` (e.g. APL, DPM, PVLP011), so
the viewer stays in sync with whatever the connectome analysis last ranked.
Change how many with `--n-hubs N`, or pass explicit ids with `--bodyids`.
(If Project 1 hasn't been run yet, it falls back to a built-in trio.)

Needs a **free** token:
1. Sign in at https://neuprint.janelia.org
2. Account menu (top-right) → copy **Auth Token**
3. Set it and open a new terminal:
   ```bash
   setx NEUPRINT_TOKEN "your.token.here"      # Windows
   export NEUPRINT_TOKEN="your.token.here"     # macOS/Linux
   ```
4. Run:
   ```bash
   ../../.venv/Scripts/python view_neurons.py --neuprint                 # top 3 hubs from Project 1
   ../../.venv/Scripts/python view_neurons.py --neuprint --n-hubs 5      # top 5 hubs
   ../../.venv/Scripts/python view_neurons.py --neuprint --bodyids 612371421 300972942
   ```

> Why a token here but nowhere else? Individual 3D skeletons aren't part of the
> no-login hemibrain export (the full skeleton set is a 1.9 GB tarball). NAVis's
> bundled examples cover the default case; neuPrint gives authentic, arbitrary
> neurons for the price of a free token.

Data: Janelia hemibrain v1.2 (see `../../datasets/README.md`).
