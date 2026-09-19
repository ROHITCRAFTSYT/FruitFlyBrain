# Project 2 — Whole-brain neurotransmitter & cell-type atlas

Turns the FlyWire annotation of a *complete* adult brain (~140k neurons) into a
census.

**Run**
```bash
../../.venv/Scripts/python atlas.py       # Windows: ..\..\.venv\Scripts\python.exe atlas.py
```

**What it computes**
- neurons per **super-class** (optic lobe, central brain, sensory, ascending, ...)
- predicted **neurotransmitter** mix across the whole brain
- left / right / center **symmetry**
- the most numerous **cell types**
- a neurotransmitter × super-class **cross-tabulation** heatmap

**Outputs** (`outputs/`)
- `super_class_census.png`, `neurotransmitter_mix.png`, `top_cell_types.png`, `nt_by_super_class.png`
- matching `.csv` tables + `summary.txt`

**Sanity check:** the brain comes out **~62% cholinergic**, ~18% glutamatergic,
~14% GABAergic — matching the published FlyWire figures.

Data: FlyWire FAFB annotations (see `../../datasets/README.md`).
