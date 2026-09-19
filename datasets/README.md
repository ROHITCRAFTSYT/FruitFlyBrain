# Datasets — provenance, licenses & citations

Two complementary, **officially released** *Drosophila melanogaster* (fruit fly)
connectome datasets. Both were downloaded directly from their official hosts
with no login required.

---

## 1. `hemibrain/` — Janelia FlyEM *hemibrain* v1.2

A dense reconstruction of roughly one third of the central fly brain, with
**every traced neuron and every synaptic connection between them**. This is the
connectivity backbone used by projects 1, 3 and 4.

| | |
|---|---|
| **Source (official)** | Janelia Research Campus, HHMI — FlyEM project |
| **Download URL** | `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz` |
| **Portal** | https://neuprint.janelia.org · https://www.janelia.org/project-team/flyem/hemibrain |
| **License** | CC BY 4.0 |
| **SHA-256** | `07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7` |

Files (after extraction into `exported-traced-adjacencies-v1.2/`):
- `traced-neurons.csv` — `bodyId, type, instance` (21,739 neurons)
- `traced-total-connections.csv` — `bodyId_pre, bodyId_post, weight` (3.55M edges; weight = synapse count)
- `traced-roi-connections.csv` — same, split per brain region (ROI)

**Citation:** Scheffer, L.K., Xu, C.S., Januszewski, M. et al. *A connectome and
analysis of the adult Drosophila central brain.* eLife 9:e57443 (2020).
https://doi.org/10.7554/eLife.57443

---

## 2. `flywire_fafb/` — Princeton **FlyWire** whole-brain annotations (FAFB)

Annotations for the **complete adult female fly brain** — ~140,000 proofread
neurons with cell types, predicted neurotransmitters, hemisphere, lineage and
more. Used by project 2.

| | |
|---|---|
| **Source (official)** | FlyWire consortium (Princeton U. / MRC LMB) — `flyconnectome/flywire_annotations` |
| **Download URL** | `https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/` |
| **Portal** | https://codex.flywire.ai · https://flywire.ai |
| **License** | CC BY 4.0 |
| **SHA-256 (file1)** | `9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be` |

Files:
- `Supplemental_file1_neuron_annotations.tsv` — 139,248 neurons × 31 attributes
- `Supplemental_file2_non_neuron_annotations.tsv` — glia / non-neuronal segments
- `Supplemental_file3_summary_with_ngl_links.csv` — per-neuron summary + Neuroglancer links
- `README.md` — column dictionary from the consortium

**Citations:**
- Dorkenwald, S., Matsliah, A., Sterling, A.R. et al. *Neuronal wiring diagram
  of an adult brain.* Nature 634, 124–138 (2024). https://doi.org/10.1038/s41586-024-07558-y
- Schlegel, P., Yin, Y., Bates, A.S. et al. *Whole-brain annotation and
  multi-connectome cell typing of Drosophila.* Nature 634, 139–152 (2024).
  https://doi.org/10.1038/s41586-024-07686-5

---

## Re-downloading

From the repo root:

```bash
# hemibrain (~46 MB)
curl -L -o datasets/hemibrain/exported-traced-adjacencies-v1.2.tar.gz \
  https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz
tar -xzf datasets/hemibrain/exported-traced-adjacencies-v1.2.tar.gz -C datasets/hemibrain

# FlyWire annotations (~32 MB)
curl -L -o datasets/flywire_fafb/Supplemental_file1_neuron_annotations.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
```

## Want the full FlyWire *connectivity* edges too?

FlyWire's synapse-level edge tables are gated behind a free account on
[Codex](https://codex.flywire.ai) (**Info → Download Data**), or can be queried
programmatically with a free token via
[`CAVEclient`](https://github.com/seung-lab/CAVEclient) /
[`fafbseg`](https://github.com/navis-org/fafbseg-py). This workspace deliberately
uses only no-login official downloads; the hemibrain export already provides a
complete synapse-weighted graph for the network projects.
