# Methodology

## Goal of stage 1

Train multitask models on **existing public data** to rank **purchasable small molecules** for laboratory follow-up (buy → assay). This mirrors the proposal’s “model → exploratory hits → experimental validation” loop, with polymers replaced by small molecules.

## Why not COMET architecture here?

COMET (*Nat Nanotechnol* 2025) is a **Composite Material Transformer** for multi-ingredient formulations (lipids + molar % + process parameters). Stage-1 hit purchasing asks a different question: **which single molecule should we buy?**

**Decision:** use a **molecule-native** model with COMET-like **evaluation discipline**:

| COMET practice | Stage-1 implementation |
|---|---|
| Hold-out evaluation | Scaffold split 70/10/20 |
| Ranking focus | Spearman + top-10%/30% recovery metrics |
| Ensemble of 5 | Five RandomForest seeds (optional Chemprop later) |
| Multitask learning | Shared molecule table; per-task masked labels |
| Reproducible seeds | Fixed seeds in `configs/train.yaml` |

This is an explicit methodological choice, not a silent rewrite of COMET’s scientific claims.

## Label definitions (proxies)

Configured in `configs/data_chembl.yaml`:

1. **`dna_binding` (Aim 1 proxy)**  
   ChEMBL binding assays on `target_type=NUCLEIC-ACID` (and related keywords). Label = `pchembl_value` when present.

2. **`ros_scavenging` (Aim 2 proxy)**  
   Assays whose descriptions match a **closed keyword list** (DPPH, ABTS, superoxide, hydroxyl, ORAC, antioxidant, …). Prefer `pchembl_value`; otherwise convert IC50-like `standard_value` with the documented molar rule in `ic50_like_to_pchembl`. Unsupported units → drop (never invent).

3. **`cytotox` (safety)**  
   Functional cytotoxicity-like assays. Used with a **negative weight** in multi-objective ranking.

4. **`radioprotection_lit` (sparse auxiliary)**  
   Keyword search for radioprotect / ionizing radiation assays. If \(N <\) `train_if_n_ge` (default 80), **exclude from training** and keep as external validation / ranking prior only.

**Important:** these proxies are **not** comet-assay radioprotection, γ-H2AX reduction, or in vivo mucositis endpoints from the proposal. Hits remain hypotheses.

## Data handling rules

- RDKit sanitize → optional salt strip → largest fragment → canonical SMILES + InChIKey.
- Reject mixtures (`.` remaining after cleanup).
- Dedupe by `(inchikey, task)` with **median** aggregation (config-locked).
- Missing multitask labels stay **NaN** (masked); no imputation of experimental values.
- Manifest JSON records counts, SHA256, retrieval time, and ChEMBL release tag.

## Splits and leakage

- Primary split: **Bemis–Murcko scaffolds**.
- Unit tests assert zero InChIKey and scaffold overlap across train/val/test.
- Random split is **not** the default; introducing it requires an explicit config/docs change (ablation).

## Model

**Default:** Morgan FP (radius 2, 2048 bits) + per-task RandomForest, ensemble over seeds `[42..46]`.

**Optional:** Chemprop extras. If unavailable, the CLI logs an explicit fallback. We do **not** pretend Chemprop weights were used.

## Screening / hit selection

Three Aim-aligned shortlists are written (each with greedy Tanimoto diversity):

1. **Aim 1 only** (`hits_dna_binding.csv`) — rank by \(z(\hat y_{\text{dna}})\) (+ cytotox penalty).
2. **Aim 2 only** (`hits_ros_scavenging.csv`) — rank by \(z(\hat y_{\text{ros}})\) (+ cytotox penalty).
3. **Joint** (`hits_dna_and_ros.csv`, also aliased as `hits_for_purchase.csv`):

\[
\text{score} = w_{\text{dna}} z(\hat y_{\text{dna}}) + w_{\text{ros}} z(\hat y_{\text{ros}}) + w_{\text{cyt}} z(\hat y_{\text{cyt}}) - \lambda \cdot \overline{\sigma}
\]

with \(w_{\text{cyt}} < 0\), library-wise z-scoring, and ensemble std uncertainty penalty.

Single-aim lists surface specialists that a joint score can underrank; the joint list prefers dual-proxy balance.

Demo library path is a stand-in for ZINC/Enamine in-stock catalogs — replace `configs/screen.yaml` `library.path` for production screens.

## Lab data extension

`configs/lab_data.yaml` + `data/lab/lab_data_template.csv` define required columns for real radioprotection assays. Loading refuses non-numeric `value` placeholders. Lab rows must remain `data_source=lab` and must not overwrite ChEMBL provenance.

## What we will not do

- Fabricate experimental radioprotection measurements.
- Scrape Radioprotectors.org in bulk without authorization (contact required by their terms).
- Quietly treat ChEMBL antioxidant IC50 as in vivo radioprotection.
- Change salt/mixture/split/label rules without documenting the change first.
