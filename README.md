# radioprotect-sm

Stage-1 **small-molecule** pipeline to prioritize **purchasable hits** for laboratory radioprotection testing.

This repository adapts the scientific objectives of small molecules, using **ChEMBL proxy labels** and the methodological discipline of the COMET paper (held-out evaluation, ranking metrics, ensemble, reproducibility) **without** porting COMET’s multi-component formulation architecture.

## Scientific premises (do not change silently)

1. Modeling object = **one small molecule** (SMILES), not a polymer or LNP formulation.
2. Stage-1 labels from ChEMBL are **proxies** aligned with Aims 1–2:
   - `dna_binding` ≈ Aim 1 (DNA association)
   - `ros_scavenging` ≈ Aim 2 (antioxidant / ROS assays)
   - `cytotox` = safety penalty term
   - `radioprotection_lit` = sparse literature/assay keyword hits (external validation if too small to train)
3. Proxy ≠ experimental radioprotection. Hits are **hypotheses for purchase/testing**.
4. No fabricated experimental lab values. Lab data schema is ready but empty until real assays exist.
5. Primary stage-1 backend: **Morgan fingerprint + RandomForest ensemble (5 seeds)**. Chemprop is an optional extra; if missing, the pipeline states the fallback explicitly.

See [docs/methodology.md](docs/methodology.md) for full methodological notes.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Offline reproducible demo (bundled fixtures — not experimental claims)
radioprotect-sm fetch-chembl --demo
radioprotect-sm train
radioprotect-sm screen

# Or one shot:
radioprotect-sm run-all --demo
```

Outputs:

- `outputs/reports/train_report.md` — held-out Spearman / top-k metrics
- `outputs/hits_for_purchase.csv` — diverse ranked purchasable candidates
- `runs/ensemble_bundle.json` — ensemble members for screening

### Live ChEMBL pull

```bash
radioprotect-sm fetch-chembl --no-cache
radioprotect-sm train
radioprotect-sm screen
```

Network access to `www.ebi.ac.uk` is required. Filters live in `configs/data_chembl.yaml` (versioned).

### Optional Chemprop

```bash
pip install -e ".[chemprop]"
radioprotect-sm train --model chemprop
```

If Chemprop/torch is not installed, training logs an **explicit** fallback to the Morgan+RF ensemble.

## Repository layout

```
configs/               # auditable YAML (data filters, train, screen, lab schema)
src/radioprotect_sm/   # modular Python package
tests/                 # unit + demo e2e tests
data/                  # raw/processed (gitignored) + manifests + demo library
docs/methodology.md
```

## Tests

```bash
pytest
```

## Lab data later

Enable `configs/lab_data.yaml`, point `path` at a CSV matching `data/lab/lab_data_template.csv`, and keep `data_source=lab` distinct from ChEMBL rows. Do not overwrite ChEMBL labels.

## Citation / references

- Chan et al., *Nat Nanotechnol* (2025) — COMET (methodology inspiration; LNP multi-component model).
- ChEMBL — public bioactivity source for stage-1 proxies.
