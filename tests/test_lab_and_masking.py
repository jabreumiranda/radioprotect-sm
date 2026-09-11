"""Lab schema and multitask masking tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radioprotect_sm.data.chembl import long_to_wide
from radioprotect_sm.lab import REQUIRED_COLUMNS, empty_lab_template, validate_lab_dataframe
from radioprotect_sm.models.baseline import BaselineConfig, MorganRFMultitask


def test_empty_lab_template_has_required_columns():
    df = empty_lab_template()
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 0


def test_lab_validation_rejects_non_numeric_values():
    df = empty_lab_template()
    row = {c: "x" for c in REQUIRED_COLUMNS}
    row["value"] = "TBD"
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    result = validate_lab_dataframe(df)
    assert not result.ok
    assert any("non-numeric" in e for e in result.errors)


def test_lab_validation_accepts_minimal_valid_row():
    row = {c: 1 for c in REQUIRED_COLUMNS}
    row.update(
        {
            "compound_id": "LAB001",
            "smiles": "c1ccccc1O",
            "assay_name": "comet",
            "endpoint": "tail_moment_reduction_pct",
            "value": 42.0,
            "units": "percent",
            "cell_line": "HOEC",
            "radiation_dose_gy": 6,
            "pretreatment_hours": 4,
            "replicate": 1,
            "measured_at": "2026-01-01",
            "operator": "jdoe",
            "notes": "",
        }
    )
    df = pd.DataFrame([row])
    assert validate_lab_dataframe(df).ok


def test_long_to_wide_leaves_missing_as_nan():
    long_df = pd.DataFrame(
        [
            {
                "inchikey": "AAA",
                "smiles": "CCO",
                "scaffold_smiles": "C",
                "chembl_id": "CHEM1",
                "task": "dna_binding",
                "label": 6.0,
            },
            {
                "inchikey": "BBB",
                "smiles": "CCOC",
                "scaffold_smiles": "C",
                "chembl_id": "CHEM2",
                "task": "ros_scavenging",
                "label": 5.0,
            },
        ]
    )
    wide = long_to_wide(long_df, ["dna_binding", "ros_scavenging", "cytotox"])
    assert np.isnan(wide.loc[wide.inchikey == "AAA", "ros_scavenging"].iloc[0])
    assert np.isnan(wide.loc[wide.inchikey == "BBB", "dna_binding"].iloc[0])
    assert np.isnan(wide["cytotox"]).all()


def test_baseline_masks_nan_labels_per_task():
    smiles = pd.Series(["c1ccccc1O", "CCOC(=O)c1ccccc1", "NCCCCC(N)C(=O)O", "Cc1ccc(O)cc1", "Oc1ccccc1O", "c1ccccc1N"])
    y = pd.DataFrame(
        {
            "dna_binding": [6.0, 5.5, 7.0, 5.0, np.nan, 6.2],
            "ros_scavenging": [5.0, np.nan, 4.5, 6.0, 6.5, 5.2],
        }
    )
    model = MorganRFMultitask(
        tasks=["dna_binding", "ros_scavenging"],
        config=BaselineConfig(n_estimators=20, seed=0),
    )
    model.fit(smiles, y)
    pred = model.predict(smiles)
    assert pred.shape == (6, 2)
    assert np.isfinite(pred.to_numpy()).all()
