"""Split leakage and scaffold assignment tests."""

from __future__ import annotations

import pandas as pd
import pytest

from radioprotect_sm.splits import (
    assert_no_inchikey_leakage,
    assert_no_scaffold_leakage,
    attach_split_column,
    scaffold_split,
)


def _demo_wide() -> pd.DataFrame:
    return pd.read_csv("tests/fixtures/demo_labels_wide.csv")


def test_scaffold_split_no_inchikey_or_scaffold_leakage():
    df = _demo_wide()
    split = scaffold_split(df["smiles"], train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=42)
    assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == len(df)
    assert_no_inchikey_leakage(df["inchikey"], split.train_idx, split.val_idx, split.test_idx)
    assert_no_scaffold_leakage(split.scaffolds, split.train_idx, split.val_idx, split.test_idx)


def test_scaffold_split_reproducible():
    df = _demo_wide()
    a = scaffold_split(df["smiles"], seed=7)
    b = scaffold_split(df["smiles"], seed=7)
    assert (a.train_idx == b.train_idx).all()
    assert (a.val_idx == b.val_idx).all()
    assert (a.test_idx == b.test_idx).all()


def test_attach_split_column_assigns_all_rows():
    df = _demo_wide()
    split = scaffold_split(df["smiles"], seed=42)
    out = attach_split_column(df, split)
    assert set(out["split"]) <= {"train", "val", "test"}
    assert out["split"].ne("").all()


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        scaffold_split(["CCO"], train_frac=0.5, val_frac=0.5, test_frac=0.5)
