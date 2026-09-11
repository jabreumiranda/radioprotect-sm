"""Scaffold-aware splits and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from radioprotect_sm.data.canonicalize import murcko_scaffold_smiles
from radioprotect_sm.utils import set_global_seed


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    scaffolds: pd.Series


def scaffold_split(
    smiles: list[str] | pd.Series,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> SplitResult:
    """Bemis–Murcko scaffold split.

    All molecules that share a scaffold are assigned to the same fold.
    Assignment greedily fills train → test → val by remaining capacity.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    set_global_seed(seed)
    smiles_list = list(smiles)
    n = len(smiles_list)
    empty = np.array([], dtype=int)
    if n == 0:
        return SplitResult(empty, empty, empty, pd.Series(dtype=str))

    scaffolds = [
        murcko_scaffold_smiles(smi) or f"UNIQUE_{i}" for i, smi in enumerate(smiles_list)
    ]
    scaffolds_s = pd.Series(scaffolds, name="scaffold")

    groups: dict[str, list[int]] = {}
    for idx, scaf in enumerate(scaffolds):
        groups.setdefault(scaf, []).append(idx)

    rng = np.random.default_rng(seed)
    # Deterministic shuffle of scaffolds before size-stable assignment
    scaf_items = list(groups.items())
    rng.shuffle(scaf_items)
    # Process larger scaffolds first for stabler fraction control
    scaf_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    for _, members in scaf_items:
        rem = {
            "train": n_train - len(train_idx),
            "test": n_test - len(test_idx),
            "val": n_val - len(val_idx),
        }
        # Prefer the fold with the largest remaining absolute capacity
        choice = max(rem, key=rem.get)
        if choice == "train":
            train_idx.extend(members)
        elif choice == "test":
            test_idx.extend(members)
        else:
            val_idx.extend(members)

    return SplitResult(
        train_idx=np.array(sorted(train_idx), dtype=int),
        val_idx=np.array(sorted(val_idx), dtype=int),
        test_idx=np.array(sorted(test_idx), dtype=int),
        scaffolds=scaffolds_s,
    )


def assert_no_inchikey_leakage(
    inchikeys: pd.Series | list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    keys = pd.Series(list(inchikeys))
    train_k = set(keys.iloc[list(train_idx)])
    val_k = set(keys.iloc[list(val_idx)])
    test_k = set(keys.iloc[list(test_idx)])
    overlaps = [
        ("train∩val", train_k & val_k),
        ("train∩test", train_k & test_k),
        ("val∩test", val_k & test_k),
    ]
    for name, overlap in overlaps:
        if overlap:
            raise AssertionError(f"InChIKey leakage {name}: {sorted(overlap)[:5]}")


def assert_no_scaffold_leakage(
    scaffolds: pd.Series | list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    scaf = pd.Series(list(scaffolds))
    train_s = set(scaf.iloc[list(train_idx)])
    val_s = set(scaf.iloc[list(val_idx)])
    test_s = set(scaf.iloc[list(test_idx)])
    overlaps = [
        ("train∩val", train_s & val_s),
        ("train∩test", train_s & test_s),
        ("val∩test", val_s & test_s),
    ]
    for name, overlap in overlaps:
        if overlap:
            raise AssertionError(f"Scaffold leakage {name}: {sorted(overlap)[:5]}")


def attach_split_column(df: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["split"] = ""
    out.loc[list(split.train_idx), "split"] = "train"
    out.loc[list(split.val_idx), "split"] = "val"
    out.loc[list(split.test_idx), "split"] = "test"
    out["scaffold_smiles"] = split.scaffolds.values
    assert_no_inchikey_leakage(out["inchikey"], split.train_idx, split.val_idx, split.test_idx)
    assert_no_scaffold_leakage(out["scaffold_smiles"], split.train_idx, split.val_idx, split.test_idx)
    if not out["split"].isin(["train", "val", "test"]).all():
        raise AssertionError("Some molecules were not assigned to a split.")
    return out
