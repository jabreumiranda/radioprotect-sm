"""Unit tests for canonicalization and label conversion rules."""

from __future__ import annotations

import numpy as np
import pytest

from radioprotect_sm.data.canonicalize import canonicalize_smiles, morgan_fingerprint, tanimoto
from radioprotect_sm.data.chembl import ic50_like_to_pchembl


def test_canonicalize_strips_salt_and_keeps_largest_fragment():
    can = canonicalize_smiles("CCO.Cl", remove_salts=True, keep_largest_fragment=True, min_heavy_atoms=2)
    # ethanol + HCl salt → ethanol fragment (CCO has 3 heavy atoms if counting O? CCO = 3? C,C,O = 3)
    assert can is not None
    assert "." not in can.smiles
    assert can.inchikey


def test_canonicalize_rejects_unsanitizable():
    assert canonicalize_smiles("not_a_molecule") is None


def test_canonicalize_rejects_too_small():
    assert canonicalize_smiles("C", min_heavy_atoms=5) is None


def test_ic50_um_to_pchembl():
    # 1 uM = 1e-6 M → p = 6
    assert ic50_like_to_pchembl(1.0, "uM") == pytest.approx(6.0)
    assert ic50_like_to_pchembl(1.0, "nM") == pytest.approx(9.0)


def test_ic50_unknown_units_returns_none():
    assert ic50_like_to_pchembl(1.0, "mg/kg") is None


def test_ic50_nonpositive_returns_none():
    assert ic50_like_to_pchembl(0.0, "uM") is None
    assert ic50_like_to_pchembl(-1.0, "uM") is None


def test_morgan_and_tanimoto_self_similarity():
    fp = morgan_fingerprint("CCOC(=O)c1ccccc1")
    assert fp is not None
    assert tanimoto(fp, fp) == pytest.approx(1.0)
    assert np.isfinite(fp).all()
