"""Molecular canonicalization and fingerprint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SaltRemover import SaltRemover

_SALT_REMOVER = SaltRemover()


@dataclass(frozen=True)
class CanonicalMolecule:
    smiles: str
    inchikey: str
    heavy_atoms: int
    molwt: float
    logp: float
    scaffold_smiles: str


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def canonicalize_smiles(
    smiles: str,
    *,
    remove_salts: bool = True,
    keep_largest_fragment: bool = True,
    reject_mixtures: bool = True,
    max_heavy_atoms: int = 80,
    min_heavy_atoms: int = 5,
) -> CanonicalMolecule | None:
    """Sanitize and canonicalize a SMILES string.

    Rules (documented; do not change silently):
    - Reject unparseable / unsanitizable structures.
    - Optionally strip salts and keep the largest organic fragment.
    - Reject '.' mixtures when reject_mixtures=True after salt removal.
    - Enforce heavy-atom bounds.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    if remove_salts:
        mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)

    if keep_largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    can = Chem.MolToSmiles(mol, canonical=True)
    if reject_mixtures and "." in can:
        return None

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < min_heavy_atoms or n_heavy > max_heavy_atoms:
        return None

    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""

    return CanonicalMolecule(
        smiles=can,
        inchikey=inchikey,
        heavy_atoms=n_heavy,
        molwt=float(Descriptors.MolWt(mol)),
        logp=float(Descriptors.MolLogP(mol)),
        scaffold_smiles=scaffold_smiles,
    )


def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    except Exception:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    on_bits = fp.GetOnBits()
    arr[list(on_bits)] = 1.0
    return arr


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    denom = float(a.sum() + b.sum() - inter)
    if denom <= 0:
        return 0.0
    return inter / denom


def passes_druglike_filters(
    smiles: str,
    *,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    min_molwt: float,
    max_molwt: float,
    max_logp: float,
) -> tuple[bool, dict[str, Any]]:
    can = canonicalize_smiles(
        smiles,
        remove_salts=True,
        keep_largest_fragment=True,
        reject_mixtures=True,
        max_heavy_atoms=max_heavy_atoms,
        min_heavy_atoms=min_heavy_atoms,
    )
    if can is None:
        return False, {}
    ok = (
        min_molwt <= can.molwt <= max_molwt
        and can.logp <= max_logp
        and min_heavy_atoms <= can.heavy_atoms <= max_heavy_atoms
    )
    meta = {
        "smiles": can.smiles,
        "inchikey": can.inchikey,
        "molwt": can.molwt,
        "logp": can.logp,
        "heavy_atoms": can.heavy_atoms,
        "scaffold_smiles": can.scaffold_smiles,
    }
    return ok, meta
