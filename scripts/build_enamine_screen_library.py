#!/usr/bin/env python3
"""Build a stratified ~50k Enamine REAL screening CSV from a CXSMILES dump.

Streams the large file (does not load 13.6M rows into memory). Prefers
precomputed MW/HAC/sLogP columns when present; cleans CXSMILES extensions.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def _truthy(val: str) -> bool:
    return val.strip().lower() in {"true", "1", "yes", "y", "t"}


def _plain_smiles(smi: str) -> str:
    """Drop CXSMILES extensions (space + '|' block) when present."""
    s = smi.strip()
    if " |" in s:
        s = s.split(" |", 1)[0].strip()
    return s


def _passes_prefilter(
    *,
    smiles: str,
    mw: float | None,
    hac: float | None,
    slogp: float | None,
    min_hac: int,
    max_hac: int,
    min_mw: float,
    max_mw: float,
    max_logp: float,
) -> bool:
    if not smiles or "." in smiles:
        return False
    if hac is not None and not (min_hac <= hac <= max_hac):
        return False
    if mw is not None and not (min_mw <= mw <= max_mw):
        return False
    if slogp is not None and slogp > max_logp:
        return False
    return True


def _parse_float(val: str) -> float | None:
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def reservoir_add(bucket: list[dict], item: dict, seen: int, capacity: int, rng: random.Random) -> int:
    """Classic reservoir sampling; returns updated seen count for this stratum."""
    seen += 1
    if len(bucket) < capacity:
        bucket.append(item)
    else:
        j = rng.randrange(seen)
        if j < capacity:
            bucket[j] = item
    return seen


def build_library(
    source: Path,
    output: Path,
    *,
    n_total: int = 50_000,
    n_natural: int = 15_000,
    seed: int = 42,
    vendor: str = "Enamine_REAL",
    min_hac: int = 8,
    max_hac: int = 50,
    min_mw: float = 150.0,
    max_mw: float = 600.0,
    max_logp: float = 5.5,
) -> dict:
    if n_natural > n_total:
        raise ValueError("n_natural cannot exceed n_total")
    n_other = n_total - n_natural
    rng = random.Random(seed)

    natural: list[dict] = []
    other: list[dict] = []
    seen_nat = 0
    seen_other = 0
    n_read = 0
    n_kept = 0

    with source.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"No header in {source}")
        fields = {f.strip(): f for f in reader.fieldnames}
        required = ["smiles", "id"]
        for r in required:
            if r not in fields:
                raise RuntimeError(f"Missing column {r!r} in {source}; got {reader.fieldnames}")

        smi_key = fields["smiles"]
        id_key = fields["id"]
        nat_key = fields.get("natural_product-like")
        mw_key = fields.get("MW")
        hac_key = fields.get("HAC")
        logp_key = fields.get("sLogP")

        for row in reader:
            n_read += 1
            if n_read % 1_000_000 == 0:
                print(
                    f"... scanned {n_read:,} | kept {n_kept:,} | "
                    f"nat_pool={len(natural):,}/{seen_nat:,} other_pool={len(other):,}/{seen_other:,}",
                    flush=True,
                )

            smiles = _plain_smiles(row.get(smi_key, ""))
            catalog_id = (row.get(id_key) or "").strip()
            if not catalog_id:
                continue
            mw = _parse_float(row.get(mw_key, "")) if mw_key else None
            hac = _parse_float(row.get(hac_key, "")) if hac_key else None
            slogp = _parse_float(row.get(logp_key, "")) if logp_key else None
            if not _passes_prefilter(
                smiles=smiles,
                mw=mw,
                hac=hac,
                slogp=slogp,
                min_hac=min_hac,
                max_hac=max_hac,
                min_mw=min_mw,
                max_mw=max_mw,
                max_logp=max_logp,
            ):
                continue

            n_kept += 1
            is_nat = _truthy(row.get(nat_key, "")) if nat_key else False
            item = {
                "catalog_id": catalog_id,
                "smiles": smiles,
                "vendor": vendor,
                "natural_product_like": "true" if is_nat else "false",
            }
            if is_nat:
                seen_nat = reservoir_add(natural, item, seen_nat, n_natural, rng)
            else:
                seen_other = reservoir_add(other, item, seen_other, n_other, rng)

    # If natural stratum undersampled, top up from other (and vice versa).
    selected = list(natural) + list(other)
    if len(selected) < n_total:
        print(
            f"WARNING: only {len(selected)} molecules after filters "
            f"(wanted {n_total}; nat={len(natural)}, other={len(other)})",
            flush=True,
        )

    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["catalog_id", "smiles", "vendor", "natural_product_like"],
        )
        writer.writeheader()
        writer.writerows(selected)

    stats = {
        "source": str(source),
        "output": str(output),
        "n_read": n_read,
        "n_passed_prefilter": n_kept,
        "n_natural_eligible": seen_nat,
        "n_other_eligible": seen_other,
        "n_written": len(selected),
        "n_natural_written": sum(1 for r in selected if r["natural_product_like"] == "true"),
        "seed": seed,
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("2026.01_Enamine_REAL_DB_13.6M.cxsmiles")
        / "2026.01_Enamine_REAL_DB_13.6M.cxsmiles",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/enamine_real_50k.csv"),
    )
    p.add_argument("--n-total", type=int, default=50_000)
    p.add_argument("--n-natural", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 1

    print(f"Building library from {args.source} → {args.output}", flush=True)
    stats = build_library(
        args.source,
        args.output,
        n_total=args.n_total,
        n_natural=args.n_natural,
        seed=args.seed,
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
