"""Schema for future laboratory experimental data — no fabricated values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from radioprotect_sm.utils import load_yaml, project_root


REQUIRED_COLUMNS = [
    "compound_id",
    "smiles",
    "assay_name",
    "endpoint",
    "value",
    "units",
    "cell_line",
    "radiation_dose_gy",
    "pretreatment_hours",
    "replicate",
    "measured_at",
    "operator",
    "notes",
]


@dataclass(frozen=True)
class LabSchemaValidation:
    ok: bool
    missing_columns: list[str]
    n_rows: int
    errors: list[str]


def validate_lab_dataframe(df: pd.DataFrame) -> LabSchemaValidation:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if "value" in df.columns:
        # Refuse non-numeric fabricated placeholders like 'TBD'
        bad = df["value"].map(lambda x: not _is_number(x))
        if bad.any():
            errors.append("Column 'value' contains non-numeric entries; refusing load.")
    if "data_source" in df.columns and (df["data_source"] != "lab").any():
        errors.append("Lab data must have data_source='lab' when the column is present.")
    return LabSchemaValidation(
        ok=not errors and not missing,
        missing_columns=missing,
        n_rows=len(df),
        errors=errors,
    )


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def load_lab_data(config_path: str | Path = "configs/lab_data.yaml") -> pd.DataFrame | None:
    """Load lab CSV if enabled and present. Returns None when disabled/empty.

    Never invents rows.
    """
    cfg = load_yaml(config_path)
    if not cfg.get("enabled"):
        return None
    path = cfg.get("path")
    if not path:
        raise ValueError("lab_data.enabled=true but path is null")
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Lab data file not found: {p}")
    df = pd.read_csv(p)
    result = validate_lab_dataframe(df)
    if not result.ok:
        raise ValueError(f"Lab schema validation failed: {result.errors + result.missing_columns}")
    df = df.copy()
    df["data_source"] = "lab"
    return df


def empty_lab_template() -> pd.DataFrame:
    """Return an empty frame with the required schema (headers only)."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)
