"""Optional Chemprop multitask wrapper.

Chemprop is an optional dependency (`pip install radioprotect-sm[chemprop]`).
When unavailable, callers must fall back explicitly — never claim Chemprop was used.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def chemprop_available() -> bool:
    try:
        import chemprop  # noqa: F401
        import torch  # noqa: F401

        return shutil.which("chemprop") is not None or True
    except Exception:
        return False


def write_chemprop_csv(df: pd.DataFrame, tasks: list[str], path: Path) -> None:
    """Write Chemprop-style CSV: smiles + task columns (empty string for missing)."""
    out = pd.DataFrame({"smiles": df["smiles"]})
    for t in tasks:
        col = df[t].astype(object).where(df[t].notna(), other="")
        out[t] = col
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def train_chemprop_ensemble_member(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tasks: list[str],
    *,
    seed: int,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
) -> Path:
    """Train one Chemprop model via CLI if installed.

    Raises RuntimeError with a clear message if Chemprop is not available.
    """
    if not chemprop_available():
        raise RuntimeError(
            "Chemprop/torch not installed. Install with: pip install 'radioprotect-sm[chemprop]'"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_csv = tmp_path / "train.csv"
        val_csv = tmp_path / "val.csv"
        write_chemprop_csv(train_df, tasks, train_csv)
        write_chemprop_csv(val_df, tasks, val_csv)
        cmd = [
            "chemprop",
            "train",
            "--data-path",
            str(train_csv),
            "--separate-val-path",
            str(val_csv),
            "--dataset-type",
            "regression",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--pytorch-seed",
            str(seed),
            "--save-dir",
            str(out_dir),
        ]
        logger.info("Running Chemprop: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "chemprop CLI not found on PATH after import. "
                "Ensure the chemprop package exposes the `chemprop` command."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Chemprop training failed with code {exc.returncode}") from exc
    return out_dir


def describe_backend(requested: str) -> dict[str, Any]:
    available = chemprop_available()
    if requested == "chemprop" and not available:
        return {
            "requested": requested,
            "resolved": "morgan_rf_ensemble",
            "chemprop_available": False,
            "note": "Explicit fallback to Morgan+RF; Chemprop extras not installed.",
        }
    if requested == "chemprop" and available:
        return {
            "requested": requested,
            "resolved": "chemprop_available_but_baseline_orchestrated",
            "chemprop_available": True,
            "note": (
                "Chemprop is installed. Stage-1 default orchestration still trains the "
                "documented Morgan+RF ensemble for reproducibility without GPU. "
                "Use train_chemprop_ensemble_member() for Chemprop weight export."
            ),
        }
    return {
        "requested": requested,
        "resolved": "morgan_rf_ensemble",
        "chemprop_available": available,
        "note": "Baseline Morgan+RF ensemble (stage-1 default).",
    }
