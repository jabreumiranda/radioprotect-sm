"""Virtual screening, multi-objective ranking, and diversity selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radioprotect_sm.data.canonicalize import (
    morgan_fingerprint,
    passes_druglike_filters,
    tanimoto,
)
from radioprotect_sm.models.baseline import MorganRFMultitask
from radioprotect_sm.utils import ensure_dir, load_yaml, project_root, write_json

logger = logging.getLogger(__name__)

# Aim-aligned hit lists: isolate each proxy + joint (both aims).
AIM1_TASK = "dna_binding"
AIM2_TASK = "ros_scavenging"
HIT_LIST_KEYS = ("dna_binding", "ros_scavenging", "dna_and_ros")


def load_ensemble(bundle_path: Path) -> tuple[list[str], list[MorganRFMultitask]]:
    import json

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    tasks = bundle["tasks"]
    models = [MorganRFMultitask.load(Path(p)) for p in bundle["member_paths"]]
    return tasks, models


def ensemble_predict(models: list[MorganRFMultitask], smiles: list[str], tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = [m.predict(smiles) for m in models]
    mean = pd.DataFrame({t: np.mean([p[t].to_numpy() for p in preds], axis=0) for t in tasks})
    std = pd.DataFrame({t: np.std([p[t].to_numpy() for p in preds], axis=0) for t in tasks})
    return mean, std


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def multiobjective_score(
    mean: pd.DataFrame,
    std: pd.DataFrame,
    weights: dict[str, float],
    *,
    uncertainty_penalty: float = 0.25,
    zscore_within_library: bool = True,
) -> pd.Series:
    """Explicit weighted sum of task predictions minus uncertainty penalty.

    Positive weights: higher predicted value preferred.
    Negative weights (e.g. cytotox): higher potency penalized.
    """
    score = pd.Series(np.zeros(len(mean)), index=mean.index, dtype=float)
    unc = pd.Series(np.zeros(len(mean)), index=mean.index, dtype=float)
    for task, w in weights.items():
        if task not in mean.columns:
            logger.warning("Ranking weight task '%s' missing from predictions; skipping.", task)
            continue
        col = mean[task]
        if zscore_within_library:
            col = zscore(col)
        score = score + float(w) * col
        if task in std.columns:
            unc = unc + std[task].astype(float)
    n_tasks = max(1, len([t for t in weights if t in std.columns]))
    score = score - float(uncertainty_penalty) * (unc / n_tasks)
    return score


def build_hit_list_weights(base_weights: dict[str, float]) -> dict[str, dict[str, float]]:
    """Derive Aim-1-only, Aim-2-only, and joint weight maps.

    Negative weights (e.g. cytotox safety penalty) are kept in every list.
    """
    safety = {t: w for t, w in base_weights.items() if float(w) < 0}
    aim1_w = float(base_weights.get(AIM1_TASK, 1.0))
    aim2_w = float(base_weights.get(AIM2_TASK, 1.0))
    return {
        "dna_binding": {AIM1_TASK: aim1_w, **safety},
        "ros_scavenging": {AIM2_TASK: aim2_w, **safety},
        "dna_and_ros": {
            AIM1_TASK: aim1_w,
            AIM2_TASK: aim2_w,
            **safety,
        },
    }


def diversity_select(
    smiles: list[str],
    scores: np.ndarray,
    *,
    tanimoto_threshold: float = 0.60,
    max_hits: int = 50,
    n_bits: int = 2048,
    radius: int = 2,
) -> list[int]:
    """Greedy max-score selection with Tanimoto diversity constraint."""
    order = list(np.argsort(-scores))
    fps: list[np.ndarray | None] = [
        morgan_fingerprint(s, n_bits=n_bits, radius=radius) for s in smiles
    ]
    selected: list[int] = []
    selected_fps: list[np.ndarray] = []
    for idx in order:
        fp = fps[idx]
        if fp is None:
            continue
        if any(tanimoto(fp, sfp) >= tanimoto_threshold for sfp in selected_fps):
            continue
        selected.append(idx)
        selected_fps.append(fp)
        if len(selected) >= max_hits:
            break
    return selected


_SELECTION_REASONS = {
    "dna_binding": (
        "score(dna_binding↑ [, cytotox↓]) + ensemble uncertainty penalty + Tanimoto diversity"
    ),
    "ros_scavenging": (
        "score(ros_scavenging↑ [, cytotox↓]) + ensemble uncertainty penalty + Tanimoto diversity"
    ),
    "dna_and_ros": (
        "multiobjective_score(dna_binding↑, ros_scavenging↑ [, cytotox↓]) "
        "+ ensemble uncertainty penalty + Tanimoto diversity"
    ),
}


def _select_hits(
    candidates: pd.DataFrame,
    score_col: str,
    *,
    list_key: str,
    div: dict[str, Any],
) -> pd.DataFrame:
    selected_idx = diversity_select(
        candidates["smiles"].tolist(),
        candidates[score_col].to_numpy(),
        tanimoto_threshold=float(div["tanimoto_threshold"]),
        max_hits=int(div["max_hits"]),
        n_bits=int(div.get("n_bits", 2048)),
        radius=int(div.get("radius", 2)),
    )
    hits = candidates.iloc[selected_idx].copy()
    hits.insert(0, "rank", np.arange(1, len(hits) + 1))
    hits["hit_list"] = list_key
    hits["score"] = hits[score_col]
    hits["selection_reason"] = _SELECTION_REASONS[list_key]
    return hits


def run_screen(
    *,
    screen_config: str | Path = "configs/screen.yaml",
    bundle_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Screen library and write three Aim-aligned hit CSVs + full scores."""
    root = project_root()
    cfg = load_yaml(screen_config)
    bundle_path = bundle_path or root / "runs" / "ensemble_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing ensemble bundle: {bundle_path}. Train first.")

    tasks, models = load_ensemble(bundle_path)
    lib_cfg = cfg["library"]
    lib_path = Path(lib_cfg["path"])
    if not lib_path.is_absolute():
        lib_path = root / lib_path
    if not lib_path.exists():
        raise FileNotFoundError(f"Purchasable library not found: {lib_path}")

    lib = pd.read_csv(lib_path)
    id_col = lib_cfg["id_col"]
    smiles_col = lib_cfg["smiles_col"]
    vendor_col = lib_cfg.get("vendor_col")

    filt = cfg["filters"]
    kept_rows: list[dict[str, Any]] = []
    for _, row in lib.iterrows():
        ok, meta = passes_druglike_filters(
            row[smiles_col],
            min_heavy_atoms=int(filt["min_heavy_atoms"]),
            max_heavy_atoms=int(filt["max_heavy_atoms"]),
            min_molwt=float(filt["min_molwt"]),
            max_molwt=float(filt["max_molwt"]),
            max_logp=float(filt["max_logp"]),
        )
        if not ok:
            continue
        kept_rows.append(
            {
                "catalog_id": row[id_col],
                "vendor": row[vendor_col] if vendor_col and vendor_col in row else None,
                **meta,
            }
        )
    candidates = pd.DataFrame(kept_rows)
    if candidates.empty:
        raise RuntimeError("No library molecules passed filters.")

    mean, std = ensemble_predict(models, candidates["smiles"].tolist(), tasks)
    for t in tasks:
        candidates[f"pred_{t}"] = mean[t].to_numpy()
        candidates[f"std_{t}"] = std[t].to_numpy()

    rank_cfg = cfg["ranking"]
    list_weights = build_hit_list_weights(dict(rank_cfg["weights"]))
    unc = float(rank_cfg.get("uncertainty_penalty", 0.25))
    zlib = bool(rank_cfg.get("zscore_within_library", True))

    score_cols: dict[str, str] = {}
    for key, weights in list_weights.items():
        col = f"score_{key}"
        candidates[col] = multiobjective_score(
            mean,
            std,
            weights=weights,
            uncertainty_penalty=unc,
            zscore_within_library=zlib,
        )
        score_cols[key] = col

    # Primary combined score column for convenience / sorting of full library.
    candidates["score"] = candidates[score_cols["dna_and_ros"]]

    div = cfg["diversity"]
    hit_tables: dict[str, pd.DataFrame] = {}
    for key in HIT_LIST_KEYS:
        hit_tables[key] = _select_hits(
            candidates, score_cols[key], list_key=key, div=div
        )

    out_cfg = cfg["output"]
    scores_path = Path(out_cfg["scores_csv"])
    if not scores_path.is_absolute():
        scores_path = root / scores_path
    ensure_dir(scores_path.parent)

    default_hit_paths = {
        "dna_binding": "outputs/hits_dna_binding.csv",
        "ros_scavenging": "outputs/hits_ros_scavenging.csv",
        "dna_and_ros": "outputs/hits_dna_and_ros.csv",
    }
    configured_hits = out_cfg.get("hits") or {}
    hit_paths: dict[str, Path] = {}
    for key in HIT_LIST_KEYS:
        rel = configured_hits.get(key, default_hit_paths[key])
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        ensure_dir(path.parent)
        hit_tables[key].to_csv(path, index=False)
        hit_paths[key] = path

    # Backward-compatible alias of the joint list.
    alias = Path(out_cfg.get("hits_csv", "outputs/hits_for_purchase.csv"))
    if not alias.is_absolute():
        alias = root / alias
    ensure_dir(alias.parent)
    hit_tables["dna_and_ros"].to_csv(alias, index=False)

    candidates.sort_values("score", ascending=False).to_csv(scores_path, index=False)

    write_json(
        root / "outputs" / "reports" / "screen_report.json",
        {
            "n_library": int(len(lib)),
            "n_after_filters": int(len(candidates)),
            "n_hits": {k: int(len(hit_tables[k])) for k in HIT_LIST_KEYS},
            "list_weights": list_weights,
            "base_weights": rank_cfg["weights"],
            "uncertainty_penalty": rank_cfg.get("uncertainty_penalty"),
            "tanimoto_threshold": div["tanimoto_threshold"],
            "hits_csv": {k: str(p.relative_to(root)) for k, p in hit_paths.items()},
            "hits_csv_alias_joint": str(alias.relative_to(root)),
            "scores_csv": str(scores_path.relative_to(root)),
            "caveat": (
                "Hits are computational prioritization hypotheses for purchase/testing. "
                "Proxy ChEMBL labels ≠ experimental radioprotection. "
                "Single-aim lists can surface specialists that the joint score underranks."
            ),
        },
    )
    return hit_tables
