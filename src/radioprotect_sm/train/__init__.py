"""Training orchestration: splits, ensemble, metrics reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radioprotect_sm.metrics import multitask_metrics
from radioprotect_sm.models.baseline import BaselineConfig, MorganRFMultitask
from radioprotect_sm.models.chemprop_wrapper import chemprop_available, describe_backend
from radioprotect_sm.splits import attach_split_column, scaffold_split
from radioprotect_sm.utils import ensure_dir, load_yaml, project_root, set_global_seed, write_json

logger = logging.getLogger(__name__)


def _select_training_tasks(wide: pd.DataFrame, train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> list[str]:
    tasks = list(train_cfg.get("tasks", []))
    lit_cfg = data_cfg.get("tasks", {}).get("radioprotection_lit", {})
    if "radioprotection_lit" in tasks:
        n = int(wide["radioprotection_lit"].notna().sum()) if "radioprotection_lit" in wide.columns else 0
        threshold = int(lit_cfg.get("train_if_n_ge", 80))
        if n < threshold:
            logger.warning(
                "radioprotection_lit n=%d < %d — excluding from training; kept as external validation only.",
                n,
                threshold,
            )
            tasks = [t for t in tasks if t != "radioprotection_lit"]
    # Keep only tasks present with enough labels
    kept = []
    for t in tasks:
        if t in wide.columns and wide[t].notna().sum() >= 5:
            kept.append(t)
        else:
            logger.warning("Dropping task %s from training (insufficient labels).", t)
    return kept


def prepare_splits(
    wide_path: Path,
    train_cfg: dict[str, Any],
) -> pd.DataFrame:
    wide = pd.read_csv(wide_path)
    if "inchikey" not in wide.columns or "smiles" not in wide.columns:
        raise ValueError("labels_wide.csv must contain inchikey and smiles")
    split_cfg = train_cfg["split"]
    if split_cfg.get("strategy") != "scaffold":
        raise ValueError(
            f"Refusing undocumented split strategy '{split_cfg.get('strategy')}'. "
            "Use 'scaffold' (random only as an explicit ablation change)."
        )
    split = scaffold_split(
        wide["smiles"],
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
        seed=int(train_cfg.get("seed", 42)),
    )
    return attach_split_column(wide, split)


def train_ensemble(
    *,
    train_config: str | Path = "configs/train.yaml",
    data_config: str | Path = "configs/data_chembl.yaml",
    wide_path: Path | None = None,
    model_backend: str = "baseline",
) -> dict[str, Any]:
    root = project_root()
    train_cfg = load_yaml(train_config)
    data_cfg = load_yaml(data_config)
    set_global_seed(int(train_cfg.get("seed", 42)))

    wide_path = wide_path or root / train_cfg["paths"]["processed_dir"] / "labels_wide.csv"
    if not wide_path.exists():
        raise FileNotFoundError(f"Missing {wide_path}; run fetch/prepare first.")

    split_df = prepare_splits(wide_path, train_cfg)
    tasks = _select_training_tasks(split_df, train_cfg, data_cfg)
    if not tasks:
        raise RuntimeError("No trainable tasks after filtering.")

    runs_dir = ensure_dir(root / train_cfg["paths"]["runs_dir"])
    reports_dir = ensure_dir(root / train_cfg["paths"]["reports_dir"])
    split_out = ensure_dir(root / train_cfg["paths"]["processed_dir"]) / "labels_wide_split.csv"
    split_df.to_csv(split_out, index=False)

    backend_info = describe_backend(model_backend)
    logger.info("Model backend resolution: %s", backend_info)
    if model_backend == "chemprop" and not chemprop_available():
        logger.warning(
            "Chemprop requested but not installed; falling back to Morgan+RF baseline "
            "(explicit fallback, not a silent scientific change of architecture claims)."
        )

    bcfg = train_cfg.get("baseline", {})
    seeds = list(train_cfg.get("ensemble_seeds", [42, 43, 44, 45, 46]))

    train_mask = split_df["split"].eq("train")
    val_mask = split_df["split"].eq("val")
    test_mask = split_df["split"].eq("test")

    member_paths: list[str] = []
    val_preds: list[pd.DataFrame] = []
    test_preds: list[pd.DataFrame] = []

    for seed in seeds:
        cfg = BaselineConfig(
            n_bits=int(bcfg.get("n_bits", 2048)),
            radius=int(bcfg.get("radius", 2)),
            n_estimators=int(bcfg.get("n_estimators", 300)),
            max_depth=bcfg.get("max_depth"),
            min_samples_leaf=int(bcfg.get("min_samples_leaf", 2)),
            n_jobs=int(bcfg.get("n_jobs", -1)),
            seed=int(seed),
        )
        model = MorganRFMultitask(tasks=tasks, config=cfg)
        model.fit(split_df.loc[train_mask, "smiles"], split_df.loc[train_mask, tasks])
        out_path = runs_dir / f"baseline_seed{seed}.joblib"
        model.save(out_path)
        member_paths.append(str(out_path.relative_to(root)))
        val_preds.append(model.predict(split_df.loc[val_mask, "smiles"]))
        test_preds.append(model.predict(split_df.loc[test_mask, "smiles"]))

    def _ensemble_mean_std(preds: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        stack = {t: np.vstack([p[t].to_numpy() for p in preds]) for t in tasks}
        mean = pd.DataFrame({t: stack[t].mean(axis=0) for t in tasks})
        std = pd.DataFrame({t: stack[t].std(axis=0) for t in tasks})
        return mean, std

    val_mean, val_std = _ensemble_mean_std(val_preds)
    test_mean, test_std = _ensemble_mean_std(test_preds)

    y_val = {t: split_df.loc[val_mask, t].to_numpy(dtype=float) for t in tasks}
    y_test = {t: split_df.loc[test_mask, t].to_numpy(dtype=float) for t in tasks}
    val_metrics = multitask_metrics(y_val, {t: val_mean[t].to_numpy() for t in tasks})
    test_metrics = multitask_metrics(y_test, {t: test_mean[t].to_numpy() for t in tasks})

    # Persist predictions for audit
    val_out = split_df.loc[val_mask, ["inchikey", "smiles", "split"]].reset_index(drop=True)
    for t in tasks:
        val_out[f"y_true_{t}"] = y_val[t]
        val_out[f"y_pred_{t}"] = val_mean[t].to_numpy()
        val_out[f"y_std_{t}"] = val_std[t].to_numpy()
    test_out = split_df.loc[test_mask, ["inchikey", "smiles", "split"]].reset_index(drop=True)
    for t in tasks:
        test_out[f"y_true_{t}"] = y_test[t]
        test_out[f"y_pred_{t}"] = test_mean[t].to_numpy()
        test_out[f"y_std_{t}"] = test_std[t].to_numpy()
    val_out.to_csv(reports_dir / "val_predictions.csv", index=False)
    test_out.to_csv(reports_dir / "test_predictions.csv", index=False)

    report = {
        "model_backend": "morgan_rf_ensemble",
        "backend_resolution": backend_info,
        "chemprop_available": chemprop_available(),
        "tasks": tasks,
        "ensemble_seeds": seeds,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "member_models": member_paths,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_file": str(split_out.relative_to(root)),
        "notes": [
            "Primary stage-1 backend is Morgan fingerprint + RandomForest ensemble.",
            "Chemprop is optional; absence does not change label definitions or split policy.",
            "Metrics computed only on finite labels per task (multitask mask).",
        ],
    }
    report_path = reports_dir / "train_report.json"
    write_json(report_path, report)

    # Human-readable markdown summary
    lines = [
        "# Training report",
        "",
        f"- Backend: `{report['model_backend']}`",
        f"- Tasks: {', '.join(tasks)}",
        f"- Ensemble seeds: {seeds}",
        f"- N train/val/test: {report['n_train']}/{report['n_val']}/{report['n_test']}",
        "",
        "## Held-out test metrics",
        "",
    ]
    for task, metrics in test_metrics.items():
        lines.append(f"### {task}")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) and np.isfinite(v) else f"- {k}: {v}")
        lines.append("")
    (reports_dir / "train_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Ensemble bundle for screening
    bundle = {
        "tasks": tasks,
        "member_paths": [str(root / p) for p in member_paths],
        "baseline_config": bcfg,
    }
    write_json(runs_dir / "ensemble_bundle.json", bundle)
    return report
