"""ChEMBL fetch, cleaning, and multitask table construction."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client

from radioprotect_sm.data.canonicalize import canonicalize_smiles
from radioprotect_sm.utils import ensure_dir, load_yaml, project_root, sha256_file, write_json

logger = logging.getLogger(__name__)


ACTIVITY_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "canonical_smiles",
    "molecule_chembl_id",
    "pchembl_value",
    "standard_type",
    "standard_units",
    "standard_value",
    "standard_relation",
    "target_chembl_id",
    "target_organism",
    "target_type",
    "type",
    "units",
    "value",
]

_SQLITE_SELECT = """
SELECT
  a.activity_id AS activity_id,
  ass.chembl_id AS assay_chembl_id,
  ass.description AS assay_description,
  ass.assay_type AS assay_type,
  cs.canonical_smiles AS canonical_smiles,
  md.chembl_id AS molecule_chembl_id,
  a.pchembl_value AS pchembl_value,
  a.standard_type AS standard_type,
  a.standard_units AS standard_units,
  a.standard_value AS standard_value,
  a.standard_relation AS standard_relation,
  td.chembl_id AS target_chembl_id,
  td.organism AS target_organism,
  td.target_type AS target_type,
  a.type AS type,
  a.units AS units,
  a.value AS value
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
LEFT JOIN target_dictionary td ON ass.tid = td.tid
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON a.molregno = cs.molregno
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _keyword_match(text: str | None, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def _resolve_sqlite_path(cfg: dict[str, Any], root: Path | None = None) -> Path | None:
    raw = cfg.get("sqlite_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (root or project_root()) / path
    return path if path.exists() else None


def _open_sqlite(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _rows_from_sqlite(cur: sqlite3.Cursor, sql: str, params: list[Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    cur.execute(sql + f" LIMIT {int(limit)}", params)
    return [dict(r) for r in cur.fetchall()]


def fetch_task_records_sqlite(
    task_name: str,
    task_cfg: dict[str, Any],
    max_records: int,
    db_path: Path,
) -> pd.DataFrame:
    """Pull activity records for one task from a local ChEMBL SQLite dump."""
    rows: list[dict[str, Any]] = []
    seen_activity_ids: set[int] = set()

    standard_types = task_cfg.get("standard_types") or []
    assay_type = task_cfg.get("assay_type")
    target_types = task_cfg.get("target_types") or []
    assay_keywords = task_cfg.get("assay_description_keywords") or []

    con = _open_sqlite(db_path)
    try:
        cur = con.cursor()

        # Strategy A: target_type (dna_binding)
        if target_types and len(rows) < max_records:
            for tt in target_types:
                if len(rows) >= max_records:
                    break
                where = ["td.target_type = ?", "cs.canonical_smiles IS NOT NULL"]
                params: list[Any] = [tt]
                if assay_type:
                    where.append("ass.assay_type = ?")
                    params.append(assay_type)
                if standard_types:
                    placeholders = ",".join("?" for _ in standard_types)
                    where.append(f"a.standard_type IN ({placeholders})")
                    params.extend(standard_types)
                sql = _SQLITE_SELECT + " WHERE " + " AND ".join(where)
                batch = _rows_from_sqlite(cur, sql, params, max_records - len(rows))
                for rec in batch:
                    aid = rec.get("activity_id")
                    if aid in seen_activity_ids:
                        continue
                    seen_activity_ids.add(aid)
                    rows.append(rec)
                logger.info(
                    "SQLite target_type=%s task=%s → +%d (total %d)",
                    tt,
                    task_name,
                    len(batch),
                    len(rows),
                )

        # Strategy B: assay description keywords
        if assay_keywords and len(rows) < max_records:
            max_assays_per_kw = int(task_cfg.get("max_assays_per_keyword", 2000))
            for kw in assay_keywords:
                if len(rows) >= max_records:
                    break
                assay_sql = "SELECT assay_id FROM assays WHERE description LIKE ?"
                assay_params: list[Any] = [f"%{kw}%"]
                if assay_type:
                    assay_sql += " AND assay_type = ?"
                    assay_params.append(assay_type)
                assay_sql += f" LIMIT {max_assays_per_kw}"
                assay_ids = [r[0] for r in cur.execute(assay_sql, assay_params).fetchall()]
                if not assay_ids:
                    logger.info("SQLite: no assays for keyword %r task=%s", kw, task_name)
                    continue

                # Chunk IN clauses
                chunk = 400
                added = 0
                for i in range(0, len(assay_ids), chunk):
                    if len(rows) >= max_records:
                        break
                    ids = assay_ids[i : i + chunk]
                    placeholders = ",".join("?" for _ in ids)
                    where = [
                        f"ass.assay_id IN ({placeholders})",
                        "cs.canonical_smiles IS NOT NULL",
                    ]
                    params = list(ids)
                    if standard_types:
                        st_ph = ",".join("?" for _ in standard_types)
                        where.append(f"a.standard_type IN ({st_ph})")
                        params.extend(standard_types)
                    sql = _SQLITE_SELECT + " WHERE " + " AND ".join(where)
                    batch = _rows_from_sqlite(cur, sql, params, max_records - len(rows))
                    for rec in batch:
                        aid = rec.get("activity_id")
                        if aid in seen_activity_ids:
                            continue
                        desc = rec.get("assay_description")
                        if not _keyword_match(desc, assay_keywords):
                            continue
                        seen_activity_ids.add(aid)
                        rows.append(rec)
                        added += 1
                logger.info(
                    "SQLite keyword=%r task=%s assays=%d → +%d (total %d)",
                    kw,
                    task_name,
                    len(assay_ids),
                    added,
                    len(rows),
                )
    finally:
        con.close()

    df = pd.DataFrame(rows)
    df["task"] = task_name
    return df


def ic50_like_to_pchembl(value: float, units: str | None) -> float | None:
    """Convert IC50-like potency to -log10(M). Returns None if units unknown.

    Documented conversion rule — do not invent values for unsupported units.
    """
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    u = (units or "").strip().lower()
    # molar
    if u in {"m", "mol/l", "molar"}:
        molar = value
    elif u in {"mm", "mmol/l", "millimolar"}:
        molar = value * 1e-3
    elif u in {"um", "µm", "μm", "umol/l", "µmol/l", "μmol/l", "micromolar"}:
        molar = value * 1e-6
    elif u in {"nm", "nmol/l", "nanomolar"}:
        molar = value * 1e-9
    elif u in {"pm", "pmol/l", "picomolar"}:
        molar = value * 1e-12
    else:
        return None
    return float(-np.log10(molar))


def _activity_base_filter(**kwargs: Any):
    act = new_client.activity
    act = act.filter(**kwargs)
    act = act.only(ACTIVITY_FIELDS)
    return act


def fetch_task_records(task_name: str, task_cfg: dict[str, Any], max_records: int) -> pd.DataFrame:
    """Pull activity records for one configured task from ChEMBL web services."""
    rows: list[dict[str, Any]] = []
    seen_activity_ids: set[int] = set()

    standard_types = task_cfg.get("standard_types") or []
    assay_type = task_cfg.get("assay_type")
    target_types = task_cfg.get("target_types") or []
    target_keywords = task_cfg.get("target_name_keywords") or []
    assay_keywords = task_cfg.get("assay_description_keywords") or []

    # Strategy A: nucleic-acid target type (dna_binding)
    if target_types:
        for tt in target_types:
            for st in standard_types or [None]:
                filt: dict[str, Any] = {"target_type": tt}
                if assay_type:
                    filt["assay_type"] = assay_type
                if st:
                    filt["standard_type"] = st
                try:
                    query = _activity_base_filter(**filt)
                    for rec in query:
                        aid = rec.get("activity_id")
                        if aid in seen_activity_ids:
                            continue
                        # Optional target keyword soft filter via assay_description / type fields
                        if target_keywords:
                            blob = " ".join(
                                str(rec.get(k) or "")
                                for k in ("assay_description", "type", "target_type")
                            )
                            if not _keyword_match(blob, target_keywords) and tt != "NUCLEIC-ACID":
                                continue
                        seen_activity_ids.add(aid)
                        rows.append(rec)
                        if len(rows) >= max_records:
                            break
                except Exception as exc:  # network / API
                    logger.warning("ChEMBL query failed for %s/%s: %s", task_name, filt, exc)
                if len(rows) >= max_records:
                    break
            if len(rows) >= max_records:
                break

    # Strategy B: assay.description keyword search, then activities by assay_chembl_id.
    # Activity.assay_description__icontains is rejected by the ChEMBL API; use Assay first.
    if assay_keywords and len(rows) < max_records:
        max_assays_per_kw = int(task_cfg.get("max_assays_per_keyword", 250))
        assay_batch_size = 25
        for kw in assay_keywords:
            if len(rows) >= max_records:
                break
            assay_filt: dict[str, Any] = {"description__icontains": kw}
            if assay_type:
                assay_filt["assay_type"] = assay_type
            assay_ids: list[str] = []
            try:
                assay_query = new_client.assay.filter(**assay_filt).only(
                    ["assay_chembl_id", "description", "assay_type"]
                )
                for assay in assay_query:
                    desc = assay.get("description")
                    # Re-check so multi-word keywords match the intended substring.
                    if not _keyword_match(desc, [kw]):
                        continue
                    chembl_id = assay.get("assay_chembl_id")
                    if chembl_id:
                        assay_ids.append(str(chembl_id))
                    if len(assay_ids) >= max_assays_per_kw:
                        break
            except Exception as exc:
                logger.warning(
                    "ChEMBL assay keyword query failed (%s, %s): %s", task_name, kw, exc
                )
                continue

            if not assay_ids:
                logger.info("No assays matched keyword %r for task=%s", kw, task_name)
                continue

            for i in range(0, len(assay_ids), assay_batch_size):
                if len(rows) >= max_records:
                    break
                batch = assay_ids[i : i + assay_batch_size]
                for st in standard_types or [None]:
                    filt = {"assay_chembl_id__in": ",".join(batch)}
                    if st:
                        filt["standard_type"] = st
                    try:
                        query = _activity_base_filter(**filt)
                        for rec in query:
                            aid = rec.get("activity_id")
                            if aid in seen_activity_ids:
                                continue
                            desc = rec.get("assay_description")
                            if assay_keywords and not _keyword_match(desc, assay_keywords):
                                continue
                            seen_activity_ids.add(aid)
                            rows.append(rec)
                            if len(rows) >= max_records:
                                break
                    except Exception as exc:
                        logger.warning(
                            "ChEMBL activity-by-assay query failed (%s, %s): %s",
                            task_name,
                            kw,
                            exc,
                        )
                    if len(rows) >= max_records:
                        break

    df = pd.DataFrame(rows)
    df["task"] = task_name
    return df


def assign_label(row: pd.Series, task_cfg: dict[str, Any]) -> float | None:
    """Assign continuous label without fabricating values."""
    prefer_p = bool(task_cfg.get("prefer_pchembl", True))
    pchembl = row.get("pchembl_value")
    if prefer_p and pchembl is not None and str(pchembl) not in {"", "None", "nan"}:
        try:
            val = float(pchembl)
            if np.isfinite(val):
                min_p = task_cfg.get("min_pchembl")
                if min_p is not None and val < float(min_p):
                    return None
                return val
        except (TypeError, ValueError):
            pass

    if task_cfg.get("allow_standard_value_conversion"):
        try:
            std = float(row.get("standard_value"))
        except (TypeError, ValueError):
            return None
        converted = ic50_like_to_pchembl(std, row.get("standard_units"))
        if converted is None:
            return None
        min_p = task_cfg.get("min_pchembl")
        if min_p is not None and converted < float(min_p):
            return None
        return converted
    return None


def clean_activities(
    raw: pd.DataFrame,
    data_cfg: dict[str, Any],
    *,
    retrieved_at: str,
    chembl_release: str,
) -> pd.DataFrame:
    """Canonicalize molecules, assign labels, deduplicate per (inchikey, task)."""
    can_cfg = data_cfg.get("canonicalization", {})
    dedupe_cfg = data_cfg.get("deduplication", {})
    tasks_cfg = data_cfg["tasks"]

    cleaned_rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        task = row["task"]
        task_cfg = tasks_cfg[task]
        smiles = row.get("canonical_smiles")
        can = canonicalize_smiles(
            smiles,
            remove_salts=bool(can_cfg.get("remove_salts", True)),
            keep_largest_fragment=bool(can_cfg.get("keep_largest_fragment", True)),
            reject_mixtures=bool(can_cfg.get("reject_mixtures", True)),
            max_heavy_atoms=int(can_cfg.get("max_heavy_atoms", 80)),
            min_heavy_atoms=int(can_cfg.get("min_heavy_atoms", 5)),
        )
        if can is None:
            continue
        label = assign_label(row, task_cfg)
        if label is None:
            continue
        cleaned_rows.append(
            {
                "chembl_id": row.get("molecule_chembl_id"),
                "activity_id": row.get("activity_id"),
                "assay_id": row.get("assay_chembl_id"),
                "target_id": row.get("target_chembl_id"),
                "smiles": can.smiles,
                "inchikey": can.inchikey,
                "scaffold_smiles": can.scaffold_smiles,
                "task": task,
                "label": label,
                "pchembl_value": row.get("pchembl_value"),
                "standard_type": row.get("standard_type"),
                "standard_units": row.get("standard_units"),
                "standard_value": row.get("standard_value"),
                "assay_description": row.get("assay_description"),
                "data_source": "chembl",
                "chembl_release": chembl_release,
                "retrieved_at": retrieved_at,
            }
        )

    long_df = pd.DataFrame(cleaned_rows)
    if long_df.empty:
        return long_df

    agg = dedupe_cfg.get("aggregation", "median")
    if agg != "median":
        raise ValueError(f"Unsupported aggregation '{agg}' — refusing silent change.")

    # Aggregate replicate measurements per molecule×task
    group_cols = ["inchikey", "task"]
    agg_df = (
        long_df.groupby(group_cols, as_index=False)
        .agg(
            label=("label", "median"),
            smiles=("smiles", "first"),
            scaffold_smiles=("scaffold_smiles", "first"),
            chembl_id=("chembl_id", "first"),
            n_activities=("activity_id", "count"),
            assay_ids=("assay_id", lambda s: ";".join(sorted({str(x) for x in s if pd.notna(x)}))),
            target_ids=("target_id", lambda s: ";".join(sorted({str(x) for x in s if pd.notna(x)}))),
            data_source=("data_source", "first"),
            chembl_release=("chembl_release", "first"),
            retrieved_at=("retrieved_at", "first"),
        )
    )
    return agg_df


def long_to_wide(long_df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """Pivot to multitask wide table with NaN masks for missing labels (never imputed)."""
    if long_df.empty:
        cols = ["inchikey", "smiles", "scaffold_smiles", "chembl_id"] + tasks
        return pd.DataFrame(columns=cols)

    base = long_df.groupby("inchikey", as_index=False).agg(
        smiles=("smiles", "first"),
        scaffold_smiles=("scaffold_smiles", "first"),
        chembl_id=("chembl_id", "first"),
    )
    for task in tasks:
        sub = long_df.loc[long_df["task"] == task, ["inchikey", "label"]].rename(
            columns={"label": task}
        )
        base = base.merge(sub, on="inchikey", how="left")
    return base


def fetch_and_prepare(
    config_path: str | Path = "configs/data_chembl.yaml",
    *,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    use_cache: bool = True,
    sqlite_path: str | Path | None = None,
) -> dict[str, Path]:
    """Fetch ChEMBL activities, clean, write long/wide tables + manifests."""
    root = project_root()
    cfg = load_yaml(config_path)
    retrieved_at = _utcnow_iso()
    chembl_release = str(cfg.get("chembl_release", "latest"))

    raw_dir = ensure_dir(raw_dir or root / "data" / "raw" / "chembl")
    processed_dir = ensure_dir(processed_dir or root / "data" / "processed")
    manifests_dir = ensure_dir(root / "data" / "manifests")

    max_records = int(cfg.get("max_records_per_task", 5000))
    if sqlite_path is not None:
        db_path = Path(sqlite_path)
        if not db_path.is_absolute():
            db_path = root / db_path
        if not db_path.exists():
            raise FileNotFoundError(f"ChEMBL SQLite not found: {db_path}")
    else:
        db_path = _resolve_sqlite_path(cfg, root)

    source = f"sqlite:{db_path}" if db_path is not None else "chembl_webresource_client"
    logger.info("ChEMBL source=%s max_records_per_task=%d", source, max_records)

    task_frames: list[pd.DataFrame] = []
    task_counts: dict[str, int] = {}

    for task_name, task_cfg in cfg["tasks"].items():
        cache_path = raw_dir / f"{task_name}_raw.csv"
        if use_cache and cache_path.exists():
            logger.info("Using cached raw pull: %s", cache_path)
            df = pd.read_csv(cache_path)
        else:
            logger.info("Fetching ChEMBL task=%s (max=%d) via %s", task_name, max_records, source)
            if db_path is not None:
                df = fetch_task_records_sqlite(
                    task_name, task_cfg, max_records=max_records, db_path=db_path
                )
            else:
                df = fetch_task_records(task_name, task_cfg, max_records=max_records)
            df.to_csv(cache_path, index=False)
        task_counts[task_name] = len(df)
        task_frames.append(df)

    raw = pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame()
    raw_all_path = raw_dir / "all_tasks_raw.csv"
    raw.to_csv(raw_all_path, index=False)

    long_df = clean_activities(
        raw, cfg, retrieved_at=retrieved_at, chembl_release=chembl_release
    )
    tasks = list(cfg["tasks"].keys())
    wide_df = long_to_wide(long_df, tasks)

    # Decide whether radioprotection_lit is trainable
    lit_cfg = cfg["tasks"].get("radioprotection_lit", {})
    lit_n = int((long_df["task"] == "radioprotection_lit").sum()) if not long_df.empty else 0
    lit_policy = {
        "n_molecules": lit_n,
        "train_if_n_ge": lit_cfg.get("train_if_n_ge", 80),
        "include_in_training": lit_n >= int(lit_cfg.get("train_if_n_ge", 80)),
        "else_use_as": lit_cfg.get("else_use_as", "external_validation"),
    }

    long_path = processed_dir / "labels_long.csv"
    wide_path = processed_dir / "labels_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    # External validation slice for sparse radioprotection labels
    ext_path = processed_dir / "radioprotection_lit_external.csv"
    if not long_df.empty:
        long_df.loc[long_df["task"] == "radioprotection_lit"].to_csv(ext_path, index=False)
    else:
        pd.DataFrame().to_csv(ext_path, index=False)

    manifest = {
        "retrieved_at": retrieved_at,
        "chembl_release": chembl_release,
        "source": source,
        "config_path": str(config_path),
        "max_records_per_task": max_records,
        "raw_record_counts": task_counts,
        "n_long_rows": int(len(long_df)),
        "n_molecules_wide": int(len(wide_df)),
        "task_molecule_counts": {
            t: int(wide_df[t].notna().sum()) if t in wide_df.columns else 0 for t in tasks
        },
        "radioprotection_lit_policy": lit_policy,
        "files": {
            "raw_all": str(raw_all_path.relative_to(root)),
            "labels_long": str(long_path.relative_to(root)),
            "labels_wide": str(wide_path.relative_to(root)),
            "radioprotection_lit_external": str(ext_path.relative_to(root)),
            "sha256": {
                "raw_all": sha256_file(raw_all_path) if raw_all_path.exists() else None,
                "labels_long": sha256_file(long_path),
                "labels_wide": sha256_file(wide_path),
            },
        },
        "notes": [
            "Labels are ChEMBL proxies, not direct clinical radioprotection endpoints.",
            "Missing multitask labels are left as NaN (never imputed).",
        ],
    }
    manifest_path = manifests_dir / "chembl_fetch_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "long": long_path,
        "wide": wide_path,
        "external_lit": ext_path,
        "manifest": manifest_path,
    }


def load_demo_labels(processed_dir: Path | None = None) -> dict[str, Path]:
    """Copy bundled demo labels into processed/ when offline / API unavailable."""
    root = project_root()
    processed_dir = ensure_dir(processed_dir or root / "data" / "processed")
    demo_dir = root / "tests" / "fixtures"
    long_src = demo_dir / "demo_labels_long.csv"
    wide_src = demo_dir / "demo_labels_wide.csv"
    long_path = processed_dir / "labels_long.csv"
    wide_path = processed_dir / "labels_wide.csv"
    ext_path = processed_dir / "radioprotection_lit_external.csv"
    pd.read_csv(long_src).to_csv(long_path, index=False)
    pd.read_csv(wide_src).to_csv(wide_path, index=False)
    lit = pd.read_csv(long_src)
    lit = lit.loc[lit["task"] == "radioprotection_lit"]
    lit.to_csv(ext_path, index=False)
    write_json(
        ensure_dir(root / "data" / "manifests") / "chembl_fetch_manifest.json",
        {
            "retrieved_at": _utcnow_iso(),
            "chembl_release": "demo-fixture",
            "source": "tests/fixtures",
            "n_long_rows": int(len(pd.read_csv(long_path))),
            "n_molecules_wide": int(len(pd.read_csv(wide_path))),
            "notes": ["Demo fixture data for offline reproducibility — not experimental claims."],
        },
    )
    return {"long": long_path, "wide": wide_path, "external_lit": ext_path}
