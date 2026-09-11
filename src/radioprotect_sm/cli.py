"""Command-line interface for the radioprotect-sm pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from radioprotect_sm.utils import project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("radioprotect_sm")


@click.group()
@click.version_option(package_name="radioprotect-sm")
def main() -> None:
    """Small-molecule radioprotectant hit prioritization (stage 1)."""


@main.command("fetch-chembl")
@click.option("--config", default="configs/data_chembl.yaml", show_default=True)
@click.option("--no-cache", is_flag=True, help="Ignore raw CSV caches and refetch.")
@click.option(
    "--demo",
    is_flag=True,
    help="Use bundled demo fixtures instead of live ChEMBL (offline/repro).",
)
@click.option(
    "--sqlite",
    "sqlite_path",
    default=None,
    type=click.Path(exists=False),
    help="Optional local ChEMBL SQLite path (overrides config sqlite_path).",
)
def fetch_chembl(config: str, no_cache: bool, demo: bool, sqlite_path: str | None) -> None:
    """Fetch/clean ChEMBL proxy labels (or load demo fixtures)."""
    from radioprotect_sm.data.chembl import fetch_and_prepare, load_demo_labels

    if demo:
        paths = load_demo_labels()
        click.echo(f"Loaded demo labels → {paths}")
        return
    try:
        paths = fetch_and_prepare(
            config_path=config,
            use_cache=not no_cache,
            sqlite_path=sqlite_path,
        )
        click.echo(f"Prepared ChEMBL labels → {paths}")
    except Exception as exc:
        logger.error("Live ChEMBL fetch failed (%s). Falling back to --demo fixtures.", exc)
        paths = load_demo_labels()
        click.echo(f"Fallback demo labels → {paths}")


@main.command("prepare-splits")
@click.option("--train-config", default="configs/train.yaml", show_default=True)
def prepare_splits_cmd(train_config: str) -> None:
    """Write scaffold train/val/test assignments."""
    from radioprotect_sm.train import prepare_splits
    from radioprotect_sm.utils import load_yaml

    cfg = load_yaml(train_config)
    wide = project_root() / cfg["paths"]["processed_dir"] / "labels_wide.csv"
    df = prepare_splits(wide, cfg)
    out = project_root() / cfg["paths"]["processed_dir"] / "labels_wide_split.csv"
    df.to_csv(out, index=False)
    click.echo(f"Wrote split table → {out}")


@main.command("train")
@click.option("--train-config", default="configs/train.yaml", show_default=True)
@click.option("--data-config", default="configs/data_chembl.yaml", show_default=True)
@click.option(
    "--model",
    "model_backend",
    type=click.Choice(["baseline", "chemprop"]),
    default="baseline",
    show_default=True,
)
def train_cmd(train_config: str, data_config: str, model_backend: str) -> None:
    """Train ensemble and write held-out metrics."""
    from radioprotect_sm.train import train_ensemble

    report = train_ensemble(
        train_config=train_config,
        data_config=data_config,
        model_backend=model_backend,
    )
    click.echo(f"Training complete. Test metrics: {report['test_metrics']}")


@main.command("screen")
@click.option("--screen-config", default="configs/screen.yaml", show_default=True)
def screen_cmd(screen_config: str) -> None:
    """Rank purchasable library into Aim-1, Aim-2, and joint hit CSVs."""
    from radioprotect_sm.screen import run_screen

    hits = run_screen(screen_config=screen_config)
    click.echo(
        "Selected diverse hits → "
        f"dna_binding={len(hits['dna_binding'])}, "
        f"ros_scavenging={len(hits['ros_scavenging'])}, "
        f"dna_and_ros={len(hits['dna_and_ros'])} "
        "(outputs/hits_*.csv)"
    )


@main.command("run-all")
@click.option("--demo/--live-chembl", default=True, show_default=True, help="Demo fixtures (default) or live ChEMBL.")
def run_all(demo: bool) -> None:
    """End-to-end: prepare data → train → screen."""
    from radioprotect_sm.data.chembl import fetch_and_prepare, load_demo_labels
    from radioprotect_sm.screen import run_screen
    from radioprotect_sm.train import train_ensemble

    if demo:
        paths = load_demo_labels()
        click.echo(f"Loaded demo labels → {paths}")
    else:
        try:
            paths = fetch_and_prepare(use_cache=True)
            click.echo(f"Prepared ChEMBL labels → {paths}")
        except Exception as exc:
            logger.error("Live ChEMBL fetch failed (%s). Use --demo or fix network.", exc)
            sys.exit(1)

    report = train_ensemble(model_backend="baseline")
    click.echo(f"Training complete. Test metrics: {report['test_metrics']}")
    hits = run_screen()
    click.echo(
        "Selected diverse hits → "
        f"dna_binding={len(hits['dna_binding'])}, "
        f"ros_scavenging={len(hits['ros_scavenging'])}, "
        f"dna_and_ros={len(hits['dna_and_ros'])}"
    )
    click.echo("Pipeline finished.")


if __name__ == "__main__":
    main()
