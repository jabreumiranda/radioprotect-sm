"""End-to-end smoke test on demo fixtures."""

from __future__ import annotations

from pathlib import Path

from radioprotect_sm.data.chembl import load_demo_labels
from radioprotect_sm.screen import run_screen
from radioprotect_sm.train import train_ensemble


def test_demo_pipeline_train_and_screen(tmp_path, monkeypatch):
    # Keep outputs inside repo paths expected by configs; use real project dirs.
    paths = load_demo_labels()
    assert paths["wide"].exists()
    report = train_ensemble(
        train_config="configs/train.yaml",
        data_config="configs/data_chembl.yaml",
        model_backend="baseline",
    )
    assert "test_metrics" in report
    assert Path("runs/ensemble_bundle.json").exists()
    hits = run_screen(screen_config="configs/screen.yaml")
    assert set(hits) == {"dna_binding", "ros_scavenging", "dna_and_ros"}
    assert all(len(hits[k]) >= 1 for k in hits)
    assert Path("outputs/hits_dna_binding.csv").exists()
    assert Path("outputs/hits_ros_scavenging.csv").exists()
    assert Path("outputs/hits_dna_and_ros.csv").exists()
    assert Path("outputs/hits_for_purchase.csv").exists()
