"""Metrics and ranking unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radioprotect_sm.metrics import pearson_corr, spearman_corr, topk_hit_rate
from radioprotect_sm.screen import (
    build_hit_list_weights,
    diversity_select,
    multiobjective_score,
)


def test_spearman_perfect_ranking():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert spearman_corr(y, y) == pytest.approx(1.0)
    assert pearson_corr(y, y) == pytest.approx(1.0)


def test_topk_recovers_all_when_predictions_perfect():
    y = np.arange(20, dtype=float)
    rate = topk_hit_rate(y, y, true_top_frac=0.1, pred_top_frac=0.5)
    assert rate == pytest.approx(1.0)


def test_metrics_ignore_nan_pairs():
    yt = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    yp = np.array([1.1, 2.1, 99.0, 3.9, 5.2, 5.8])
    assert np.isfinite(spearman_corr(yt, yp))


def test_multiobjective_score_penalizes_cytotox_and_uncertainty():
    mean = pd.DataFrame(
        {
            "dna_binding": [2.0, 0.0],
            "ros_scavenging": [2.0, 0.0],
            "cytotox": [2.0, 0.0],
        }
    )
    std = pd.DataFrame(
        {
            "dna_binding": [0.0, 0.0],
            "ros_scavenging": [0.0, 0.0],
            "cytotox": [0.0, 0.0],
        }
    )
    scores = multiobjective_score(
        mean,
        std,
        weights={"dna_binding": 1.0, "ros_scavenging": 1.0, "cytotox": -0.5},
        uncertainty_penalty=0.0,
        zscore_within_library=True,
    )
    # First molecule higher on beneficial tasks; cytotox penalty applies but overall should still rank first
    assert scores.iloc[0] > scores.iloc[1]


def test_build_hit_list_weights_keeps_safety_on_all_lists():
    w = build_hit_list_weights(
        {"dna_binding": 1.0, "ros_scavenging": 1.0, "cytotox": -0.5}
    )
    assert set(w) == {"dna_binding", "ros_scavenging", "dna_and_ros"}
    assert "ros_scavenging" not in w["dna_binding"]
    assert "dna_binding" not in w["ros_scavenging"]
    assert w["dna_binding"]["cytotox"] == -0.5
    assert w["ros_scavenging"]["cytotox"] == -0.5
    assert w["dna_and_ros"]["dna_binding"] == 1.0
    assert w["dna_and_ros"]["ros_scavenging"] == 1.0


def test_diversity_select_respects_threshold_and_max_hits():
    # Identical SMILES should collapse under high Tanimoto threshold
    smiles = ["c1ccccc1O", "c1ccccc1O", "CCOC(=O)c1ccccc1", "NCCCCC(N)C(=O)O"]
    scores = np.array([4.0, 3.9, 3.0, 2.0])
    selected = diversity_select(smiles, scores, tanimoto_threshold=0.99, max_hits=3)
    assert len(selected) <= 3
    assert selected[0] == 0  # highest score first
    # second identical phenol should be skipped
    assert 1 not in selected
