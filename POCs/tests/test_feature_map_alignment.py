"""Tests for feature-map alignment cross table (ZZ vs Pauli-Y)."""

from __future__ import annotations

import numpy as np
import pytest

from qml_v2.feature_map_alignment import (
    AlignConfig,
    build_feature_map,
    make_map_native,
    run_alignment,
    summarize,
)


def test_build_maps_have_two_params():
    for kind in ("zz", "pauli_y"):
        fmap = build_feature_map(kind, n_qubits=2, reps=1, entanglement="full")
        assert fmap.num_qubits == 2
        assert len(fmap.parameters) == 2


def test_generators_produce_both_classes():
    for kind in ("zz", "pauli_y"):
        X, y = make_map_native(kind, n_samples=60, reps=1, random_state=0)
        assert X.shape == (60, 2)
        assert set(np.unique(y)) == {0, 1}


def test_quick_alignment_runs():
    cfg = AlignConfig(n_total=40, n_repeats=1, reps=1, random_state=1)
    df = run_alignment(cfg)
    assert set(df["generator"]) == {"zz", "pauli_y"}
    assert set(df["model"]) == {"QSVM-ZZ", "QSVM-PauliY", "SVM-RBF"}
    # One repeat × 2 generators × 3 models
    assert len(df) == 6
    summary = summarize(df)
    assert len(summary) == 6
