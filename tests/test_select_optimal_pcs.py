#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for select_optimal_pcs.
"""

import numpy as np
import pytest
from pcafeat.pca_feature_select import select_optimal_pcs


def test_select_optimal_pcs_basic():
    """Select PCs with high target scores and low noise scores."""
    target_pcs_lists = [
        [(0, 1.0), (1, 2.0), (2, 1.5)],
        [(1, 1.5), (2, 2.0), (3, 1.0)],
    ]
    noise_pcs_lists = [
        [(3, 2.0), (4, 1.0)],
        [(4, 2.0), (5, 1.0)],
    ]

    # target_score: PC0=1, PC1=3.5, PC2=3.5, PC3=1
    # noise_score:  PC3=2, PC4=3, PC5=1
    selected, info = select_optimal_pcs(
        target_pcs_lists, noise_pcs_lists, n_pcs=3
    )

    np.testing.assert_array_equal(
        info['target_score'], np.array([1.0, 3.5, 3.5, 1.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(
        info['noise_score'], np.array([0.0, 0.0, 0.0, 2.0, 3.0, 1.0])
    )
    # Best are PC1 and PC2 (tie), then PC0
    assert selected[0] in {1, 2}
    assert selected[1] in {1, 2}
    assert selected[2] == 0


def test_select_optimal_pcs_prefers_shared_targets():
    """PCs appearing in multiple target lists with high scores rank higher."""
    target_pcs_lists = [
        [(0, 1.0), (1, 2.0)],
        [(1, 2.0), (2, 1.0)],
        [(1, 2.0), (3, 1.0)],
    ]
    noise_pcs_lists = [[(4, 1.0)]]

    selected, info = select_optimal_pcs(
        target_pcs_lists, noise_pcs_lists, n_pcs=1
    )

    # PC1: target = 6.0, noise = 0 -> best
    assert selected[0] == 1
    np.testing.assert_array_equal(
        info['target_score'], np.array([1.0, 6.0, 1.0, 1.0, 0.0])
    )


def test_select_optimal_pcs_penalizes_noise_scores():
    """PCs with high noise scores are penalized."""
    target_pcs_lists = [
        [(0, 1.0), (1, 2.0)],
        [(1, 2.0), (2, 1.0)],
    ]
    noise_pcs_lists = [
        [(0, 3.0)],
        [(0, 3.0)],
    ]

    selected, info = select_optimal_pcs(
        target_pcs_lists, noise_pcs_lists, n_pcs=1
    )

    # PC0: target=1, noise=6 -> score 0.167
    # PC1: target=4, noise=0 -> score huge
    # PC2: target=1, noise=0 -> score huge
    assert selected[0] == 1


def test_select_optimal_pcs_returns_info_dict():
    """The returned info dict should contain expected keys."""
    target_pcs_lists = [[(0, 1.0), (1, 2.0)]]
    noise_pcs_lists = [[(2, 1.0), (3, 2.0)]]

    selected, info = select_optimal_pcs(
        target_pcs_lists, noise_pcs_lists, n_pcs=1
    )

    expected_keys = {
        'final_score', 'target_score', 'noise_score',
        'target_pcs_lists', 'noise_pcs_lists',
    }
    assert expected_keys.issubset(info.keys())
    assert info['target_pcs_lists'] == target_pcs_lists
    assert info['noise_pcs_lists'] == noise_pcs_lists


def test_select_optimal_pcs_empty_target_lists_raises():
    """Empty target_pcs_lists should raise ValueError."""
    with pytest.raises(ValueError, match="target_pcs_lists cannot be empty"):
        select_optimal_pcs([], [[(0, 1.0)]])


def test_select_optimal_pcs_empty_noise_lists_raises():
    """Empty noise_pcs_lists should raise ValueError."""
    with pytest.raises(ValueError, match="noise_pcs_lists cannot be empty"):
        select_optimal_pcs([[(0, 1.0)]], [])


def test_select_optimal_pcs_empty_all_lists():
    """Empty input lists return empty selection."""
    selected, info = select_optimal_pcs([[]], [[]], n_pcs=1)
    assert len(selected) == 0


def test_select_optimal_pcs_negative_scores_use_absolute():
    """Negative scores are treated as absolute values."""
    target_pcs_lists = [[(0, -3.0), (1, -1.0)]]
    noise_pcs_lists = [[(1, -2.0)]]

    selected, info = select_optimal_pcs(
        target_pcs_lists, noise_pcs_lists, n_pcs=1
    )

    # PC0: target=3, noise=0 -> best
    # PC1: target=1, noise=2 -> worse
    assert selected[0] == 0
    np.testing.assert_array_equal(
        info['target_score'], np.array([3.0, 1.0])
    )
    np.testing.assert_array_equal(
        info['noise_score'], np.array([0.0, 2.0])
    )
