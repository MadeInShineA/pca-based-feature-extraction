#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for select_optimal_pcs.
"""

import numpy as np
import pytest
from pcafeat.pca_feature_select import select_optimal_pcs


def test_select_optimal_pcs_target_only():
    """When no noise is provided, select by target score only."""
    target_stats = {
        'diagnosis': np.array([2.0, 1.0, 0.5]),
        'bdi': np.array([1.5, 0.5, 2.0]),
    }
    selected, info = select_optimal_pcs(target_stats, n_pcs=2)
    # sum target scores: [3.5, 1.5, 2.5]
    np.testing.assert_array_equal(selected, np.array([0, 2]))
    assert info['target_mode'] == 'sum'
    assert info['noise_mode'] == 'combined'
    np.testing.assert_array_equal(info['noise_score'], np.zeros(3))


def test_select_optimal_pcs_max_noise_penalizes_single_strong_confound():
    """
    A single strong confound should be penalized more harshly with
    noise_mode='max' than with noise_mode='sum'.
    """
    target_stats = {
        'diagnosis': np.array([3.0, 3.0, 3.0]),
    }
    # PC 0 has a single strong motion confound.
    # PC 1 is cleanest overall.
    # PC 2 is in between (same max as PC 1 in old data, so avoid tie here).
    noise_stats = {
        'motion': np.array([0.70, 0.20, 0.30]),
        'age': np.array([0.05, 0.20, 0.20]),
        'site': np.array([0.05, 0.20, 0.10]),
    }

    # With max noise: PC1 max=0.20, PC2 max=0.30, PC0 max=0.70 -> [1, 2, 0]
    selected_max, info_max = select_optimal_pcs(
        target_stats, noise_stats, n_pcs=3, noise_mode='max'
    )
    np.testing.assert_array_equal(selected_max, np.array([1, 2, 0]))

    # With sum noise: PC1 sum=0.60, PC2 sum=0.60, PC0 sum=0.80 -> tie between 1 and 2.
    selected_sum, info_sum = select_optimal_pcs(
        target_stats, noise_stats, n_pcs=3, noise_mode='sum'
    )
    # PC 0 must be last; PC 1 and PC 2 may tie depending on argsort stability.
    assert selected_sum[-1] == 0
    assert set(selected_sum[:2]) == {1, 2}


def test_select_optimal_pcs_combined_noise_balances_both():
    """Combined mode penalizes both single strong and total confound burden."""
    target_stats = {
        'diagnosis': np.array([3.0, 3.0, 3.0, 3.0]),
    }
    noise_stats = {
        'motion': np.array([0.70, 0.05, 0.30, 0.25]),
        'age': np.array([0.05, 0.05, 0.30, 0.25]),
        'site': np.array([0.05, 0.05, 0.30, 0.25]),
    }
    selected, info = select_optimal_pcs(
        target_stats,
        noise_stats,
        n_pcs=4,
        noise_mode='combined',
        noise_max_weight=0.7,
        noise_sum_weight=0.3,
    )
    # PC 0 has a single strong confound -> last.
    # PC 2 has many moderate confounds -> third.
    # PC 3 has moderate total burden -> second.
    # PC 1 is clean -> first.
    np.testing.assert_array_equal(selected, np.array([1, 3, 2, 0]))


def test_select_optimal_pcs_target_modes():
    """Test all supported target aggregation modes with deterministic data."""
    # PC 0 is strictly better than PC 1 across all aggregation modes.
    target_stats = {
        'a': np.array([2.5, 1.0]),
        'b': np.array([1.1, 2.0]),
    }

    # sum: [3.6, 3.0] -> [0, 1]
    selected_sum, _ = select_optimal_pcs(
        target_stats, n_pcs=2, target_mode='sum'
    )
    np.testing.assert_array_equal(selected_sum, np.array([0, 1]))

    # min: [1.1, 1.0] -> [0, 1]
    selected_min, _ = select_optimal_pcs(
        target_stats, n_pcs=2, target_mode='min'
    )
    np.testing.assert_array_equal(selected_min, np.array([0, 1]))

    # max: [2.5, 2.0] -> [0, 1]
    selected_max, _ = select_optimal_pcs(
        target_stats, n_pcs=2, target_mode='max'
    )
    np.testing.assert_array_equal(selected_max, np.array([0, 1]))

    # product: [2.75, 2.0] -> [0, 1]
    selected_prod, _ = select_optimal_pcs(
        target_stats, n_pcs=2, target_mode='product'
    )
    np.testing.assert_array_equal(selected_prod, np.array([0, 1]))


def test_select_optimal_pcs_min_mode_requires_all_targets():
    """min mode ranks highest the PC with the weakest target signal."""
    target_stats = {
        'diagnosis': np.array([3.0, 3.0, 1.0]),
        'bdi': np.array([3.0, 1.0, 3.0]),
    }
    selected, info = select_optimal_pcs(
        target_stats, n_pcs=3, target_mode='min'
    )
    # min: [3, 1, 1]; target_score is absolute, so PCs 1 and 2 tie at 1.0
    # PC 0 is best (3.0), followed by 1 and 2.
    np.testing.assert_array_equal(info['target_score'], np.array([3.0, 1.0, 1.0]))
    assert selected[0] == 0


def test_select_optimal_pcs_empty_target_raises():
    """Empty target_stats should raise ValueError."""
    with pytest.raises(ValueError, match="target_stats cannot be empty"):
        select_optimal_pcs({})


def test_select_optimal_pcs_unknown_target_mode_raises():
    """Unknown target_mode should raise ValueError."""
    target_stats = {'a': np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="Unknown target_mode"):
        select_optimal_pcs(target_stats, target_mode='unknown')


def test_select_optimal_pcs_unknown_noise_mode_raises():
    """Unknown noise_mode should raise ValueError."""
    target_stats = {'a': np.array([1.0, 2.0])}
    noise_stats = {'b': np.array([0.1, 0.2])}
    with pytest.raises(ValueError, match="Unknown noise_mode"):
        select_optimal_pcs(
            target_stats, noise_stats, noise_mode='unknown'
        )


def test_select_optimal_pcs_returns_info_dict():
    """The returned info dict should contain expected keys."""
    target_stats = {'a': np.array([1.0, 2.0])}
    noise_stats = {'b': np.array([0.1, 0.2])}
    selected, info = select_optimal_pcs(
        target_stats,
        noise_stats,
        n_pcs=1,
        target_mode='sum',
        noise_mode='combined',
        noise_max_weight=0.6,
        noise_sum_weight=0.4,
    )
    expected_keys = {
        'target_mode', 'noise_mode', 'noise_max_weight', 'noise_sum_weight',
        'candidate_pcs', 'final_score', 'target_score', 'noise_score',
        'target_stats', 'noise_stats',
    }
    assert expected_keys.issubset(info.keys())
    assert info['noise_max_weight'] == 0.6
    assert info['noise_sum_weight'] == 0.4
    assert info['candidate_pcs'] is None


def test_select_optimal_pcs_candidate_pcs_restricts_selection():
    """candidate_pcs should restrict selection to provided indices."""
    target_stats = {'a': np.array([3.0, 2.0, 1.0, 0.5])}
    noise_stats = {'b': np.array([0.1, 0.9, 0.1, 0.1])}

    # Without candidates, PC 0 wins (target 3.0, noise 0.1).
    selected_all, _ = select_optimal_pcs(target_stats, noise_stats, n_pcs=1)
    assert selected_all[0] == 0

    # With candidates [1, 2, 3], PC 0 is excluded.
    # PC 2 has target 1.0 and noise 0.1 -> score 10.0
    # PC 1 has target 2.0 and noise 0.9 -> score 2.22
    # PC 3 has target 0.5 and noise 0.1 -> score 5.0
    selected_cand, info = select_optimal_pcs(
        target_stats,
        noise_stats,
        n_pcs=1,
        candidate_pcs=[1, 2, 3]
    )
    assert selected_cand[0] == 2
    assert np.all(np.isin(selected_cand, [1, 2, 3]))


def test_select_optimal_pcs_candidate_pcs_noise_computed_globally():
    """Noise scores should still be computed over all PCs when candidates restrict selection."""
    target_stats = {'a': np.array([3.0, 2.0, 1.0])}
    noise_stats = {'b': np.array([0.1, 0.1, 0.1])}

    selected, info = select_optimal_pcs(
        target_stats,
        noise_stats,
        n_pcs=2,
        candidate_pcs=[1, 2]
    )

    # Noise score should be available for all PCs, not just candidates.
    assert len(info['noise_score']) == 3
    # Selected PCs must come from candidates.
    assert set(selected).issubset({1, 2})
