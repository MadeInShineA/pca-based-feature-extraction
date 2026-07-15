#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for select_pca_features and pca_extract with new parameters.
"""

import numpy as np
import pandas as pd
import pytest
from pcafeat.select_pca import select_pca_features


@pytest.fixture
def sample_data():
    """Generate synthetic data where a few PCs are strongly linked to target."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    # First feature is strongly correlated with target.
    target = pd.Series(np.random.randint(0, 2, n_samples), name='target')
    df = pd.DataFrame(np.random.randn(n_samples, n_features))
    df.iloc[:, 0] = df.iloc[:, 0] + 2.0 * target
    return df, target


def test_select_pca_features_default_returns_two_arrays(sample_data):
    """Default call should return two arrays: cons and cons_pc."""
    df, target = sample_data
    result = select_pca_features(df, target)
    assert len(result) == 2
    cons, cons_pc = result
    assert isinstance(cons, np.ndarray)
    assert isinstance(cons_pc, np.ndarray)
    assert len(cons) == len(cons_pc)


def test_select_pca_features_return_statistics(sample_data):
    """return_statistics=True should return a third array of statistics."""
    df, target = sample_data
    result = select_pca_features(df, target, return_statistics=True)
    assert len(result) == 3
    cons, cons_pc, statistics = result
    assert isinstance(statistics, np.ndarray)
    assert statistics.shape[0] == df.shape[1]


def test_select_pca_features_n_pcs_selects_top_n(sample_data):
    """n_pcs should select the top N PCs by absolute statistic."""
    df, target = sample_data
    cons, cons_pc, statistics = select_pca_features(
        df, target, n_pcs=3, return_statistics=True
    )
    # Verify top 3 PCs by absolute statistic.
    top_pcs = np.argsort(np.abs(statistics))[::-1][:3]
    # cons_pc maps extracted connections to their source PC.
    # All returned connections must originate from one of the selected top PCs.
    selected_pcs = np.unique(cons_pc).astype(int)
    assert set(selected_pcs).issubset(set(top_pcs))
    # With the seeded data, at least one connection should be found.
    assert len(cons) > 0


def test_select_pca_features_alpha_parameter(sample_data):
    """alpha should control significance threshold for p-value selection."""
    df, target = sample_data
    # Strict alpha should select fewer or equal PCs than lenient alpha.
    cons_strict, _ = select_pca_features(df, target, alpha=0.001)
    cons_lenient, _ = select_pca_features(df, target, alpha=0.5)
    assert len(cons_strict) <= len(cons_lenient)


def test_select_pca_features_invalid_input_shape(sample_data):
    """Mismatched lengths between df_X_train and target should raise ValueError."""
    df, target = sample_data
    with pytest.raises(ValueError, match="same number of samples"):
        select_pca_features(df.iloc[:-1], target)


def test_select_pca_features_fig_dir_required_when_plotting(sample_data):
    """fig_plot=True without fig_dir should raise ValueError."""
    df, target = sample_data
    with pytest.raises(ValueError, match="fig_dir must be provided"):
        select_pca_features(df, target, fig_plot=True)


def test_select_pca_features_n_pcs_zero(sample_data):
    """n_pcs=0 should select no PCs and therefore no connections."""
    df, target = sample_data
    cons, cons_pc = select_pca_features(df, target, n_pcs=0)
    assert len(cons) == 0
    assert len(cons_pc) == 0
