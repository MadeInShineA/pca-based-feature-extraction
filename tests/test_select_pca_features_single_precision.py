#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for select_pca_features_single_precision (float32 precision PCA).
"""

import numpy as np
import pandas as pd
import pytest
from pcafeat.select_pca import (
    select_pca_features,
    select_pca_features_single_precision,
)


@pytest.fixture
def sample_data():
    """Generate synthetic data where a few PCs are strongly linked to target."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    target = pd.Series(np.random.randint(0, 2, n_samples), name="target")
    df = pd.DataFrame(np.random.randn(n_samples, n_features))
    df.iloc[:, 0] = df.iloc[:, 0] + 2.0 * target
    return df, target


def test_returns_two_arrays(sample_data):
    df, target = sample_data
    result = select_pca_features_single_precision(df, target)
    assert len(result) == 2
    cons, cons_pc = result
    assert isinstance(cons, np.ndarray)
    assert isinstance(cons_pc, np.ndarray)
    assert len(cons) == len(cons_pc)


def test_return_statistics(sample_data):
    df, target = sample_data
    result = select_pca_features_single_precision(
        df, target, return_statistics=True
    )
    assert len(result) == 3
    cons, cons_pc, statistics = result
    assert isinstance(statistics, np.ndarray)


def test_results_differ_from_float64(sample_data):
    df, target = sample_data

    float64_cons, float64_cons_pc, float64_stats = select_pca_features(
        df, target, return_statistics=True
    )
    float32_cons, float32_cons_pc, float32_stats = select_pca_features_single_precision(
        df, target, return_statistics=True
    )

    # Results should differ due to reduced precision
    assert not np.allclose(float64_stats, float32_stats, equal_nan=True)


def test_invalid_input_shape(sample_data):
    df, target = sample_data
    with pytest.raises(ValueError, match="same number of samples"):
        select_pca_features_single_precision(df.iloc[:-1], target)


def test_invalid_df_type_raises(sample_data):
    _, target = sample_data
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        select_pca_features_single_precision([[1, 2], [3, 4]], target)
