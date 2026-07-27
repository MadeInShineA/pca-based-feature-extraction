#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for select_perturbated_pca_features (Docker-based fuzzy PCA).
"""

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from pcafeat.select_pca import (
    select_pca_features,
    select_perturbated_pca_features,
)

DOCKER_IMAGE = "verificarlo/fuzzy:v2.0.0-lapack-python3.8.5-numpy-scipy-sklearn"


def _docker_available():
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker is not available or not working",
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
    result = select_perturbated_pca_features(DOCKER_IMAGE, df, target)
    assert len(result) == 2
    cons, cons_pc = result
    assert isinstance(cons, np.ndarray)
    assert isinstance(cons_pc, np.ndarray)
    assert len(cons) == len(cons_pc)


def test_return_statistics(sample_data):
    df, target = sample_data
    result = select_perturbated_pca_features(
        DOCKER_IMAGE, df, target, return_statistics=True
    )
    assert len(result) == 3
    cons, cons_pc, statistics = result
    assert isinstance(statistics, np.ndarray)
    assert statistics.shape[0] == df.shape[1]


def test_results_are_perturbed(sample_data):
    df, target = sample_data

    local_cons, local_cons_pc, local_stats = select_pca_features(
        df, target, return_statistics=True
    )
    docker_cons, docker_cons_pc, docker_stats = select_perturbated_pca_features(
        DOCKER_IMAGE, df, target, return_statistics=True
    )

    # Verify perturbation happened - results should differ
    # Docker results may have NaN or different values due to floating-point perturbation
    assert not np.array_equal(local_stats, docker_stats, equal_nan=True)


def test_docker_cleanup(sample_data):
    df, target = sample_data

    before = subprocess.run(
        ["docker", "ps", "-a", "--filter", "status=exited", "--format", "{{.ID}}"],
        capture_output=True, text=True,
    )
    exited_before = set(before.stdout.strip().splitlines())

    select_perturbated_pca_features(DOCKER_IMAGE, df, target)

    after = subprocess.run(
        ["docker", "ps", "-a", "--filter", "status=exited", "--format", "{{.ID}}"],
        capture_output=True, text=True,
    )
    exited_after = set(after.stdout.strip().splitlines())

    assert exited_after == exited_before


def test_invalid_input_shape(sample_data):
    df, target = sample_data
    with pytest.raises(ValueError, match="same number of samples"):
        select_perturbated_pca_features(DOCKER_IMAGE, df.iloc[:-1], target)


def test_invalid_df_type_raises(sample_data):
    _, target = sample_data
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        select_perturbated_pca_features(DOCKER_IMAGE, [[1, 2], [3, 4]], target)
