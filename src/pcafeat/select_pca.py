#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA-based Feature Selection Pipeline

This module provides a function to identify connections (features) associated with
a target variable using Principal Component Analysis (PCA).

The pipeline:
1. Performs PCA on the input feature matrix (df_X_train)
2. Identifies principal components significantly associated with the target variable
3. Extracts connections (features) that contribute to the selected components

Author: Ayumu Yamashita
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pcafeat.pca_feature_select import pca_extract, con_extract
import os


def select_pca_features(df_X_train, target, method_pick_pca='fdr_bh',
                        method_pick_con='fdr_bh', fig_plot=False,
                        fig_dir=None, bar_color='blue'):
    """
    Identify connections (features) associated with a target variable using PCA.

    This function performs PCA on the input feature matrix, identifies principal
    components (PCs) significantly associated with the target variable, and extracts
    the connections (features) that contribute to those PCs.

    Parameters
    ----------
    df_X_train : pandas.DataFrame
        Feature matrix with shape (n_samples, n_features).
        Each row represents a sample, each column represents a feature (e.g.,
        functional connectivity edge).
    target : pandas.Series or array-like
        Target variable with shape (n_samples,).
        Can be binary (0/1), continuous, or categorical.
        NaN values should be excluded before calling this function.
    method_pick_pca : str, default='fdr_bh'
        Multiple comparison correction method for selecting significant PCs.
        Options: 'fdr_bh' (False Discovery Rate, Benjamini-Hochberg),
                 'bonferroni', or other methods supported by statsmodels.
    method_pick_con : str, default='fdr_bh'
        Multiple comparison correction method for selecting significant connections.
        Options: 'fdr_bh', 'bonferroni', or 'optimal_sigma'.
    fig_plot : bool, default=False
        Whether to generate and save plots of statistics.
    fig_dir : str or None, default=None
        Directory path to save plots. Required if fig_plot=True.
    bar_color : str, default='blue'
        Color for the bar plot of statistics.

    Returns
    -------
    cons : numpy.ndarray
        Indices of selected connections (features) associated with the target.
        Shape: (n_selected_connections,)
    cons_pc : numpy.ndarray
        Principal component indices corresponding to each selected connection.
        Shape: (n_selected_connections,)
        Each element indicates which PC the connection contributes to.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pcafeat.select_pca import select_pca_features
    >>>
    >>> # Prepare data
    >>> df_X_train = pd.DataFrame(np.random.randn(100, 1000))  # 100 samples, 1000 features
    >>> target = pd.Series(np.random.randint(0, 2, 100))  # Binary target
    >>>
    >>> # Remove NaN values
    >>> if pd.api.types.is_numeric_dtype(target):
    ...     use_sub = ~np.isnan(target)
    ... else:
    ...     use_sub = ~target.isna()
    >>> df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
    >>> target_clean = target[use_sub].reset_index(drop=True)
    >>>
    >>> # Extract features
    >>> cons, cons_pc = select_pca_features(
    ...     df_X_train_clean,
    ...     target_clean,
    ...     method_pick_pca='fdr_bh',
    ...     method_pick_con='fdr_bh',
    ...     fig_plot=True,
    ...     fig_dir='./output/'
    ... )
    >>> print(f"Selected {len(cons)} connections")
    >>> print(f"Associated with PCs: {np.unique(cons_pc)}")
    """

    # Validate inputs
    if not isinstance(df_X_train, pd.DataFrame):
        raise TypeError("df_X_train must be a pandas.DataFrame")

    if len(df_X_train) != len(target):
        raise ValueError("df_X_train and target must have the same number of samples")

    if fig_plot and fig_dir is None:
        raise ValueError("fig_dir must be provided when fig_plot=True")

    if fig_plot and not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Convert target to Series if needed
    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    # Perform PCA on the feature matrix
    pca = PCA().fit(df_X_train)
    coeff_train = pca.components_.T  # PCA coefficients (loadings): shape (n_features, n_components)
    score = pca.transform(df_X_train)  # PCA scores: shape (n_samples, n_components)

    # Create DataFrame with PCA scores
    n_components = score.shape[1]
    pca_id = [f'pca_score{i}' for i in range(1, n_components + 1)]
    df_score = pd.DataFrame(score, columns=pca_id)

    # Identify principal components significantly associated with the target variable
    # This automatically determines the appropriate statistical test:
    # - Binary (2 values): t-test
    # - Continuous (>2 unique values): Pearson correlation
    # - Categorical: ANOVA
    select_pc, statistics = pca_extract(
        df_score,
        target,
        method_pick_pca,
        fig_plot=fig_plot,
        fig_dir=fig_dir,
        bar_color=bar_color
    )

    # Extract connections (features) associated with the selected components
    cons, cons_pc = con_extract(coeff_train, select_pc, method_pick_con)

    return cons, cons_pc, df_score


if __name__ == "__main__":
    """
    Example usage of select_pca_features function.
    """
    import matplotlib.pyplot as plt

    # Example: Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 1000

    # Create feature matrix
    df_X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
    df_X_train.columns = [f'feature_{i}' for i in range(n_features)]

    # Create binary target variable
    target = pd.Series(np.random.randint(0, 2, n_samples), name='diagnosis')

    # Remove NaN values (if any)
    if pd.api.types.is_numeric_dtype(target):
        use_sub = ~np.isnan(target)
    else:
        use_sub = ~target.isna()

    df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
    target_clean = target[use_sub].reset_index(drop=True)

    # Create output directory
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)

    # Extract features associated with target
    print("Running PCA-based feature selection...")
    cons, cons_pc = select_pca_features(
        df_X_train_clean,
        target_clean,
        method_pick_pca='fdr_bh',
        method_pick_con='fdr_bh',
        fig_plot=True,
        fig_dir=output_dir,
        bar_color='blue'
    )

    print(f"\nResults:")
    print(f"  Number of selected connections: {len(cons)}")
    print(f"  Associated with {len(np.unique(cons_pc))} principal components")
    print(f"  PC indices: {sorted(np.unique(cons_pc))}")
