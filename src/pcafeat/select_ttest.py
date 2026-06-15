#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-test-based Feature Selection Pipeline

This module provides a function to identify connections (features) associated with
a binary target variable using independent t-test.

The pipeline:
1. Performs independent t-test between two groups (e.g., case vs. control)
2. Applies multiple comparison correction
3. Returns indices of significantly different features

Author: Ayumu Yamashita
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smt
import os


def select_ttest_features(df_X_train, target, control_group=None,
                          method='fdr_bh', save_results=False,
                          output_dir=None, output_prefix='ttest'):
    """
    Identify connections (features) associated with a binary target variable using t-test.

    This function performs independent t-test between two groups defined by the target
    variable and returns features that show significant differences after multiple
    comparison correction.

    Parameters
    ----------
    df_X_train : pandas.DataFrame
        Feature matrix with shape (n_samples, n_features).
        Each row represents a sample, each column represents a feature (e.g.,
        functional connectivity edge).
    target : pandas.Series or array-like
        Binary target variable with shape (n_samples,).
        Should contain exactly 2 unique values (e.g., 0/1, 'HC'/'MDD').
        NaN values should be excluded before calling this function.
    control_group : scalar or None, default=None
        Value in target that represents the control group.
        If None, the function will use the first unique value as control.
        For example, if target contains [0, 1], control_group=0 means group 0 is control.
    method : str, default='fdr_bh'
        Multiple comparison correction method.
        Options: 'fdr_bh' (False Discovery Rate, Benjamini-Hochberg),
                 'bonferroni', or other methods supported by statsmodels.
    save_results : bool, default=False
        Whether to save selected feature indices to CSV file.
    output_dir : str or None, default=None
        Directory path to save results. Required if save_results=True.
    output_prefix : str, default='ttest'
        Prefix for output filename.

    Returns
    -------
    selected_indices : numpy.ndarray
        Indices of selected features (connections) that show significant differences.
        Shape: (n_selected_features,)
    t_statistics : numpy.ndarray
        T-statistics for all features. Shape: (n_features,)
    p_values : numpy.ndarray
        P-values for all features. Shape: (n_features,)

    Raises
    ------
    TypeError
        If df_X_train is not a pandas.DataFrame.
    ValueError
        If df_X_train and target have different number of samples.
    ValueError
        If target does not contain exactly 2 unique values.
    ValueError
        If save_results=True but output_dir is not provided.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pcafeat.select_ttest import select_ttest_features
    >>>
    >>> # Prepare data
    >>> df_X_train = pd.DataFrame(np.random.randn(100, 1000))  # 100 samples, 1000 features
    >>> target = pd.Series([0]*50 + [1]*50)  # Binary target: 0=control, 1=case
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
    >>> selected_indices, t_stats, p_vals = select_ttest_features(
    ...     df_X_train_clean,
    ...     target_clean,
    ...     control_group=0,
    ...     method='fdr_bh',
    ...     save_results=True,
    ...     output_dir='./output/'
    ... )
    >>> print(f"Selected {len(selected_indices)} features")
    """

    # Validate inputs
    if not isinstance(df_X_train, pd.DataFrame):
        raise TypeError("df_X_train must be a pandas.DataFrame")

    if len(df_X_train) != len(target):
        raise ValueError("df_X_train and target must have the same number of samples")

    if save_results and output_dir is None:
        raise ValueError("output_dir must be provided when save_results=True")

    if save_results and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert target to Series if needed
    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    # Check that target has exactly 2 unique values
    unique_vals = target.unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Target must contain exactly 2 unique values, got {len(unique_vals)}: {unique_vals}")

    # Determine control and case groups
    if control_group is None:
        control_group = unique_vals[0]
        case_group = unique_vals[1]
    else:
        if control_group not in unique_vals:
            raise ValueError(f"control_group {control_group} not found in target values: {unique_vals}")
        case_group = unique_vals[unique_vals != control_group][0]

    # Create masks for groups
    control_mask = target == control_group
    case_mask = target == case_group

    # Check that both groups have at least 2 samples
    if control_mask.sum() < 2:
        raise ValueError(f"Control group has only {control_mask.sum()} samples. Need at least 2.")
    if case_mask.sum() < 2:
        raise ValueError(f"Case group has only {case_mask.sum()} samples. Need at least 2.")

    # Perform t-test
    t_statistics, p_values = ttest_ind(
        df_X_train.loc[case_mask],
        df_X_train.loc[control_mask],
        axis=0
    )

    # Apply multiple comparison correction
    significant, p_corrected, _, _ = smt.multipletests(p_values, method=method)
    selected_indices = np.where(significant)[0]

    # Save results if requested
    if save_results:
        # Create feature names
        feature_names = [f'con{i+1}' for i in selected_indices]
        df_results = pd.DataFrame(feature_names, columns=['selected_con'])

        # Save to CSV
        output_file = os.path.join(output_dir, f'{output_prefix}_{method}_selected_features.csv')
        df_results.to_csv(output_file, index=False)
        print(f"Saved selected features to: {output_file}")
        print(f"  Number of selected features: {len(selected_indices)}")
        print(f"  Control group: {control_group} (n={control_mask.sum()})")
        print(f"  Case group: {case_group} (n={case_mask.sum()})")

    return selected_indices, t_statistics, p_values


if __name__ == "__main__":
    """
    Example usage of select_ttest_features function.
    """
    import matplotlib.pyplot as plt

    # Example: Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 1000

    # Create feature matrix
    df_X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
    df_X_train.columns = [f'feature_{i}' for i in range(n_features)]

    # Create binary target variable (50 control, 50 case)
    # Add some signal: first 10 features differ between groups
    df_X_train.iloc[:50, :10] += 0.5  # Add signal to control group
    df_X_train.iloc[50:, :10] -= 0.5   # Add signal to case group

    target = pd.Series([0]*50 + [1]*50, name='diagnosis')

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

    # Extract features using t-test
    print("Running t-test-based feature selection...")
    selected_indices, t_stats, p_vals = select_ttest_features(
        df_X_train_clean,
        target_clean,
        control_group=0,
        method='fdr_bh',
        save_results=True,
        output_dir=output_dir,
        output_prefix='example_ttest'
    )

    print(f"\nResults:")
    print(f"  Number of selected features: {len(selected_indices)}")
    print(f"  Selected feature indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"  Selected feature indices: {selected_indices}")
