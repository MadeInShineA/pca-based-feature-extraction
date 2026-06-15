import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smt
import os


def select_ttest_features(df_X_train, target, control_group=None,
                          method='fdr_bh', save_results=False,
                          output_dir=None, output_prefix='ttest'):
    if not isinstance(df_X_train, pd.DataFrame):
        raise TypeError("df_X_train must be a pandas.DataFrame")

    if len(df_X_train) != len(target):
        raise ValueError("df_X_train and target must have the same number of samples")

    if save_results and output_dir is None:
        raise ValueError("output_dir must be provided when save_results=True")

    if save_results and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    unique_vals = target.unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Target must contain exactly 2 unique values, got {len(unique_vals)}: {unique_vals}")

    if control_group is None:
        control_group = unique_vals[0]
        case_group = unique_vals[1]
    else:
        if control_group not in unique_vals:
            raise ValueError(f"control_group {control_group} not found in target values: {unique_vals}")
        case_group = unique_vals[unique_vals != control_group][0]

    control_mask = target == control_group
    case_mask = target == case_group

    if control_mask.sum() < 2:
        raise ValueError(f"Control group has only {control_mask.sum()} samples. Need at least 2.")
    if case_mask.sum() < 2:
        raise ValueError(f"Case group has only {case_mask.sum()} samples. Need at least 2.")

    t_statistics, p_values = ttest_ind(
        df_X_train.loc[case_mask],
        df_X_train.loc[control_mask],
        axis=0
    )

    significant, p_corrected, _, _ = smt.multipletests(p_values, method=method)
    selected_indices = np.where(significant)[0]

    if save_results:
        feature_names = [f'con{i+1}' for i in selected_indices]
        df_results = pd.DataFrame(feature_names, columns=['selected_con'])

        output_file = os.path.join(output_dir, f'{output_prefix}_{method}_selected_features.csv')
        df_results.to_csv(output_file, index=False)
        print(f"Saved selected features to: {output_file}")
        print(f"  Number of selected features: {len(selected_indices)}")
        print(f"  Control group: {control_group} (n={control_mask.sum()})")
        print(f"  Case group: {case_group} (n={case_mask.sum()})")

    return selected_indices, t_statistics, p_values
