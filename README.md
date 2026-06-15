# Feature Selection Pipeline for Neuroimaging Data

A Python package for identifying features (connections) associated with a target variable using two complementary approaches:
1. **PCA-based method**: Uses Principal Component Analysis for dimensionality reduction
2. **T-test-based method**: Uses direct independent t-test for binary group comparisons

This pipeline is particularly useful for high-dimensional neuroimaging data, such as functional connectivity matrices.

## Overview

### PCA-based Method

This method performs the following steps:

1. **Principal Component Analysis (PCA)**: Reduces dimensionality of the feature matrix
2. **Component Selection**: Identifies principal components significantly associated with the target variable
3. **Feature Extraction**: Extracts original features (connections) that contribute to the selected components

The pipeline automatically selects the appropriate statistical test based on the target variable type:
- **Binary variables** (2 values): Independent t-test
- **Continuous variables** (>2 unique values): Pearson correlation
- **Categorical variables**: One-way ANOVA

### T-test-based Method

This method performs direct feature-wise comparison:

1. **Independent T-test**: Performs t-test between two groups for each feature
2. **Multiple Comparison Correction**: Applies correction (FDR-BH or Bonferroni)
3. **Feature Selection**: Returns indices of significantly different features

**Use when**: You have a binary target variable (e.g., case vs. control) and want direct feature-level comparisons without dimensionality reduction.

## Installation

### Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- scipy
- statsmodels
- matplotlib
- joblib
- tqdm

### Install from local source

```bash
pip install .
```

### Install directly from GitHub

```bash
pip install git+https://github.com/<USER>/<REPO>.git
```

### Install in development mode

```bash
pip install -e .
```

### Run examples

```bash
python examples/run_pca.py
python examples/run_ttest.py
```

## Quick Start

### PCA-based Method

```python
import pandas as pd
import numpy as np
from pcafeat import select_pca_features

# Prepare your data
# df_X_train: Feature matrix (n_samples × n_features)
# target: Target variable (n_samples,)

# Remove NaN values before calling the function
if pd.api.types.is_numeric_dtype(target):
    use_sub = ~np.isnan(target)
else:
    use_sub = ~target.isna()

df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
target_clean = target[use_sub].reset_index(drop=True)

# Extract features associated with target
cons, cons_pc = select_pca_features(
    df_X_train_clean,
    target_clean,
    method_pick_pca='fdr_bh',
    method_pick_con='fdr_bh',
    fig_plot=True,
    fig_dir='./output/'
)

print(f"Selected {len(cons)} connections")
print(f"Associated with PCs: {np.unique(cons_pc)}")
```

### T-test-based Method

```python
import pandas as pd
import numpy as np
from pcafeat import select_ttest_features

# Prepare your data
# df_X_train: Feature matrix (n_samples × n_features)
# target: Binary target variable (n_samples,) with exactly 2 values

# Remove NaN values before calling the function
if pd.api.types.is_numeric_dtype(target):
    use_sub = ~np.isnan(target)
else:
    use_sub = ~target.isna()

df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
target_clean = target[use_sub].reset_index(drop=True)

# Extract features using t-test
selected_indices, t_stats, p_vals = select_ttest_features(
    df_X_train_clean,
    target_clean,
    control_group=0,  # 0 = control, 1 = case
    method='fdr_bh',
    save_results=True,
    output_dir='./output/'
)

print(f"Selected {len(selected_indices)} features")
```

### Example with Different Target Types

#### Binary Target (e.g., Diagnosis: HC vs. MDD)

```python
# Binary target: 0 = HC, 1 = MDD
target = pd.Series([0, 0, 1, 1, 0, 1, ...], name='diagnosis')

cons, cons_pc = select_pca_features(
    df_X_train_clean,
    target_clean,
    method_pick_pca='fdr_bh',
    bar_color='orange'
)
```

#### Continuous Target (e.g., Age)

```python
# Continuous target: age in years
target = pd.Series([25.3, 30.1, 28.5, ...], name='age')

cons, cons_pc = select_pca_features(
    df_X_train_clean,
    target_clean,
    method_pick_pca='fdr_bh',
    bar_color='blue'
)
```

#### Categorical Target (e.g., Site)

```python
# Categorical target: site labels
target = pd.Series(['site1', 'site2', 'site1', 'site3', ...], name='site')

cons, cons_pc = select_pca_features(
    df_X_train_clean,
    target_clean,
    method_pick_pca='fdr_bh',
    bar_color='red'
)
```

## API Reference

### `select_pca_features`

Main function for PCA-based feature selection.

#### Parameters

- **df_X_train** (pandas.DataFrame): Feature matrix with shape `(n_samples, n_features)`. Each row represents a sample, each column represents a feature.

- **target** (pandas.Series or array-like): Target variable with shape `(n_samples,)`. Can be binary, continuous, or categorical. **NaN values must be excluded before calling this function.**

- **method_pick_pca** (str, default='fdr_bh'): Multiple comparison correction method for selecting significant principal components. Options:
  - `'fdr_bh'`: False Discovery Rate (Benjamini-Hochberg) - recommended
  - `'bonferroni'`: Bonferroni correction
  - Other methods supported by `statsmodels.stats.multitest`

- **method_pick_con** (str, default='fdr_bh'): Multiple comparison correction method for selecting significant connections. Options:
  - `'fdr_bh'`: False Discovery Rate (Benjamini-Hochberg) - recommended
  - `'bonferroni'`: Bonferroni correction
  - `'optimal_sigma'`: Optimal sigma method (computationally intensive)

- **fig_plot** (bool, default=False): Whether to generate and save plots of statistics.

- **fig_dir** (str or None, default=None): Directory path to save plots. Required if `fig_plot=True`.

- **bar_color** (str, default='blue'): Color for the bar plot of statistics.

#### Returns

- **cons** (numpy.ndarray): Indices of selected connections (features) associated with the target. Shape: `(n_selected_connections,)`

- **cons_pc** (numpy.ndarray): Principal component indices corresponding to each selected connection. Shape: `(n_selected_connections,)`. Each element indicates which PC the connection contributes to.

#### Raises

- **TypeError**: If `df_X_train` is not a pandas.DataFrame
- **ValueError**: If `df_X_train` and `target` have different number of samples
- **ValueError**: If `fig_plot=True` but `fig_dir` is not provided

### `select_ttest_features`

Main function for t-test-based feature selection.

#### Parameters

- **df_X_train** (pandas.DataFrame): Feature matrix with shape `(n_samples, n_features)`. Each row represents a sample, each column represents a feature.

- **target** (pandas.Series or array-like): Binary target variable with shape `(n_samples,)`. Must contain exactly 2 unique values (e.g., 0/1, 'HC'/'MDD'). **NaN values must be excluded before calling this function.**

- **control_group** (scalar or None, default=None): Value in target that represents the control group. If None, the first unique value is used as control.

- **method** (str, default='fdr_bh'): Multiple comparison correction method. Options:
  - `'fdr_bh'`: False Discovery Rate (Benjamini-Hochberg) - recommended
  - `'bonferroni'`: Bonferroni correction
  - Other methods supported by `statsmodels.stats.multitest`

- **save_results** (bool, default=False): Whether to save selected feature indices to CSV file.

- **output_dir** (str or None, default=None): Directory path to save results. Required if `save_results=True`.

- **output_prefix** (str, default='ttest'): Prefix for output filename.

#### Returns

- **selected_indices** (numpy.ndarray): Indices of selected features that show significant differences. Shape: `(n_selected_features,)`

- **t_statistics** (numpy.ndarray): T-statistics for all features. Shape: `(n_features,)`

- **p_values** (numpy.ndarray): P-values for all features. Shape: `(n_features,)`

#### Raises

- **TypeError**: If `df_X_train` is not a pandas.DataFrame
- **ValueError**: If `df_X_train` and `target` have different number of samples
- **ValueError**: If target does not contain exactly 2 unique values
- **ValueError**: If `save_results=True` but `output_dir` is not provided

## Statistical Methods

### Principal Component Selection

The pipeline automatically selects the appropriate statistical test:

1. **Binary variables** (2 unique values):
   - Uses independent t-test (`scipy.stats.ttest_ind`)
   - Compares PCA scores between two groups

2. **Continuous variables** (>2 unique values):
   - Uses Pearson correlation (`scipy.stats.pearsonr`)
   - Tests correlation between PCA scores and target variable

3. **Categorical variables** (non-numeric or >2 categories):
   - Uses one-way ANOVA (`scipy.stats.f_oneway`)
   - Tests differences in PCA scores across categories

### Multiple Comparison Correction

After computing statistics for all principal components, multiple comparison correction is applied using the specified method (default: FDR-BH).

### Connection Selection

For each selected principal component, connections (features) are selected based on their contribution to the component (using chi-square test with multiple comparison correction).

### T-test Method

1. **Independent T-test**: For each feature, performs independent t-test between two groups
2. **Multiple Comparison Correction**: Applies correction to control for false positives across all features
3. **Feature Selection**: Returns indices of features with significant differences after correction

## File Structure

```
.
├── pyproject.toml          # Package configuration
├── src/
│   └── pcafeat/
│       ├── __init__.py              # Package entry point
│       ├── select_pca.py            # PCA-based feature selection
│       ├── select_ttest.py          # T-test-based feature selection
│       └── pca_feature_select.py    # Core functions (pca_extract, con_extract)
├── examples/
│   ├── run_pca.py                  # PCA method example
│   └── run_ttest.py                # T-test method example
├── tests/                          # Test directory
└── README.md
```

## Output

### PCA Method

#### Console Output

The function prints:
- Indices of principal components significantly associated with the target
- Number of selected connections

#### Plots (if `fig_plot=True`)

- `{target_name}.png` and `{target_name}.svg`: Bar plot showing statistics for the top 20 principal components

### T-test Method

#### Console Output

The function prints:
- Number of selected features
- Group sizes (control and case)
- Output file path (if `save_results=True`)

#### CSV Files (if `save_results=True`)

- `{output_prefix}_{method}_selected_features.csv`: Contains selected feature names (e.g., 'con1', 'con2', ...)

## Notes

- **NaN Handling**: NaN values must be excluded from both `df_X_train` and `target` before calling the function. The function does not handle NaN values internally.

- **Data Alignment**: Ensure that `df_X_train` and `target` are properly aligned (same samples in the same order).

- **High-Dimensional Data**: This pipeline is designed for high-dimensional data (e.g., thousands of features). For very large datasets, consider using parallel processing (already implemented internally for PCA method).

- **Multiple Comparison Correction**: The default FDR-BH method is recommended for most applications as it provides a good balance between statistical power and false positive control.

- **Choosing Between Methods**:
  - **Use PCA method** when: You want dimensionality reduction, have continuous/categorical targets, or want to capture multivariate patterns
  - **Use T-test method** when: You have a binary target (case vs. control), want direct feature-level comparisons, or need interpretable feature selection

## Citation

If you use this pipeline in your research, please cite the original paper (to be added).

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact information].

## Changelog

### Version 1.1.0
- Added t-test-based feature selection method
- Support for direct binary group comparisons
- CSV output for selected features

### Version 1.0.0
- Initial release
- PCA-based feature selection
- Support for binary, continuous, and categorical targets
- Automatic statistical test selection
- Multiple comparison correction methods
- Plotting functionality

