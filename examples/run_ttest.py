import pandas as pd
import numpy as np
import os
from pcafeat import select_ttest_features

np.random.seed(42)
n_samples = 100
n_features = 1000

df_X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
df_X_train.columns = [f'feature_{i}' for i in range(n_features)]

df_X_train.iloc[:50, :10] += 0.5
df_X_train.iloc[50:, :10] -= 0.5

target = pd.Series([0]*50 + [1]*50, name='diagnosis')

if pd.api.types.is_numeric_dtype(target):
    use_sub = ~np.isnan(target)
else:
    use_sub = ~target.isna()

df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
target_clean = target[use_sub].reset_index(drop=True)

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

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
