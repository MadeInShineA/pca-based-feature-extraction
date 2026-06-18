import pandas as pd
import numpy as np
import os
from pcafeat import select_pca_features

np.random.seed(42)
n_samples = 100
n_features = 1000

df_X_train = pd.DataFrame(np.random.randn(n_samples, n_features))
df_X_train.columns = [f'feature_{i}' for i in range(n_features)]

target = pd.Series(np.random.randint(0, 2, n_samples), name='diagnosis')

if pd.api.types.is_numeric_dtype(target):
    use_sub = ~np.isnan(target)
else:
    use_sub = ~target.isna()

df_X_train_clean = df_X_train[use_sub].reset_index(drop=True)
target_clean = target[use_sub].reset_index(drop=True)

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

print("Running PCA-based feature selection...")
cons, cons_pc, df_score = select_pca_features(
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
